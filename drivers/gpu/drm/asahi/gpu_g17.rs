// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! G17 (Apple M5 / T6050) top-level GPU manager and firmware bring-up.
//!
//! This is the G17-specific `gpu::GpuManager`. Unlike the `#[versions(AGX)]`
//! `GpuManager::ver` (G13/G14), G17 boots a **new firmware ABI**: RTKit
//! 3514.0.7 with four endpoints and a 3-level init_data tree. This module wires
//! the reverse-engineered, host-verified building blocks together:
//!
//!   * identity / UAT geometry  -> [`crate::hw::t6050`], [`crate::pgtable_g17`]
//!   * init_data layout+builder -> [`crate::fw::initdata_g17`]
//!   * RTKit + kick codecs      -> [`crate::transport_g17`]
//!
//! and drives the ordered boot of
//! `research/gpu/G17_HOST_DRIVER_SEQUENCE_2026-07-24.md`:
//!
//!   1. create the UAT (GPU MMU; no DART) and the kernel allocators,
//!   2. build BOTH roles' init_data roots (GFX role 0, then GFX1 role 1),
//!   3. bring up RTKit (EP0 handshake -> version 0xc -> EPMAP),
//!   4. hand off the init_data root on EP 0x20 (0x81-tagged, 44-bit VA),
//!   5. wait for the firmware "started" signal.
//!
//! HONESTY / SCOPE. The ABI/UAT/transport code this module calls is byte-exact
//! and host-verified. This glue module is **structural**: it compiles only
//! within the full kernel tree (it depends on `mmu::Uat`, `alloc`, `gem`,
//! `apple-rtkit`) and cannot be host-built here. The command-submission half of
//! the `GpuManager` trait needs the full firmware object graph (work queues,
//! channels, buffer manager), which is NOT yet ported for the G17 ABI; those
//! methods are honest `FIXME(on-device)` stubs so the first bring-up checkpoint
//! (does the firmware ACK the init_data doorbell?) can be reached and observed.

use core::any::Any;
use core::mem::{offset_of, size_of};
use core::ops::Range;
use core::sync::atomic::{AtomicBool, Ordering};

use kernel::{
    c_str,
    new_mutex,
    prelude::*,
    soc::apple::rtkit,
    sync::{
        lock::{mutex::MutexBackend, Guard},
        Arc, Mutex,
    },
    types::ForeignOwnable,
};

use crate::driver::{AsahiDevRef, AsahiDevice};
use crate::fw::channels::ChannelErrorType;
use crate::fw::initdata_g17 as g17id;
use crate::gpu::{GpuManager, KernelAllocators, RtkitObject, SequenceIDs};
use crate::transport_g17 as g17tp;
use crate::{alloc, event, gem, hw, mmu, queue, regs};

// Kernel-half VA ranges for the G17 firmware structures. Distinct, non-
// overlapping windows inside the TTBR1 kernel half (0xffffff8000000000..).
const IOVA_G17_INITDATA_RANGE: Range<u64> = 0xffffffb000000000..0xffffffb040000000;
const IOVA_G17_RTKIT_RANGE: Range<u64> = 0xffffffb040000000..0xffffffb080000000;
const IOVA_G17_KERN_PRIV_RANGE: Range<u64> = 0xffffffb100000000..0xffffffb300000000;
// Base of the four 8 GiB shared/gpu allocator pools (see make_allocators).
const IOVA_G17_KERN_SHARED_BASE: u64 = 0xffffffb400000000;

/// GPU virtual address of one role's init_data root (sent on EP 0x20).
#[derive(Copy, Clone, Default)]
struct RoleAlloc {
    root_va: u64,
}

/// Top-level G17 GPU manager.
#[pin_data]
pub(crate) struct GpuManagerG17 {
    dev: AsahiDevRef,
    cfg: &'static hw::HwConfig,
    dyncfg: hw::DynConfig,
    uat: KBox<mmu::Uat>,
    crashed: AtomicBool,
    #[pin]
    alloc: Mutex<KernelAllocators>,
    #[pin]
    rtkit: Mutex<Option<rtkit::RtKit<GpuManagerG17>>>,
    ids: SequenceIDs,
    #[allow(dead_code)]
    event_manager: Arc<event::EventManager>,
    /// Per-role init_data root VAs (index 0 = GFX, 1 = GFX1).
    role: [RoleAlloc; g17id::G17_FW_ROLE_COUNT],
    /// GEM objects backing the init_data, kept alive for the manager's lifetime.
    #[allow(dead_code)]
    initdata_objs: KVec<gem::ObjectRef>,
    #[allow(dead_code)]
    initdata_maps: KVec<mmu::KernelMapping>,
}

unsafe impl Send for GpuManagerG17 {}
unsafe impl Sync for GpuManagerG17 {}

#[vtable]
impl rtkit::Operations for GpuManagerG17 {
    type Data = Arc<GpuManagerG17>;
    type Buffer = RtkitObject;

    fn recv_message(data: <Self::Data as ForeignOwnable>::Borrowed<'_>, ep: u8, msg: u64) {
        let dev = &data.dev;
        // On G17 the runtime traffic rides EP 0x21 (gfx_ints); EP 0x20 delivers
        // the init handoff and the RX doorbell. Decode the runtime type field for
        // observability during bring-up.
        // FIXME(on-device): wire the EP 0x21 event-ring poll once the firmware
        // event ring (AGFIFirmwareEventRingEntry) layout is ported.
        if ep == g17tp::G17_EP_GFX_INTS {
            let ty = g17tp::g17_gfx_int_type(msg);
            dev_info!(
                dev.as_ref(),
                "G17: EP 0x21 runtime msg type={:#x} valid={}\n",
                ty,
                g17tp::g17_gfx_int_type_valid(ty)
            );
        } else {
            dev_info!(dev.as_ref(), "G17: RtKit msg {:#x}:{:#x}\n", ep, msg);
        }
    }

    fn crashed(data: <Self::Data as ForeignOwnable>::Borrowed<'_>, _crashlog: Option<&[u8]>) {
        let dev = &data.dev;
        data.crashed.store(true, Ordering::Relaxed);
        dev_err!(dev.as_ref(), "G17: GPU firmware crashed\n");
    }

    fn shmem_alloc(
        data: <Self::Data as ForeignOwnable>::Borrowed<'_>,
        size: usize,
    ) -> Result<Self::Buffer> {
        let dev = &data.dev;
        let mut obj = gem::new_kernel_object(dev, size)?;
        let vmap = obj.gem.owned_vmap()?;
        let mapping = obj.map_into_range(
            data.uat.kernel_vm(),
            IOVA_G17_RTKIT_RANGE,
            mmu::UAT_PGSZ as u64,
            mmu::PROT_FW_SHARED_RW,
            true,
        )?;
        Ok(RtkitObject { vmap, mapping })
    }
}

impl GpuManagerG17 {
    /// Create a new G17 GpuManager and build (but do not yet boot) the firmware
    /// init_data for both roles.
    #[inline(never)]
    pub(crate) fn new(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
    ) -> Result<Arc<GpuManagerG17>> {
        // The G17 GPU MMU is the single-stage UAT; there is NO separate DART.
        // The generic Uat already builds 16 KiB / 3-level tables at cfg.uat_oas
        // (=42 for T6050), which is format-compatible with the G17 encoders in
        // pgtable_g17. map_kernel_to_user is false for G17.
        //
        // FIXME(on-device): on G17 the host `configureMMUForBoot` is a no-op and
        // the boot page-table pointer (BPTP/SPTP) is owned by SPTM/firmware
        // (G17_BPTP_SPTP_OWNER_BOUNDARY). The generic Uat's PPL "handoff" dekker
        // handshake (mmu::Handoff, PPL_MAGIC) models the G13/G14 uPPL and may not
        // apply to the G17 SPTM boundary. If Handoff::init() times out on real
        // hardware, gate it out for G17 and rely on the SPTM-published tables.
        let uat = KBox::new(mmu::Uat::new(dev, cfg, false)?, GFP_KERNEL)?;

        let dyncfg = Self::make_dyncfg(dev, res, cfg, &uat)?;

        let mut alloc = Self::make_allocators(dev, &uat)?;
        let event_manager = Arc::new(event::EventManager::new(&mut alloc)?, GFP_KERNEL)?;

        // Build the init_data configuration from the dynamic (DT) perf states.
        let idcfg = Self::make_initdata_cfg(&dyncfg)?;

        // Allocate + populate both roles' init_data in GPU-visible memory.
        let mut initdata_objs = KVec::new();
        let mut initdata_maps = KVec::new();
        let role = Self::build_and_map_initdata(
            dev,
            &uat,
            &idcfg,
            &mut initdata_objs,
            &mut initdata_maps,
        )?;

        let mgr = Arc::pin_init(
            pin_init!(GpuManagerG17 {
                dev: dev.into(),
                cfg,
                dyncfg: KBox::<hw::DynConfig>::into_inner(dyncfg),
                uat,
                crashed: AtomicBool::new(false),
                alloc <- new_mutex!(alloc, "g17_alloc"),
                rtkit <- new_mutex!(None, "g17_rtkit"),
                ids: Default::default(),
                event_manager,
                role,
                initdata_objs,
                initdata_maps,
            }),
            GFP_KERNEL,
        )?;

        let rtkit = rtkit::RtKit::<GpuManagerG17>::new(dev.as_ref(), None, 0, mgr.clone())?;
        *mgr.rtkit.lock() = Some(rtkit);

        Ok(mgr)
    }

    /// Fetch/validate the dynamic configuration (hardware ID + DT power).
    #[inline(never)]
    fn make_dyncfg(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
        uat: &mmu::Uat,
    ) -> Result<KBox<hw::DynConfig>> {
        let gpu_id = res.get_gpu_id()?;

        dev_info!(dev.as_ref(), "G17 GPU: {:?}{:?}\n", gpu_id.gpu_gen, gpu_id.gpu_variant);
        dev_info!(
            dev.as_ref(),
            "  Clusters: {}  Cores: {}\n",
            gpu_id.num_clusters,
            gpu_id.num_cores * gpu_id.num_clusters
        );

        if gpu_id.gpu_gen != cfg.gpu_gen {
            dev_err!(dev.as_ref(), "G17: GPU gen mismatch\n");
            return Err(EIO);
        }

        let pwr = hw::PwrConfig::load(dev, cfg)?;
        let fwnode = dev.as_ref().fwnode().ok_or(ENOENT)?;

        Ok(KBox::new(
            hw::DynConfig {
                pwr,
                uat_ttb_base: uat.ttb_base(),
                id: gpu_id,
                firmware_version: fwnode
                    .property_read_array_vec(c_str!("apple,firmware-version"), 3)?
                    .or(kernel::kvec![0; 3]?),
                hw_data_a: KVVec::new(),
                hw_data_b: KVVec::new(),
                hw_globals: KVVec::new(),
            },
            GFP_KERNEL,
        )?)
    }

    /// Build the kernel allocators over the G17 kernel VA ranges.
    ///
    /// FIXME(on-device): sizes/flags/prot are copied from the G13/G14
    /// KernelAllocators (gpu.rs); confirm the G17 firmware's cacheability and
    /// per-pool layout once the init_data handshake is observed. Distinct sub-
    /// ranges are carved out from IOVA_G17_KERN_SHARED_BASE so the five pools do
    /// not overlap.
    #[inline(never)]
    fn make_allocators(dev: &AsahiDevice, uat: &mmu::Uat) -> Result<KernelAllocators> {
        let vm = uat.kernel_vm();
        let base = IOVA_G17_KERN_SHARED_BASE;
        let step = 0x2_0000_0000u64; // 8 GiB per pool
        let pool = |i: u64| -> Range<u64> { (base + i * step)..(base + (i + 1) * step) };

        Ok(KernelAllocators {
            private: alloc::DefaultAllocator::new(
                dev, vm, IOVA_G17_KERN_PRIV_RANGE, 0x80, mmu::PROT_FW_PRIV_RW,
                1024 * 1024, true, fmt!("G17 Kernel Private"), true,
            )?,
            shared: alloc::DefaultAllocator::new(
                dev, vm, pool(0), 0x80, mmu::PROT_FW_SHARED_RW,
                1024 * 1024, true, fmt!("G17 Kernel Shared"), false,
            )?,
            shared_ro: alloc::DefaultAllocator::new(
                dev, vm, pool(1), 0x80, mmu::PROT_FW_SHARED_RO,
                64 * 1024, true, fmt!("G17 Kernel RO Shared"), false,
            )?,
            gpu: alloc::DefaultAllocator::new(
                dev, vm, pool(2), 0x80, mmu::PROT_GPU_FW_SHARED_RW,
                64 * 1024, true, fmt!("G17 Kernel GPU Shared"), false,
            )?,
            gpu_ro: alloc::DefaultAllocator::new(
                dev, vm, pool(3), 0x80, mmu::PROT_GPU_RO_FW_PRIV_RW,
                1024 * 1024, true, fmt!("G17 Kernel GPU RO Shared"), true,
            )?,
        })
    }

    /// Assemble the init_data builder config from the DT-loaded perf states.
    fn make_initdata_cfg(dyncfg: &hw::DynConfig) -> Result<g17id::G17InitDataCfg> {
        let states = &dyncfg.pwr.perf_states;
        let count = states.len().min(g17id::G17_PERF_STATE_MAX);
        if count == 0 {
            return Err(EINVAL);
        }
        let mut mhz = [0u32; g17id::G17_PERF_STATE_MAX];
        for (i, s) in states.iter().take(count).enumerate() {
            // hw+0x10bc is MHz: the host divides Hz by 1e6.
            mhz[i] = s.freq_hz / 1_000_000;
        }
        Ok(g17id::G17InitDataCfg {
            // VAs are filled in build_and_map_initdata once memory is allocated.
            va: Default::default(),
            perf_state_count: count as u32,
            perf_state_clock_mhz: mhz,
        })
    }

    /// Allocate GPU-visible memory for both roles' init_data, populate it via the
    /// verified builder, and return the per-role root VAs.
    #[inline(never)]
    fn build_and_map_initdata(
        dev: &AsahiDevice,
        uat: &mmu::Uat,
        base_cfg: &g17id::G17InitDataCfg,
        objs: &mut KVec<gem::ObjectRef>,
        maps: &mut KVec<mmu::KernelMapping>,
        ) -> Result<[RoleAlloc; g17id::G17_FW_ROLE_COUNT]> {
        // One shared allocation (root+0x20 ptr_shared, identical for both roles).
        let (shared_va, shared_obj, shared_map) = Self::alloc_gpu_buf(dev, uat, mmu::UAT_PGSZ)?;
        objs.push(shared_obj, GFP_KERNEL)?;
        maps.push(shared_map, GFP_KERNEL)?;

        let mut cfg = *base_cfg;
        let mut role_alloc: [RoleAlloc; g17id::G17_FW_ROLE_COUNT] = Default::default();

        // Per role: one contiguous allocation holding {root, rt, hw}. The three
        // GPU VAs are the allocation base plus the offset of each level within
        // G17InitDataRole. Auxiliary pointers (ptr_role, fw_va_shared_alloc,
        // cross_role) point at the shared allocation so nothing dereferences a
        // null/unmapped VA during the handshake.
        // FIXME(on-device): ptr_role / cross_role should point at their real
        // per-role firmware objects (root+0xb0/0xb8/0xc0 peer objects are
        // UNKNOWN-C). Pointing them at a valid mapped buffer is enough to pass
        // the firmware's magic + host-mapped-alloc gates and reach the first
        // ACK checkpoint, but not enough for steady-state operation.
        let role_sz = size_of::<g17id::G17InitDataRole>();
        for r in 0..g17id::G17_FW_ROLE_COUNT {
            let (base_va, obj, map) = Self::alloc_gpu_buf(dev, uat, role_sz)?;
            let rt_va = base_va + offset_of!(g17id::G17InitDataRole, rt) as u64;
            let hw_va = base_va + offset_of!(g17id::G17InitDataRole, hw) as u64;

            cfg.va[r] = g17id::G17RoleVa {
                root: base_va,
                runtime_pointers: rt_va,
                hw_globals: hw_va,
                fw_va_shared_alloc: shared_va,
                ptr_shared: shared_va,
                ptr_role: shared_va,
                cross_role: shared_va,
                cross_role_2: if r == g17id::G17_FW_ROLE_GFX1 as usize {
                    shared_va
                } else {
                    0
                },
            };
            objs.push(obj, GFP_KERNEL)?;
            maps.push(map, GFP_KERNEL)?;
            role_alloc[r].root_va = base_va;
        }

        // Populate both roles in a stack buffer, then copy into the GPU objects.
        // build_pair enforces GFX-before-GFX1 population and the cross-role rules.
        let mut roles: KBox<[g17id::G17InitDataRole; g17id::G17_FW_ROLE_COUNT]> =
            KBox::new([Default::default(), Default::default()], GFP_KERNEL)?;
        g17id::build_pair(&mut roles, &cfg)?;

        for r in 0..g17id::G17_FW_ROLE_COUNT {
            // objs[1 + r] is this role's allocation (index 0 is the shared buf).
            let role_bytes = unsafe {
                core::slice::from_raw_parts(
                    &roles[r] as *const _ as *const u8,
                    role_sz,
                )
            };
            Self::write_gpu_buf(&mut objs[1 + r], role_bytes)?;
            dev_info!(
                dev.as_ref(),
                "G17: init_data role {} built, root VA {:#x}\n",
                r,
                role_alloc[r].root_va
            );
        }

        Ok(role_alloc)
    }

    /// Allocate a kernel GEM object, map it into the G17 init_data VA range, and
    /// return (gpu_va, object, mapping).
    fn alloc_gpu_buf(
        dev: &AsahiDevice,
        uat: &mmu::Uat,
        size: usize,
    ) -> Result<(u64, gem::ObjectRef, mmu::KernelMapping)> {
        let mut obj = gem::new_kernel_object(dev, size)?;
        let mapping = obj.map_into_range(
            uat.kernel_vm(),
            IOVA_G17_INITDATA_RANGE,
            mmu::UAT_PGSZ as u64,
            mmu::PROT_GPU_FW_SHARED_RW,
            true,
        )?;
        let va = mapping.iova();
        Ok((va, obj, mapping))
    }

    /// Copy bytes into a kernel GEM object's CPU mapping.
    fn write_gpu_buf(obj: &mut gem::ObjectRef, bytes: &[u8]) -> Result {
        let vmap = obj.gem.owned_vmap()?;
        let mut dst = vmap.get();
        // FIXME(on-device): confirm the IoSysMap write helper on the target
        // kernel. The init_data structs are plain little-endian bytes; a straight
        // copy into the WB kernel mapping is correct on Apple ARM64.
        dst.as_mut_slice()
            .get_mut(..bytes.len())
            .ok_or(EINVAL)?
            .copy_from_slice(bytes);
        Ok(())
    }
}

impl GpuManager for GpuManagerG17 {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn arc_as_any(self: Arc<Self>) -> Arc<dyn Any + Sync + Send> {
        self as Arc<dyn Any + Sync + Send>
    }

    /// The core deliverable: attempt real firmware bring-up.
    fn init(&self) -> Result {
        let root_gfx = self.role[g17id::G17_FW_ROLE_GFX as usize].root_va;
        let root_gfx1 = self.role[g17id::G17_FW_ROLE_GFX1 as usize].root_va;

        // Encode the init_data doorbell now so a bad VA fails before we touch the
        // firmware. (44-bit field; the host bfxil would silently truncate.)
        let msg_gfx = g17tp::g17_initdata_msg_encode(root_gfx)?;
        let _msg_gfx1 = g17tp::g17_initdata_msg_encode(root_gfx1)?;

        let mut guard = self.rtkit.lock();
        let mut rtk = guard.as_mut().as_pin_mut().unwrap();

        // EP0 management handshake (HELLO -> version 0xc -> EPMAP) is performed
        // by apple-rtkit's boot(). The G17 firmware advertises exactly
        // {0x00,0x01,0x20,0x21}; see transport_g17::g17_epmap_blocks_for_g17.
        rtk.as_mut().boot()?;
        rtk.as_mut().start_endpoint(g17tp::G17_EP_GFX_MSGS)?; // 0x20
        rtk.as_mut().start_endpoint(g17tp::G17_EP_GFX_INTS)?; // 0x21

        // Hand off the GFX (role 0) init_data root on EP 0x20. This is the FIRST
        // on-device checkpoint: does the firmware validate the root magic
        // (0x0C8DE339072804C0) + the host-mapped-alloc gate and ACK?
        dev_info!(
            self.dev.as_ref(),
            "G17: sending init_data doorbell (GFX) {:#x}\n",
            msg_gfx
        );
        rtk.as_mut().send_message(g17tp::G17_EP_GFX_MSGS, msg_gfx)?;

        // FIXME(on-device): GFX1 (role 1) is the power core and must also be
        // booted (GFX then GFX1 per G17_INITDATA_AND_BOOT_ORDER §3). It is NOT
        // known whether GFX1 is a second ASC (a second apple-rtkit mailbox / DT
        // node) or shares this one. If it is a second ASC, allocate a second
        // RtKit here from its DT mailbox and send `_msg_gfx1` on its EP 0x20.
        // Both roots are already fully populated, so either doorbell order is
        // legal on the wire.

        core::mem::drop(guard);
        Ok(())
    }

    fn update_globals(&self) {
        // FIXME(on-device): the G17 globals live in the hw_globals init_data
        // region; per-field update requires the hw_globals scalar map (UNKNOWN-C).
    }

    fn alloc(&self) -> Guard<'_, KernelAllocators, MutexBackend> {
        self.alloc.lock()
    }

    fn new_vm(&self, kernel_range: Range<u64>) -> Result<mmu::Vm> {
        self.uat.new_vm(self.ids.vm.next(), kernel_range)
    }

    fn bind_vm(&self, vm: &mmu::Vm) -> Result<mmu::VmBind> {
        self.uat.bind(vm)
    }

    fn new_queue(
        &self,
        _vm: mmu::Vm,
        _ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
        _ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
        _priority: u32,
        _usc_exec_base: u64,
    ) -> Result<KBox<dyn queue::Queue>> {
        // FIXME(on-device): command submission needs the G17 firmware object
        // graph (KSMKickQueue geometry from accel+0x11208, work queues, channels,
        // buffer manager) which is not yet ported. The kick DOORBELL encoding is
        // ready (transport_g17::g17_kick_encode); the kick ENTRY ring is not.
        dev_err!(self.dev.as_ref(), "G17: queue creation not yet implemented\n");
        Err(ENOTSUPP)
    }

    fn ids(&self) -> &SequenceIDs {
        &self.ids
    }

    fn kick_firmware(&self) -> Result {
        if self.is_crashed() {
            return Err(ENODEV);
        }
        // FIXME(on-device): the steady-state firmware kick/wake message on EP
        // 0x21 for G17 (equivalent of MSG_TX_DOORBELL|DOORBELL_KICKFW) is not
        // recovered. The SKSM per-queue kick doorbell (g17_kick_encode) is the
        // work-submission doorbell, not the global wake. Left as a no-op until an
        // EP 0x21 trace resolves the wake message.
        Ok(())
    }

    fn flush_fw_cache(&self) -> Result {
        // FIXME(on-device): device-control channel (cache flush) not yet ported
        // for the G17 ABI.
        Ok(())
    }

    fn handle_timeout(&self, counter: u32, event_slot: i32, unk: u32) {
        dev_err!(
            self.dev.as_ref(),
            "G17: GPU timeout (slot {}, count {}, unk {})\n",
            event_slot,
            counter,
            unk
        );
    }

    fn handle_fault(&self) {
        let info = (*self.dev).resources.get_fault_info(self.cfg);
        dev_err!(self.dev.as_ref(), "G17: GPU fault: {:#x?}\n", info);
    }

    fn handle_channel_error(
        &self,
        error_type: ChannelErrorType,
        pipe_type: u32,
        event_slot: u32,
        event_value: u32,
    ) {
        dev_err!(
            self.dev.as_ref(),
            "G17: channel error {:?} pipe {} slot {} value {:#x}\n",
            error_type,
            pipe_type,
            event_slot,
            event_value
        );
    }

    fn ack_grow(&self, _buffer_slot: u32, _vm_slot: u32, _counter: u32) {
        // FIXME(on-device): TVB grow ack needs the device-control channel.
    }

    fn fwctl(&self, _msg: crate::fw::channels::FwCtlMsg) -> Result {
        // FIXME(on-device): firmware-control channel not yet ported for G17.
        Err(ENOTSUPP)
    }

    fn get_cfg(&self) -> &'static hw::HwConfig {
        self.cfg
    }

    fn get_dyncfg(&self) -> &hw::DynConfig {
        &self.dyncfg
    }

    fn free_context(&self, _data: KBox<crate::fw::types::GpuObject<crate::fw::workqueue::GpuContextData>>) {
        // No-op until the submission path is ported.
    }

    fn is_crashed(&self) -> bool {
        self.crashed.load(Ordering::Relaxed)
    }

    fn map_timestamp_buffer(
        &self,
        _bo: gem::ObjectRef,
        _range: Range<usize>,
    ) -> Result<mmu::KernelMapping> {
        Err(ENOTSUPP)
    }
}
