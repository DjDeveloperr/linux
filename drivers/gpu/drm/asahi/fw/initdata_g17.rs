// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! G17 (Apple M5 / T6050) GPU init_data — the three-level host-supplied ABI.
//!
//! This is the G17 firmware's own init_data grammar. It is deliberately NOT the
//! `#[versions(AGX)]` `InitData::ver` structure used by G13/G14: the G17
//! firmware (RTKit 3514.0.7, "g17s a0") replaced that single blob with a
//! 3-level pointer tree, and none of the G13/G14 field offsets carry over.
//!
//!   level 0  [`G17InitDataRoot`]  0x0c8 B  (root VA is what is sent on EP 0x20)
//!   level 1  [`G17RtPointers`]    0x4c0 B  (root+0x18 -> FW global 0x104888)
//!   level 2  [`G17HwGlobals`]     0x2828 B (runtime_pointers+0x00)
//!
//! Every offset that the RE names is a named member; every byte the RE does not
//! name is a `field_0xNNN` reserved array — never renamed, never reinterpreted.
//! Sizes and named offsets are pinned with `static_assert!`, exactly mirroring
//! the `_Static_assert`s proven in the host scaffold
//! (`research/gpu/implementation/g17_initdata.h`), which builds and passes under
//! `-Wall -Wextra -Werror -pedantic` and ASan/UBSan.
//!
//! Sources:
//!   [P2] research/gpu/G17_INITDATA_AND_BOOT_ORDER_2026-07-23.md §2.1-2.3, §3
//!   [P1] research/gpu/G17_FIRMWARE_BOOT_PROTOCOL_2026-07-23.md §4
//!
//! These are DMA-visible little-endian structures. Apple ARM64 is
//! little-endian, so plain field access matches the wire layout. The
//! runtime_pointers level is `#[repr(C, packed)]` because [P2] proves two
//! unaligned pointer groups (`rt+0x254..0x274` on a 4-byte boundary and
//! `rt+0x481` on an odd one).

use core::mem::{offset_of, size_of};

use kernel::prelude::*;
use kernel::static_assert;

/// [VERIFIED-A] root magic, compared by the firmware @0x212dc.
pub(crate) const G17_INITDATA_MAGIC: u64 = 0x0C8DE339072804C0;

/// [VERIFIED-A/B] exact structure sizes ([P2] §2).
pub(crate) const G17_INITDATA_ROOT_SIZE: usize = 0x0c8;
pub(crate) const G17_RT_POINTERS_SIZE: usize = 0x4c0;
pub(crate) const G17_HW_GLOBALS_SIZE: usize = 0x2828;

/// [VERIFIED-A] AGXK_KEXT_FW_ROLE_STRS = {"GFX","GFX1"} @0xfffffe000714a0b0.
pub(crate) const G17_FW_ROLE_GFX: u32 = 0;
pub(crate) const G17_FW_ROLE_GFX1: u32 = 1;
pub(crate) const G17_FW_ROLE_COUNT: usize = 2;

/// [VERIFIED-A] hw/globals perf tables are 16 entries wide ([P2] §2.3).
pub(crate) const G17_PERF_STATE_MAX: usize = 16;

/// [VERIFIED-A] the init_data doorbell address field is 44 bits
/// ([P2] §4 correction; FW `orr …,#0xfffff00000000000`). A GPU VA handed to the
/// firmware must be representable there: bits[63:44] must be all-zero or exactly
/// the firmware-applied prefix.
const G17_INITDATA_ADDR_MASK: u64 = 0x0000_0fff_ffff_ffff;
const G17_INITDATA_ADDR_PREFIX: u64 = 0xfffff000_00000000;

/// Returns true iff `va` fits the 44-bit init_data doorbell field.
const fn va_fits_44(va: u64) -> bool {
    let high = va & !G17_INITDATA_ADDR_MASK;
    high == 0 || high == G17_INITDATA_ADDR_PREFIX
}

// ---------------------------------------------------------------------------
// Level 2 — hw/globals, 0x2828 B ([P2] §2.3).
// ---------------------------------------------------------------------------

/// G17 hw/globals: the 0x2828-byte struct pointed to by runtime_pointers+0x00.
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub(crate) struct G17HwGlobals {
    /// 0x0000 128-bit block copied to FW global +0xab0 (FW @0x21394).
    pub(crate) block_0x0000: [u8; 0x10],
    field_0x0010: [u8; 0xf78],
    /// 0x0f88 128-bit device/HAL id block (host @b3d7b4..b3d7b8).
    pub(crate) block_dev_hal_id: [u8; 0x10],
    /// 0x0f98..0x10a7 ~40 scalar config words; individual offsets are recorded
    /// in fw-re/g17-initdata-hwglobals-fw-reads.txt but are not enumerated in
    /// [P2], so the span stays reserved.
    /// FIXME(on-device): map each host `ubfx` of accel+0x6b0 to its DT property.
    region_scalar_cfg: [u8; 0x110],
    field_0x10a8: [u8; 0x10],
    /// 0x10b8 perf-state[0] clock, MHz (host @b12c90).
    pub(crate) perf_state0_clock_mhz: u32,
    /// 0x10bc u32[16] perf-state clock frequency, MHz (host @b12ca0).
    pub(crate) perf_state_clock_mhz: [u32; G17_PERF_STATE_MAX],
    /// 0x10fc / 0x14fc: 16x16 u32, row stride 0x40, cols[0],[1] consumed.
    /// FIXME(on-device): column meaning of the two 16x16 tables is UNKNOWN-C.
    pub(crate) table_a: [[u32; G17_PERF_STATE_MAX]; G17_PERF_STATE_MAX],
    pub(crate) table_b: [[u32; G17_PERF_STATE_MAX]; G17_PERF_STATE_MAX],
    /// 0x18fc second MHz table (host-written, FW-unread here).
    pub(crate) clock_mhz_table2: [u32; G17_PERF_STATE_MAX],
    /// 0x193c / 0x197c f32[16]: IEEE-754 binary32 bit patterns, kept as u32 so a
    /// port never touches the FPU.
    pub(crate) perf_f32_a: [u32; G17_PERF_STATE_MAX],
    pub(crate) perf_f32_b: [u32; G17_PERF_STATE_MAX],
    pub(crate) perf_u32_a: [u32; G17_PERF_STATE_MAX], // 0x19bc
    pub(crate) perf_u32_b: [u32; G17_PERF_STATE_MAX], // 0x19fc
    pub(crate) perf_u32_c: [u32; G17_PERF_STATE_MAX], // 0x1a3c
    field_0x1a7c: [u8; 0xc0],
    /// 0x1b3c perf_state_count-1 ; 0x1b40 [this+0xf18].
    pub(crate) perf_state_count_minus_1: u32,
    field_0x1b40: u32,
    field_0x1b44: [u8; 0x140],
    /// 0x1c84 count-1 ; 0x1c88 [this+0xf1c] ; 0x1c8c u32[16]. 0x1c84 is also the
    /// base of a 0x148-byte block memmove'd to FW global +0x410 (FW @0x2183c).
    pub(crate) count2_minus_1: u32,
    field_0x1c88: u32,
    pub(crate) perf_u32_d: [u32; G17_PERF_STATE_MAX],
    field_0x1ccc: [u8; 0x968],
    /// 0x2634..0x26ac scheduler/DM config scalars (FW @218bc-219dc).
    /// FIXME(on-device): individual scalar semantics UNKNOWN-C.
    region_sched_dm_cfg: [u8; 0x7c],
    field_0x26b0: [u8; 0x50],
    /// 0x2700..0x2710 ring/queue params: u32,u32,u64,u32.
    pub(crate) ring_param_0: u32,
    pub(crate) ring_param_1: u32,
    pub(crate) ring_param_2: u64,
    pub(crate) ring_param_3: u32,
    field_0x2714: [u8; 0xa8],
    /// 0x27bc..0x27f0 four u32 timer/clock params; not enumerated in [P2].
    region_timer_clock: [u8; 0x38],
    pub(crate) field_0x27f4: u32, // [accel+0x6c8]
    pub(crate) field_0x27f8: u32,
    pub(crate) flag_0x27fc: u8, // [accel+0x6d0], FW @0x219fc
    pub(crate) flag_0x27fd: u8, // [accel+0x6d8], FW @0x21a10
    field_0x27fe: [u8; 0x2a],
}

/// 0x148-byte block at hw+0x1c84 memmove'd to FW global +0x410.
pub(crate) const G17_HW_MEMMOVE_BLOCK_OFF: usize = 0x1c84;
pub(crate) const G17_HW_MEMMOVE_BLOCK_LEN: usize = 0x148;

static_assert!(size_of::<G17HwGlobals>() == G17_HW_GLOBALS_SIZE);
static_assert!(offset_of!(G17HwGlobals, block_0x0000) == 0x0000);
static_assert!(offset_of!(G17HwGlobals, block_dev_hal_id) == 0x0f88);
static_assert!(offset_of!(G17HwGlobals, region_scalar_cfg) == 0x0f98);
static_assert!(offset_of!(G17HwGlobals, perf_state0_clock_mhz) == 0x10b8);
static_assert!(offset_of!(G17HwGlobals, perf_state_clock_mhz) == 0x10bc);
static_assert!(offset_of!(G17HwGlobals, table_a) == 0x10fc);
static_assert!(offset_of!(G17HwGlobals, table_b) == 0x14fc);
static_assert!(offset_of!(G17HwGlobals, clock_mhz_table2) == 0x18fc);
static_assert!(offset_of!(G17HwGlobals, perf_f32_a) == 0x193c);
static_assert!(offset_of!(G17HwGlobals, perf_f32_b) == 0x197c);
static_assert!(offset_of!(G17HwGlobals, perf_u32_a) == 0x19bc);
static_assert!(offset_of!(G17HwGlobals, perf_u32_b) == 0x19fc);
static_assert!(offset_of!(G17HwGlobals, perf_u32_c) == 0x1a3c);
static_assert!(offset_of!(G17HwGlobals, perf_state_count_minus_1) == 0x1b3c);
static_assert!(offset_of!(G17HwGlobals, field_0x1b40) == 0x1b40);
static_assert!(offset_of!(G17HwGlobals, count2_minus_1) == 0x1c84);
static_assert!(offset_of!(G17HwGlobals, field_0x1c88) == 0x1c88);
static_assert!(offset_of!(G17HwGlobals, perf_u32_d) == 0x1c8c);
static_assert!(offset_of!(G17HwGlobals, region_sched_dm_cfg) == 0x2634);
static_assert!(offset_of!(G17HwGlobals, ring_param_0) == 0x2700);
static_assert!(offset_of!(G17HwGlobals, ring_param_1) == 0x2704);
static_assert!(offset_of!(G17HwGlobals, ring_param_2) == 0x2708);
static_assert!(offset_of!(G17HwGlobals, ring_param_3) == 0x2710);
static_assert!(offset_of!(G17HwGlobals, region_timer_clock) == 0x27bc);
static_assert!(offset_of!(G17HwGlobals, field_0x27f4) == 0x27f4);
static_assert!(offset_of!(G17HwGlobals, field_0x27f8) == 0x27f8);
static_assert!(offset_of!(G17HwGlobals, flag_0x27fc) == 0x27fc);
static_assert!(offset_of!(G17HwGlobals, flag_0x27fd) == 0x27fd);
static_assert!(G17_HW_MEMMOVE_BLOCK_OFF + G17_HW_MEMMOVE_BLOCK_LEN <= G17_HW_GLOBALS_SIZE);

impl Default for G17HwGlobals {
    fn default() -> Self {
        // SAFETY: every field is an integer or byte array, so the all-zero bit
        // pattern is a valid value.
        unsafe { core::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// Level 1 — runtime_pointers, 0x4c0 B ([P2] §2.2).
// ---------------------------------------------------------------------------

/// One channel/ring descriptor. 0x20 bytes; the firmware reads entry+0x10
/// (FW @0xff8c-0x10008). The rest of the entry is UNKNOWN-B, so it stays
/// byte-reserved.
#[repr(C)]
#[derive(Copy, Clone, Default)]
#[allow(dead_code)]
pub(crate) struct G17RtChanDesc {
    field_0x00: [u8; 0x10],
    /// FW-read word.
    pub(crate) field_0x10: [u8; 0x08],
    field_0x18: [u8; 0x08],
}
static_assert!(size_of::<G17RtChanDesc>() == 0x20);

/// [P2] §2.2 states BOTH "13 x 0x20" and the byte range "0x020..0x19f". Those
/// disagree: 0x020..0x19f holds exactly 12 entries, and a 13th would overlap the
/// cursor mirrors at rt+0x1a0/0x1a8 (separately VERIFIED-B with their own FW
/// read). The layout uses the byte range (12 entries).
/// FIXME(on-device): resolve the 12-vs-13 count with a live trace.
pub(crate) const G17_RT_CHAN_DESC_OFF: usize = 0x020;
pub(crate) const G17_RT_CHAN_DESC_COUNT_LAYOUT: usize = 12;

/// G17 runtime_pointers table.
#[repr(C, packed)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub(crate) struct G17RtPointers {
    /// 0x000 GPU VA of the 0x2828-byte hw/globals struct.
    pub(crate) hw_globals_gpu_va: u64,
    pub(crate) ptr_0x08: u64, // this+0x340
    pub(crate) ptr_0x10: u64, // this+0x348, 0x88 B alloc
    field_0x018: [u8; 0x08],
    /// 0x020..0x19f channel/ring descriptor array.
    pub(crate) chan_desc: [G17RtChanDesc; G17_RT_CHAN_DESC_COUNT_LAYOUT],
    pub(crate) cursor_mirror_0: u64, // 0x1a0 write-side
    pub(crate) cursor_mirror_1: u64, // 0x1a8 write-side
    pub(crate) cursor_source_0: u64, // 0x1b0 read-side
    pub(crate) cursor_source_1: u64, // 0x1b8 read-side
    pub(crate) table2_base: u64,     // 0x1c0
    pub(crate) field_0x1c8: u64,
    pub(crate) field_0x1d0: u64,
    field_0x1d8: [u8; 0x08],
    pub(crate) field_0x1e0: u64,
    pub(crate) field_0x1e8: u64,
    pub(crate) field_0x1f0: u64,
    pub(crate) field_0x1f8: u64,
    pub(crate) ptr_role: u64, // 0x200 this+0xaa0 / this+0xbd0
    field_0x208: [u8; 0x48],
    pub(crate) zeroed_0x250: u32, // 0x250 zeroed at init
    /// 0x254..0x27b five firmware buffers; UNALIGNED by 4 — this is why the
    /// struct is packed. Sizes from [`G17_FW_ALLOC_TABLE`].
    pub(crate) fw_buf_0x254: u64,
    pub(crate) fw_buf_0x25c: u64,
    pub(crate) fw_buf_0x264: u64,
    pub(crate) fw_buf_0x26c: u64,
    pub(crate) fw_buf_0x274: u64,
    field_0x27c: [u8; 0x54],
    pub(crate) ptr_fw_mapping_0x2d0: u64, // 0x2d0 createFirmwareMapping()
    field_0x2d8: [u8; 0x108],
    pub(crate) flag_0x3e0: u8, // 0x3e0 FW @0xdf7c
    field_0x3e1: [u8; 0xa0],
    /// 0x481 role-1 only, ODD offset (add x8,x8,#0x481 @b2af4c).
    pub(crate) ptr_role1_0x481: u64,
    field_0x489: [u8; 0x37],
}

static_assert!(size_of::<G17RtPointers>() == G17_RT_POINTERS_SIZE);
static_assert!(offset_of!(G17RtPointers, hw_globals_gpu_va) == 0x000);
static_assert!(offset_of!(G17RtPointers, ptr_0x08) == 0x008);
static_assert!(offset_of!(G17RtPointers, ptr_0x10) == 0x010);
static_assert!(offset_of!(G17RtPointers, chan_desc) == G17_RT_CHAN_DESC_OFF);
static_assert!(size_of::<[G17RtChanDesc; G17_RT_CHAN_DESC_COUNT_LAYOUT]>() == 0x180);
static_assert!(offset_of!(G17RtPointers, cursor_mirror_0) == 0x1a0);
static_assert!(offset_of!(G17RtPointers, cursor_mirror_1) == 0x1a8);
static_assert!(offset_of!(G17RtPointers, cursor_source_0) == 0x1b0);
static_assert!(offset_of!(G17RtPointers, cursor_source_1) == 0x1b8);
static_assert!(offset_of!(G17RtPointers, table2_base) == 0x1c0);
static_assert!(offset_of!(G17RtPointers, field_0x1c8) == 0x1c8);
static_assert!(offset_of!(G17RtPointers, field_0x1d0) == 0x1d0);
static_assert!(offset_of!(G17RtPointers, field_0x1e0) == 0x1e0);
static_assert!(offset_of!(G17RtPointers, field_0x1e8) == 0x1e8);
static_assert!(offset_of!(G17RtPointers, field_0x1f0) == 0x1f0);
static_assert!(offset_of!(G17RtPointers, field_0x1f8) == 0x1f8);
static_assert!(offset_of!(G17RtPointers, ptr_role) == 0x200);
static_assert!(offset_of!(G17RtPointers, zeroed_0x250) == 0x250);
static_assert!(offset_of!(G17RtPointers, fw_buf_0x254) == 0x254);
static_assert!(offset_of!(G17RtPointers, fw_buf_0x25c) == 0x25c);
static_assert!(offset_of!(G17RtPointers, fw_buf_0x264) == 0x264);
static_assert!(offset_of!(G17RtPointers, fw_buf_0x26c) == 0x26c);
static_assert!(offset_of!(G17RtPointers, fw_buf_0x274) == 0x274);
static_assert!(offset_of!(G17RtPointers, ptr_fw_mapping_0x2d0) == 0x2d0);
static_assert!(offset_of!(G17RtPointers, flag_0x3e0) == 0x3e0);
static_assert!(offset_of!(G17RtPointers, ptr_role1_0x481) == 0x481);

impl Default for G17RtPointers {
    fn default() -> Self {
        // SAFETY: all fields are integers or byte arrays; all-zero is valid.
        unsafe { core::mem::zeroed() }
    }
}

/// One entry of the firmware allocation table ([P2] §2.2): a buffer the host
/// allocates whose GPU VA lands at `rt+rt_off`. `role_only == 0xff` means both
/// roles; otherwise it is the only role that populates it.
#[derive(Copy, Clone, Debug)]
pub(crate) struct G17FwAlloc {
    pub(crate) rt_off: u32,
    pub(crate) size: u32,
    pub(crate) role_only: u8,
}

/// The 8 runtime-pointer-referenced firmware buffers ([P2] §2.2). Reproduced so
/// a port can size its allocations from the spec rather than a magic number.
pub(crate) const G17_FW_ALLOC_TABLE: [G17FwAlloc; 8] = [
    G17FwAlloc { rt_off: 0x000, size: 0x2828, role_only: 0xff }, // hw_globals
    G17FwAlloc { rt_off: 0x010, size: 0x0088, role_only: 0xff }, // rt+0x010
    G17FwAlloc { rt_off: 0x254, size: 0x0c18, role_only: 0xff },
    G17FwAlloc { rt_off: 0x25c, size: 0x1048, role_only: 0xff },
    G17FwAlloc { rt_off: 0x264, size: 0x0e10, role_only: 0xff },
    G17FwAlloc { rt_off: 0x26c, size: 0x0068, role_only: 0xff },
    G17FwAlloc { rt_off: 0x274, size: 0x0800, role_only: 0xff },
    G17FwAlloc { rt_off: 0x481, size: 0x12688, role_only: 1 }, // role 1 only
];

// ---------------------------------------------------------------------------
// Level 0 — init_data root, 0xc8 B ([P2] §2.1).
// ---------------------------------------------------------------------------

/// G17 init_data root. Its GPU VA is what the host sends on EP 0x20.
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub(crate) struct G17InitDataRoot {
    pub(crate) magic: u64,              // 0x00 G17_INITDATA_MAGIC
    pub(crate) fw_va_shared_alloc: u64, // 0x08 GFX1 reads it @0x9c3c
    field_0x10: [u8; 0x08],             // never written, never read
    pub(crate) runtime_pointers_gpu_va: u64, // 0x18 -> FW global 0x104888
    pub(crate) ptr_shared: u64,         // 0x20 -> FW global 0x107658
    pub(crate) fw_role: u32,            // 0x28 0=GFX, 1=GFX1
    pub(crate) host_mapped_allocs_enabled: u32, // 0x2c must be nonzero
    /// 0x30..0x97 copied verbatim from (*(accel+0x1c328))+0x18.
    /// FIXME(on-device): identify the accel+0x1c328 object (likely UAT/GART
    /// geometry). Left zero rather than invented.
    pub(crate) blob_gpu_config: [u8; 0x68],
    field_0x98: [u8; 0x10],
    pub(crate) ptr_role: u64,          // 0xa8 -> FW global 0x1047b8
    pub(crate) ptr_role0_only: u64,    // 0xb0 -> FW global 0x105248
    pub(crate) ptr_role1_only: u64,    // 0xb8 -> FW global 0x107660
    pub(crate) ptr_role1_only_2: u64,  // 0xc0 -> FW global 0x1047b0
}

static_assert!(size_of::<G17InitDataRoot>() == G17_INITDATA_ROOT_SIZE);
static_assert!(offset_of!(G17InitDataRoot, magic) == 0x00);
static_assert!(offset_of!(G17InitDataRoot, fw_va_shared_alloc) == 0x08);
static_assert!(offset_of!(G17InitDataRoot, runtime_pointers_gpu_va) == 0x18);
static_assert!(offset_of!(G17InitDataRoot, ptr_shared) == 0x20);
static_assert!(offset_of!(G17InitDataRoot, fw_role) == 0x28);
static_assert!(offset_of!(G17InitDataRoot, host_mapped_allocs_enabled) == 0x2c);
static_assert!(offset_of!(G17InitDataRoot, blob_gpu_config) == 0x30);
static_assert!(offset_of!(G17InitDataRoot, ptr_role) == 0xa8);
static_assert!(offset_of!(G17InitDataRoot, ptr_role0_only) == 0xb0);
static_assert!(offset_of!(G17InitDataRoot, ptr_role1_only) == 0xb8);
static_assert!(offset_of!(G17InitDataRoot, ptr_role1_only_2) == 0xc0);

impl Default for G17InitDataRoot {
    fn default() -> Self {
        // SAFETY: all fields are integers or byte arrays; all-zero is valid.
        unsafe { core::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// GPU virtual addresses of one role's allocations, supplied by the caller (in a
/// kernel port, by the UAT allocator).
#[derive(Copy, Clone, Default)]
pub(crate) struct G17RoleVa {
    /// The VA sent on EP 0x20.
    pub(crate) root: u64,
    /// root+0x18.
    pub(crate) runtime_pointers: u64,
    /// rt+0x000.
    pub(crate) hw_globals: u64,
    /// root+0x08.
    pub(crate) fw_va_shared_alloc: u64,
    /// root+0x20, identical for both roles.
    pub(crate) ptr_shared: u64,
    /// root+0xa8 and rt+0x200.
    pub(crate) ptr_role: u64,
    /// Cross-role hand-off pointer: role0 -> +0xb0 ; role1 -> +0xb8.
    /// FIXME(on-device): which peer object this denotes is UNKNOWN-C; the
    /// builder only reproduces the host's population pattern.
    pub(crate) cross_role: u64,
    /// role1 -> +0xc0 ; must be 0 for role0.
    pub(crate) cross_role_2: u64,
}

/// Caller-supplied config for the init_data builder.
#[derive(Copy, Clone)]
pub(crate) struct G17InitDataCfg {
    pub(crate) va: [G17RoleVa; G17_FW_ROLE_COUNT],
    /// 1..16.
    pub(crate) perf_state_count: u32,
    /// hw+0x10bc.
    pub(crate) perf_state_clock_mhz: [u32; G17_PERF_STATE_MAX],
}

/// One role's three levels, as the host would lay them out in GPU memory.
#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct G17InitDataRole {
    pub(crate) root: G17InitDataRoot,
    pub(crate) rt: G17RtPointers,
    pub(crate) hw: G17HwGlobals,
}

impl Default for G17InitDataRole {
    fn default() -> Self {
        G17InitDataRole {
            root: Default::default(),
            rt: Default::default(),
            hw: Default::default(),
        }
    }
}

/// The firmware's init_data abort taxonomy ([P1] §3 step 6).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum G17InitDataAbort {
    IfaceMismatch,       // root magic wrong -> "Interface is mismatched"
    HostMappedDisabled,  // root+0x2c == 0
}

/// Populate one role's root + runtime_pointers + hw/globals. Fails closed on: a
/// role index outside {0,1}, any required GPU VA that is zero or not
/// representable in the 44-bit init_data doorbell field, a perf-state count
/// outside 1..16, and a role-0 config that sets `cross_role_2`.
pub(crate) fn build_role(
    out: &mut G17InitDataRole,
    role: u32,
    cfg: &G17InitDataCfg,
) -> Result {
    if role as usize >= G17_FW_ROLE_COUNT {
        return Err(EINVAL);
    }
    let perf = cfg.perf_state_count;
    if perf == 0 || perf as usize > G17_PERF_STATE_MAX {
        return Err(EINVAL);
    }
    let va = &cfg.va[role as usize];

    // Required VAs must be present and 44-bit representable.
    for req in [va.root, va.runtime_pointers, va.hw_globals, va.ptr_shared, va.ptr_role] {
        if req == 0 || !va_fits_44(req) {
            return Err(ERANGE);
        }
    }
    if role == G17_FW_ROLE_GFX && va.cross_role_2 != 0 {
        return Err(EINVAL);
    }

    // ---- level 0: root ----
    let root = &mut out.root;
    root.magic = G17_INITDATA_MAGIC;
    root.fw_va_shared_alloc = va.fw_va_shared_alloc;
    root.runtime_pointers_gpu_va = va.runtime_pointers;
    root.ptr_shared = va.ptr_shared;
    root.fw_role = role;
    root.host_mapped_allocs_enabled = 1;
    root.ptr_role = va.ptr_role;
    // Cross-role population pattern (VERIFIED-A which offset per role; the peer
    // object is UNKNOWN-C). role0 populates +0xb0 only; role1 +0xb8 and +0xc0.
    if role == G17_FW_ROLE_GFX {
        root.ptr_role0_only = va.cross_role;
    } else {
        root.ptr_role1_only = va.cross_role;
        root.ptr_role1_only_2 = va.cross_role_2;
    }

    // ---- level 1: runtime_pointers ----
    let rt = &mut out.rt;
    rt.hw_globals_gpu_va = va.hw_globals;
    rt.ptr_role = va.ptr_role;
    rt.zeroed_0x250 = 0;
    // rt+0x481 is role-1 only (0x12688-byte buffer). VA is UNKNOWN-C in meaning;
    // the builder leaves it zero (never written by the host for role 0).

    // ---- level 2: hw/globals ----
    let hw = &mut out.hw;
    hw.perf_state_count_minus_1 = perf - 1;
    hw.perf_state0_clock_mhz = cfg.perf_state_clock_mhz[0];
    hw.perf_state_clock_mhz = cfg.perf_state_clock_mhz;

    Ok(())
}

/// Build both roots and wire the cross-role pointers. Boot order (GFX then GFX1)
/// is a transport concern, not a layout one: [P2] §3 proves both roots are fully
/// populated before either bootFirmware call. Rejects a config whose two roles
/// disagree on `ptr_shared` or share a root/runtime-pointers VA.
pub(crate) fn build_pair(
    out: &mut [G17InitDataRole; G17_FW_ROLE_COUNT],
    cfg: &G17InitDataCfg,
) -> Result {
    let g = &cfg.va[G17_FW_ROLE_GFX as usize];
    let g1 = &cfg.va[G17_FW_ROLE_GFX1 as usize];
    if g.ptr_shared != g1.ptr_shared {
        return Err(EINVAL);
    }
    if g.root == g1.root || g.runtime_pointers == g1.runtime_pointers {
        return Err(EINVAL);
    }
    build_role(&mut out[G17_FW_ROLE_GFX as usize], G17_FW_ROLE_GFX, cfg)?;
    build_role(&mut out[G17_FW_ROLE_GFX1 as usize], G17_FW_ROLE_GFX1, cfg)?;
    Ok(())
}

/// The firmware's own acceptance checks, in firmware order ([P1] §3 step 6).
pub(crate) fn root_check(root: &G17InitDataRoot) -> core::result::Result<(), G17InitDataAbort> {
    if root.magic != G17_INITDATA_MAGIC {
        return Err(G17InitDataAbort::IfaceMismatch);
    }
    if root.host_mapped_allocs_enabled == 0 {
        return Err(G17InitDataAbort::HostMappedDisabled);
    }
    Ok(())
}

/// Bounds-checked channel descriptor accessor. Index >= 12 (including the
/// [GATED] 13th entry described above) returns None.
pub(crate) fn rt_chan_desc(rt: &G17RtPointers, index: usize) -> Option<&G17RtChanDesc> {
    rt.chan_desc.get(index)
}
