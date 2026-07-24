// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Hardware configuration for t6050 (Apple M5, "G17S") platforms.
//!
//! Identity constants are reverse-engineered; power/thermal curve tables are
//! NOT yet recovered for G17 and are left as clearly-marked `FIXME(on-device)`
//! placeholders. The G17 bring-up path (see `gpu_g17`) consumes the new 3-level
//! init_data grammar rather than the G13/G14 `HwConfig` curve/mapping fields, so
//! these placeholders do not block the first on-device checkpoint (does the
//! firmware ACK the init_data doorbell). They MUST be filled from a live macOS
//! `ioreg`/ADT capture before power management or real rendering is attempted.
//!
//! Sources:
//!   - research/gpu/GPU_G13_G17_DRIVER_CONTRACT_2026-07-22.md
//!     (chip 0x6050 / 24656, 20 GPU cores, Mac17,9 / J714s, G17S A0)
//!   - research/gpu/G17_HOST_DRIVER_SEQUENCE_2026-07-24.md §4
//!     (UAT single-stage, 16 KiB pages, 42-bit PA)
//!   - live ioreg: 20 cores, 2 clusters, usc_gen 3

use crate::f32;

use super::*;

pub(crate) const HWCONFIG_T6050: super::HwConfig = HwConfig {
    // VERIFIED: chip id 0x6050 (24656) — paired manifest, Mac17,9 / J714s.
    chip_id: 0x6050,
    // VERIFIED: G17 generation, "S" variant (G17S), M5.
    gpu_gen: GpuGen::G17,
    gpu_variant: GpuVariant::S,
    // See hw::GpuCore::G17S FIXME: firmware discriminant not yet confirmed.
    gpu_core: GpuCore::G17S,

    // Apple SoC timebase is 24 MHz across the whole Apple Silicon line; the G17
    // ASC/RTKit timer is the same 24 MHz clock (RTKit 3514.0.7).
    base_clock_hz: 24_000_000,
    // VERIFIED: UAT output address space is 42-bit
    // (research/gpu/G17_HOST_DRIVER_SEQUENCE_2026-07-24.md §4; PTE OA mask
    // 0x3ffffffc000 = bits[41:14]).
    uat_oas: 42,
    // VERIFIED (live ioreg): single die, 2 clusters, 20 cores total (10/cluster).
    num_dies: 1,
    max_num_clusters: 2,
    max_num_cores: 10,
    // FIXME(on-device): per-cluster frag/GP counts not separately RE'd for G17.
    // Mirror the core count (as t8112/t602x do) until a live ioreg confirms.
    max_num_frags: 10,
    max_num_gps: 4,

    // FIXME(on-device): preemption buffer sizes are firmware-version specific and
    // NOT recovered for the g17s a0 firmware. These carry the T602x (G14X) values
    // as a structural placeholder; they must be re-derived from the G17 firmware
    // (or a live macOS trace) before real command submission.
    preempt1_size: 0x540,
    preempt2_size: 0x280,
    preempt3_size: 0x40,
    compute_preempt1_size: 0x25980 * 2,
    // FIXME(on-device): clustering meta sizes not RE'd for G17. G17 is a
    // multi-cluster part so clustering is expected, but the meta1..4 block sizes
    // are unknown. Left None so the driver does not program wrong values.
    clustering: None,

    render: HwRenderConfig {
        // FIXME(on-device): tiling_control register value not RE'd for G17.
        tiling_control: 0,
    },

    // FIXME(on-device): all HwConfigA/B "unk_*" values below are NOT reverse
    // engineered for G17; they feed the G13/G14 hwdata_a/hwdata_b structs, which
    // the G17 init_data path (gpu_g17) does not use. Zeroed structurally.
    da: HwConfigA {
        unk_87c: 0,
        unk_8cc: 0,
        unk_e24: 0,
    },
    db: HwConfigB {
        unk_454: 0,
        unk_4e0: 0,
        unk_534: 0,
        unk_ab8: 0,
        unk_abc: 0,
        unk_b30: 0,
    },
    // FIXME(on-device): shared/curve/coefficient tables are power-model data not
    // recovered for G17. Empty/zero placeholders — must be captured from the ADT
    // before enabling power management.
    shared1_tab: &[],
    shared1_a4: 0,
    shared2_tab: &[],
    shared2_unk_508: 0,
    shared2_curves: None,
    shared3_unk: 0,
    shared3_tab: &[],
    idle_off_standby_timer_default: 0,
    unk_hws2_4: None,
    unk_hws2_24: 0,
    global_unk_54: 0,
    sram_k: f32!(0.0),
    unk_coef_a: &[],
    unk_coef_b: &[],
    global_tab: None,
    // FIXME(on-device): whether G17 exposes CS/AFR (shared-rail) perf states is
    // not confirmed. Set false so PwrConfig::load does not require apple,cs-opp /
    // apple,afr-opp DT nodes that may not exist yet.
    has_csafr: false,
    // FIXME(on-device): temperature sensor masks not RE'd for G17.
    fast_sensor_mask: [0, 0],
    fast_sensor_mask_alt: [0, 0],
    fast_die0_sensor_present: 0,
    // FIXME(on-device): G17 firmware MMIO mappings (Fender/AICTimer/RGX/MCache/…)
    // are NOT recovered. Empty list: the G17 init_data path supplies GPU VAs from
    // the UAT allocator and does not depend on this table for the boot doorbell.
    io_mappings: &[],
    sram_base: None,
    sram_size: None,
};
