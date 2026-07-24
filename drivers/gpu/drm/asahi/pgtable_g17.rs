// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! G17 (Apple M5 / T6050) single-stage UAT page-table entry encoders.
//!
//! On G17 the GPU MMU is the UAT and there is **no separate DART**: the UAT is
//! the GPU's page table (`AGXUATSingleStage`). It is a 3-level walk
//! (PCE(L1) -> PDE(L2) -> PTE(L3)) over **16 KiB pages** with a **42-bit**
//! output address. This module supplies the byte-exact G17 entry encoders; the
//! table walk / page allocation is the generic 16 KiB / 3-level machinery in
//! [`crate::pgtable::UatPageTable`], whose block structure (page bit 14, table
//! type 0x3) is already format-compatible with G17. What differs on G17 — and
//! why a dedicated encoder exists — is the **leaf PTE flag layout**: G13/G14
//! (`crate::pgtable::Prot`) place AP at bits[7:6] / MemAttr at bits[4:2], while
//! the G17 `encodePTEFlags` uses base words 0x403/0xc03 and packs its AP/MemAttr
//! fields high (bits 53/54/55). See the FIXMEs below.
//!
//! Byte-exact source (all VERIFIED-A unless noted):
//!   research/gpu/G17_HOST_DRIVER_SEQUENCE_2026-07-24.md §4 (AGXG17X encoders)
//!     encodePTE   @0xfffffe0008b996bc : PTE = (phys & 0x3ffffffc000) | flags
//!     encodePDE   @0xfffffe0008b996cc : PDE = (child & 0xffffffffffffc000) | 0x3
//!     encodePCE   @0xfffffe0008b99714 : PCE = (child & 0xffffffffffffc000) | 0x3
//!     encodePDEFlags/encodePCEFlags   : constant 0x3 (valid|table)
//!     isPTE/PDE/PCEValid              : entry & 1
//!     isPhysicalAddressValid @b997c0  : (pa & 0xfffffc0000003fff) == 0

#![allow(dead_code)]

use core::mem::size_of;
use kernel::static_assert;

/// Number of bits in a 16 KiB page offset. [VERIFIED-A]
pub(crate) const G17_UAT_PGBIT: usize = 14;
/// 16 KiB page size.
pub(crate) const G17_UAT_PGSZ: usize = 1 << G17_UAT_PGBIT;
/// Page offset mask.
pub(crate) const G17_UAT_PGMSK: u64 = (G17_UAT_PGSZ as u64) - 1;

/// Output address space width in bits (42-bit PA). [VERIFIED-A]
pub(crate) const G17_UAT_OAS: u32 = 42;

/// Number of translation levels (PCE -> PDE -> PTE). [VERIFIED-A]
pub(crate) const G17_UAT_LEVELS: usize = 3;

/// Entry size (all levels are 64-bit).
pub(crate) const G17_PTE_SIZE: usize = size_of::<u64>();
/// Entries per 16 KiB table (16384 / 8 = 2048).
pub(crate) const G17_UAT_NPTE: usize = G17_UAT_PGSZ / G17_PTE_SIZE;
/// Address bits consumed per level (log2(2048) = 11).
pub(crate) const G17_UAT_LVBIT: usize = G17_UAT_PGBIT - 3;
/// Per-level index mask.
pub(crate) const G17_UAT_LVMSK: u64 = (G17_UAT_NPTE as u64) - 1;

/// PTE output-address mask: OA occupies bits[41:14]. [VERIFIED-A]
///   0x3ffffffc000 = bits 14..=41.
pub(crate) const G17_PTE_OA_MASK: u64 = 0x0000_03ff_ffff_c000;

/// Table-descriptor child-address mask (bits[63:14]). [VERIFIED-A]
pub(crate) const G17_TABLE_ADDR_MASK: u64 = 0xffff_ffff_ffff_c000;

/// PDE/PCE flag constant: valid | table. [VERIFIED-A]
pub(crate) const G17_TABLE_FLAGS: u64 = 0x3;

/// Leaf-PTE base flag word for a read-only mapping (bit 0/1 = valid leaf,
/// bit 10 = AF). [VERIFIED-A: encodePTEFlags base 0x403]
pub(crate) const G17_PTE_FLAGS_RO: u64 = 0x403;
/// Leaf-PTE base flag word for a writable mapping (RO | bit 11).
/// [VERIFIED-A: encodePTEFlags base 0xc03 when arg1 != 0]
pub(crate) const G17_PTE_FLAGS_RW: u64 = 0xc03;

/// Bit 0 = valid, for all three levels. [VERIFIED-A]
pub(crate) const G17_ENTRY_VALID: u64 = 0x1;
/// Low two bits == 0b11 marks a valid table/leaf descriptor.
pub(crate) const G17_ENTRY_TYPE_MASK: u64 = 0x3;
pub(crate) const G17_ENTRY_TYPE_LEAF_TABLE: u64 = 0x3;

/// isPhysicalAddressValid mask: (pa & this) == 0 means 16 KiB-aligned and
/// PA < 2^42. [VERIFIED-A @b997c0]
pub(crate) const G17_PA_INVALID_MASK: u64 = 0xffff_fc00_0000_3fff;

// Sanity: the OA mask and the 42-bit / 16 KiB-alignment invariant agree.
static_assert!(G17_PTE_OA_MASK == ((((1u64 << G17_UAT_OAS) - 1)) & !G17_UAT_PGMSK));
static_assert!(G17_PA_INVALID_MASK == !(((1u64 << G17_UAT_OAS) - 1) & !G17_UAT_PGMSK));
static_assert!(G17_UAT_NPTE == 2048);
static_assert!(G17_UAT_LVBIT == 11);

/// `isPhysicalAddressValid`: true iff `pa` is 16 KiB-aligned and below 2^42.
pub(crate) const fn g17_pa_valid(pa: u64) -> bool {
    (pa & G17_PA_INVALID_MASK) == 0
}

/// `isValid` for any level: bit 0 set.
pub(crate) const fn g17_entry_valid(entry: u64) -> bool {
    (entry & G17_ENTRY_VALID) != 0
}

/// True iff the entry is a valid table/leaf descriptor (low two bits == 0b11).
pub(crate) const fn g17_entry_is_table(entry: u64) -> bool {
    (entry & G17_ENTRY_TYPE_MASK) == G17_ENTRY_TYPE_LEAF_TABLE
}

/// Memory-attribute / access-permission fields that G17 `encodePTEFlags` ORs in
/// above the 0x403/0xc03 base. The RE recovered the *bit positions* the encoder
/// writes (53/54/55 and a low `<< 2` field) but NOT the per-bit semantics
/// (which MemAttr index maps to which cacheability, AP bit meaning, etc.).
///
/// FIXME(on-device): resolve the exact mapping of MAIR/AP/shareability to these
/// bits. Until then a caller passes 0 (device/uncached-ish default implied by
/// the bare 0x403/0xc03 base) and this stays a no-op. What to try on device:
/// map a scratch page RW, read back the PTE the firmware/uPPL accepted, and diff
/// against 0xc03 to recover the attribute bits before enabling cached mappings.
#[derive(Copy, Clone, Default)]
pub(crate) struct G17PteAttr {
    /// Raw bits to OR into the high field region (bits 53..=55). Structural
    /// until the semantics are traced; 0 = leave base flags only.
    pub(crate) high_bits: u64,
    /// Raw bits to OR into the low `<< 2` field. 0 until traced.
    pub(crate) low_bits: u64,
}

/// `encodePTEFlags`: base flag word for a leaf PTE.
///   writable == false -> 0x403 (RO), true -> 0xc03 (RW).
/// The attribute fields are applied per [`G17PteAttr`] (FIXME above).
pub(crate) const fn g17_encode_pte_flags(writable: bool, attr: G17PteAttr) -> u64 {
    let base = if writable {
        G17_PTE_FLAGS_RW
    } else {
        G17_PTE_FLAGS_RO
    };
    // High attribute field region is bits 53..=55 per the RE (<<0x35/0x36/0x37);
    // low field is <<2. Callers pass 0 until the semantics are traced.
    base | (attr.high_bits << 53) | (attr.low_bits << 2)
}

/// `encodePTE`: leaf PTE = (phys & 0x3ffffffc000) | flags.
///
/// Returns None if `phys` is not a valid (16 KiB-aligned, < 2^42) physical
/// address — the host `isPhysicalAddressValid` gate.
pub(crate) fn g17_encode_pte(phys: u64, flags: u64) -> Option<u64> {
    if !g17_pa_valid(phys) {
        return None;
    }
    Some((phys & G17_PTE_OA_MASK) | flags)
}

/// `encodePDE`/`encodePCE`: table descriptor = (child_table_PA & ~0x3fff) | 0x3.
///
/// Returns None if `child_pa` is not 16 KiB-aligned / < 2^42.
pub(crate) fn g17_encode_table(child_pa: u64) -> Option<u64> {
    if !g17_pa_valid(child_pa) {
        return None;
    }
    Some((child_pa & G17_TABLE_ADDR_MASK) | G17_TABLE_FLAGS)
}

/// Recover the output physical address from a leaf PTE (inverse of encodePTE for
/// the address field).
pub(crate) const fn g17_pte_output_addr(pte: u64) -> u64 {
    pte & G17_PTE_OA_MASK
}

/// Recover the child table physical address from a PDE/PCE.
pub(crate) const fn g17_table_child_addr(entry: u64) -> u64 {
    entry & G17_TABLE_ADDR_MASK & !G17_UAT_PGMSK
}

/// The per-level table index for a given input VA. `level` 0 = PTE (leaf),
/// 1 = PDE, 2 = PCE (root), matching the shift `PGBIT + level*LVBIT`.
pub(crate) const fn g17_level_index(iova: u64, level: usize) -> usize {
    let shift = G17_UAT_PGBIT + level * G17_UAT_LVBIT;
    ((iova >> shift) & G17_UAT_LVMSK) as usize
}
