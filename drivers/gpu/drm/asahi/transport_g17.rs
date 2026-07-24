// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! G17 (Apple M5 / T6050) GPU RTKit wire ABI + SKSM kick-doorbell codec.
//!
//! Pure wire ABI: no MMIO, no DMA, no allocation. The kernel bring-up in
//! [`crate::gpu_g17`] drives an `apple-rtkit` mailbox through these codecs; this
//! module only encodes/decodes the words. Ported byte-for-byte from the
//! reverse-engineered host scaffold `research/gpu/implementation/g17_rtkit_abi.h`
//! and `g17_transport.h`, which build clean and pass under ASan/UBSan.
//!
//! Grades carried inline from the specs:
//!   VERIFIED-A : byte-confirmed on both firmware and host disassembly
//!   [GATED]    : required by the scaffold but the exact wire value / bit
//!                position is NOT byte-confirmed; a single point of change once a
//!                live boot trace lands. Nothing here is carried over from
//!                G13/G14 (whose EPs and init word are entirely different).
//!
//! Sources:
//!   [P1] research/gpu/G17_FIRMWARE_BOOT_PROTOCOL_2026-07-23.md
//!   [P2] research/gpu/G17_INITDATA_AND_BOOT_ORDER_2026-07-23.md §4

#![allow(dead_code)]

use kernel::prelude::*;

// ---------------------------------------------------------------------------
// 1. Endpoint map. [VERIFIED-A] [P1] §2: the G17 GPU firmware instantiates
//    EXACTLY these four RTKit endpoints. No syslog/ioreport/oslog/kdebug on this
//    ASC; 0x22/0x23 are internal device records, never host endpoints.
// ---------------------------------------------------------------------------

pub(crate) const G17_EP_MGMT: u8 = 0x00; // management / boot handshake, AP power
pub(crate) const G17_EP_CRASHLOG: u8 = 0x01; // crash / coredump
pub(crate) const G17_EP_GFX_MSGS: u8 = 0x20; // GPU control: init_data delivery
pub(crate) const G17_EP_GFX_INTS: u8 = 0x21; // GPU runtime loop: kick/power/recovery

pub(crate) const G17_ENDPOINTS: [u8; 4] =
    [G17_EP_MGMT, G17_EP_CRASHLOG, G17_EP_GFX_MSGS, G17_EP_GFX_INTS];
/// 6-bit endpoint id space ([P1] §3 EPMAP).
pub(crate) const G17_EP_ID_MAX: u8 = 0x3f;

/// True iff `ep` is one of the four endpoints this firmware exposes.
pub(crate) fn g17_ep_is_present(ep: u8) -> bool {
    G17_ENDPOINTS.contains(&ep)
}

/// Fail-closed check that a discovered endpoint set is exactly the four
/// endpoints above (order-independent).
pub(crate) fn g17_ep_set_expected(eps: &[u8]) -> Result {
    if eps.len() != G17_ENDPOINTS.len() {
        return Err(EINVAL);
    }
    for want in G17_ENDPOINTS {
        if eps.iter().filter(|&&e| e == want).count() != 1 {
            return Err(EINVAL);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. EP0 management message framing. [VERIFIED-A] [P1] §3: the management
//    message type lives in msg bits[55:52] (dispatch @0x37078).
// ---------------------------------------------------------------------------

pub(crate) const G17_MGMT_TYPE_SHIFT: u32 = 52;
pub(crate) const G17_MGMT_TYPE_MASK: u64 = 0xf;
pub(crate) const G17_MGMT_TYPE_FIELD_MASK: u64 = 0x00f0_0000_0000_0000;

/// [VERIFIED-A] EPMAP is management type 8 (@0x37140).
pub(crate) const G17_MGMT_TYPE_EPMAP: u8 = 0x8;
/// [VERIFIED-A/INFERENCE-B] STARTEP / AP-power path is type 0xb (@0x370a0).
pub(crate) const G17_MGMT_TYPE_AP_PWR: u8 = 0xb;
/// [GATED] HELLO / HELLO-ACK type numbers are NOT byte-confirmed. What IS
/// confirmed: the type field position [55:52], and that the reply carries two
/// 16-bit version fields checked == 0xc (@0x37110).
/// FIXME(on-device): confirm the HELLO/HELLO-ACK type ids from an EP0 trace.
pub(crate) const G17_MGMT_TYPE_HELLO: u8 = 0x1; // [GATED]
pub(crate) const G17_MGMT_TYPE_HELLO_ACK: u8 = 0x2; // [GATED]

/// [VERIFIED-A] interface version negotiated on EP0 is 0xc (v12).
pub(crate) const G17_IFACE_VERSION: u16 = 0xc;

/// [VERIFIED-A] observed AP-power payload literal ("movk 0x8000"). The semantic
/// name of the state is [GATED]; the value is not.
pub(crate) const G17_AP_PWR_STATE_ON: u16 = 0x8000;

/// [GATED] HELLO version field positions ([15:0] / [31:16] — conventional RTKit
/// layout, not a byte-confirmed G17 fact).
pub(crate) const G17_HELLO_VER_MIN_SHIFT: u32 = 0;
pub(crate) const G17_HELLO_VER_MAX_SHIFT: u32 = 16;

/// [GATED] EPMAP payload field positions. Type 8, bitmap iteration and 6-bit ids
/// are confirmed; bitmap width (32) / base position (bit 32) / last flag (bit 51)
/// are the scaffold's encoding choice, consistent with the 6-bit id space.
pub(crate) const G17_EPMAP_BITMAP_MASK: u64 = 0x0000_0000_ffff_ffff;
pub(crate) const G17_EPMAP_BASE_SHIFT: u32 = 32;
pub(crate) const G17_EPMAP_BASE_MASK: u64 = 0x7;
pub(crate) const G17_EPMAP_LAST_BIT: u32 = 51;
pub(crate) const G17_EPMAP_BLOCK_EPS: u32 = 32;
pub(crate) const G17_EPMAP_BLOCK_MAX: u8 = 1; // ids 0x00..0x3f => blocks 0 and 1

/// Generic EP0 codec. `payload` must not overlap the type field.
pub(crate) fn g17_ep0_encode(ty: u8, payload: u64) -> Result<u64> {
    if (ty as u64) > G17_MGMT_TYPE_MASK {
        return Err(EINVAL);
    }
    if payload & G17_MGMT_TYPE_FIELD_MASK != 0 {
        return Err(EINVAL);
    }
    Ok(((ty as u64) << G17_MGMT_TYPE_SHIFT) | payload)
}

pub(crate) fn g17_ep0_decode(msg: u64) -> (u8, u64) {
    let ty = ((msg >> G17_MGMT_TYPE_SHIFT) & G17_MGMT_TYPE_MASK) as u8;
    let payload = msg & !G17_MGMT_TYPE_FIELD_MASK;
    (ty, payload)
}

/// HELLO / HELLO-ACK: two 16-bit interface-version fields.
pub(crate) fn g17_ep0_hello_encode(ty: u8, ver_min: u16, ver_max: u16) -> Result<u64> {
    let payload =
        ((ver_min as u64) << G17_HELLO_VER_MIN_SHIFT) | ((ver_max as u64) << G17_HELLO_VER_MAX_SHIFT);
    g17_ep0_encode(ty, payload)
}

pub(crate) fn g17_ep0_hello_decode(msg: u64) -> (u8, u16, u16) {
    let (ty, payload) = g17_ep0_decode(msg);
    let ver_min = (payload >> G17_HELLO_VER_MIN_SHIFT) as u16;
    let ver_max = (payload >> G17_HELLO_VER_MAX_SHIFT) as u16;
    (ty, ver_min, ver_max)
}

/// [VERIFIED-A] The firmware accepts the handshake only when both 16-bit version
/// fields equal 0xc.
pub(crate) fn g17_iface_version_ok(ver_min: u16, ver_max: u16) -> Result {
    if ver_min == G17_IFACE_VERSION && ver_max == G17_IFACE_VERSION {
        Ok(())
    } else {
        Err(EINVAL)
    }
}

/// One EPMAP block: endpoint id = base*32 + bit.
#[derive(Copy, Clone, Default)]
pub(crate) struct G17EpmapBlock {
    pub(crate) base: u8,
    pub(crate) last: u8,
    pub(crate) bitmap: u32,
}

pub(crate) fn g17_ep0_epmap_encode(blk: &G17EpmapBlock) -> Result<u64> {
    if (blk.base as u64) > G17_EPMAP_BASE_MASK {
        return Err(EINVAL);
    }
    if blk.base > G17_EPMAP_BLOCK_MAX {
        return Err(ERANGE);
    }
    let mut payload = (blk.bitmap as u64) & G17_EPMAP_BITMAP_MASK;
    payload |= (blk.base as u64) << G17_EPMAP_BASE_SHIFT;
    if blk.last != 0 {
        payload |= 1u64 << G17_EPMAP_LAST_BIT;
    }
    g17_ep0_encode(G17_MGMT_TYPE_EPMAP, payload)
}

pub(crate) fn g17_ep0_epmap_decode(msg: u64) -> Result<G17EpmapBlock> {
    let (ty, payload) = g17_ep0_decode(msg);
    if ty != G17_MGMT_TYPE_EPMAP {
        return Err(EINVAL);
    }
    let base = ((payload >> G17_EPMAP_BASE_SHIFT) & G17_EPMAP_BASE_MASK) as u8;
    if base > G17_EPMAP_BLOCK_MAX {
        return Err(ERANGE);
    }
    Ok(G17EpmapBlock {
        base,
        last: ((payload >> G17_EPMAP_LAST_BIT) & 1) as u8,
        bitmap: (payload & G17_EPMAP_BITMAP_MASK) as u32,
    })
}

/// Expand a block into endpoint ids, pushing to `eps`. Rejects ids above the
/// 6-bit endpoint space.
pub(crate) fn g17_epmap_block_expand(blk: &G17EpmapBlock, eps: &mut KVec<u8>) -> Result {
    for bit in 0..G17_EPMAP_BLOCK_EPS {
        if blk.bitmap & (1u32 << bit) != 0 {
            let id = (blk.base as u32) * G17_EPMAP_BLOCK_EPS + bit;
            if id > G17_EP_ID_MAX as u32 {
                return Err(ERANGE);
            }
            eps.push(id as u8, GFP_KERNEL)?;
        }
    }
    Ok(())
}

/// Build the block list that advertises exactly {0x00,0x01,0x20,0x21}. Endpoints
/// 0x00/0x01 sit in block 0, 0x20/0x21 in block 1.
pub(crate) fn g17_epmap_blocks_for_g17() -> [G17EpmapBlock; 2] {
    [
        G17EpmapBlock {
            base: 0,
            last: 0,
            bitmap: (1 << 0) | (1 << 1), // 0x00, 0x01
        },
        G17EpmapBlock {
            base: 1,
            last: 1,
            bitmap: (1 << 0) | (1 << 1), // 0x20, 0x21 (32*1 + {0,1})
        },
    ]
}

/// AP-power / start-endpoints message (type 0xb).
pub(crate) fn g17_ep0_ap_pwr_encode(state: u16) -> Result<u64> {
    g17_ep0_encode(G17_MGMT_TYPE_AP_PWR, state as u64)
}

pub(crate) fn g17_ep0_ap_pwr_decode(msg: u64) -> Result<u16> {
    let (ty, payload) = g17_ep0_decode(msg);
    if ty != G17_MGMT_TYPE_AP_PWR {
        return Err(EINVAL);
    }
    Ok(payload as u16)
}

// ---------------------------------------------------------------------------
// 3. init_data handoff on EP 0x20. [VERIFIED-A] both sides ([P2] §4 correction:
//    the address field is 44 bits, NOT 48):
//      host  @0xfffffe0008b226bc:  mov x1,#0x81000000000000 ; bfxil x1,x0,#0,#0x2c
//      fw    @0xfffffc000002127c:  (msg & 0x00bf000000000000) == 0x0081000000000000
//                                  addr = 0xfffff00000000000 | (msg & 0xfffffffffff)
// ---------------------------------------------------------------------------

pub(crate) const G17_INITDATA_TAG: u64 = 0x81;
pub(crate) const G17_INITDATA_TAG_SHIFT: u32 = 48;
pub(crate) const G17_INITDATA_TAG_CHECK_MASK: u64 = 0x00bf_0000_0000_0000;
pub(crate) const G17_INITDATA_TAG_CHECK_VALUE: u64 = 0x0081_0000_0000_0000;
pub(crate) const G17_INITDATA_ADDR_BITS: u32 = 44;
pub(crate) const G17_INITDATA_ADDR_MASK: u64 = 0x0000_0fff_ffff_ffff;
pub(crate) const G17_INITDATA_ADDR_PREFIX: u64 = 0xfffff000_00000000;

/// Encode the init_data doorbell. Unlike the host (which truncates with bfxil),
/// this fails closed: `fw_va` must be representable in the 44-bit field, i.e. its
/// bits[63:44] must be all-zero or exactly the firmware-applied prefix.
pub(crate) fn g17_initdata_msg_encode(fw_va: u64) -> Result<u64> {
    let high = fw_va & !G17_INITDATA_ADDR_MASK;
    if high != 0 && high != G17_INITDATA_ADDR_PREFIX {
        return Err(ERANGE);
    }
    Ok((G17_INITDATA_TAG << G17_INITDATA_TAG_SHIFT) | (fw_va & G17_INITDATA_ADDR_MASK))
}

/// Decode as the firmware does: verify the tag under the 0x00bf.. mask (bad tag -> -EINVAL) and return
/// the address with bits[63:44] set to the firmware prefix.
pub(crate) fn g17_initdata_msg_decode(msg: u64) -> Result<u64> {
    if msg & G17_INITDATA_TAG_CHECK_MASK != G17_INITDATA_TAG_CHECK_VALUE {
        return Err(EINVAL);
    }
    Ok(G17_INITDATA_ADDR_PREFIX | (msg & G17_INITDATA_ADDR_MASK))
}

// ---------------------------------------------------------------------------
// 4. EP 0x21 runtime message type field. [VERIFIED-A] [P1] §4:
//    ubfx x19,x1,#44,#16 @0x2532c; validity mask 0x13ffffbe.
// ---------------------------------------------------------------------------

pub(crate) const G17_GFX_INT_TYPE_SHIFT: u32 = 44;
pub(crate) const G17_GFX_INT_TYPE_MASK: u64 = 0xffff;
/// [VERIFIED-A] byte-exact mask. NOTE: [P1] §4 also prints the prose range
/// "1..0x11, 0x19..0x1e", which does NOT equal this mask's bit set. The
/// byte-exact mask is authoritative here.
/// FIXME(on-device): confirm the valid runtime-type set against a live EP 0x21
/// trace (mask vs. prose discrepancy is UNKNOWN).
pub(crate) const G17_GFX_INT_TYPE_VALID_MASK: u32 = 0x13ffffbe;

pub(crate) fn g17_gfx_int_type(msg: u64) -> u16 {
    ((msg >> G17_GFX_INT_TYPE_SHIFT) & G17_GFX_INT_TYPE_MASK) as u16
}

pub(crate) fn g17_gfx_int_type_valid(ty: u16) -> bool {
    ty < 32 && (G17_GFX_INT_TYPE_VALID_MASK & (1u32 << ty)) != 0
}

// ---------------------------------------------------------------------------
// 5. SKSM kick doorbell ([P2] §4, byte-exact on both sides).
//
//   word = stamp_index[7:0] | stamp[39:8] | QID[46:40]
//   TS   = word[39:0]        (firmware prints TS=0x%010llx)
//   bit 46 doubles as the selector of the 64-bit valid-queue mask word
//   bits[63:47] have no producer and no consumer
//
// There is NO context id and NO priority in this word: priority feeds the kick
// ENTRY (AGXKSMKickQueueQOSConfig), not the doorbell. QID shift/mask come from
// halSKSMGetKickIDQueueIDConfig -> {0x28, 0x7f}.
// ---------------------------------------------------------------------------

pub(crate) const G17_KICK_STAMP_INDEX_MASK: u64 = 0xff;
pub(crate) const G17_KICK_STAMP_SHIFT: u32 = 8;
pub(crate) const G17_KICK_STAMP_MASK: u64 = 0xffff_ffff;
pub(crate) const G17_KICK_TS_MASK: u64 = 0xff_ffff_ffff; // bits 39:0
pub(crate) const G17_KICK_QID_SHIFT: u32 = 40;
pub(crate) const G17_KICK_QID_MASK: u64 = 0x7f;
pub(crate) const G17_KICK_QID_MAX: u8 = 0x7f;
pub(crate) const G17_KICK_MASK_SEL_BIT: u32 = 46;
pub(crate) const G17_KICK_RESERVED_MASK: u64 = 0xffff_8000_0000_0000; // bits 63:47

/// Per-queue firmware record: base + 0x10 + QID*0x20 (FW @0x10c94).
pub(crate) const G17_KICK_QUEUE_REC_BASE: u64 = 0x10;
pub(crate) const G17_KICK_QUEUE_REC_STRIDE: u64 = 0x20;

/// Number of KSM kick queues (7-bit QID space). 128 per §3h.
/// FIXME(on-device): per-queue geometry ([q+0x10] numStamps, [q+0x14] entry
/// stride, [q+0x48] stamp seed) is loaded from accel+0x11208 at runtime and is
/// NOT a static constant. A boot trace is required to size the entry ring.
pub(crate) const G17_KSM_KICK_QUEUE_COUNT: usize = 128;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct G17Kick {
    pub(crate) stamp_index: u8,
    pub(crate) stamp: u32,
    pub(crate) qid: u8,
    /// bits 39:0, exactly what the firmware prints.
    pub(crate) ts: u64,
    /// bit 46.
    pub(crate) mask_sel: u8,
}

/// Encode a doorbell word. QID > 0x7f is -ERANGE.
pub(crate) fn g17_kick_encode(qid: u8, stamp: u32, stamp_index: u8) -> Result<u64> {
    if qid > G17_KICK_QID_MAX {
        return Err(ERANGE);
    }
    let ts = ((stamp as u64) << G17_KICK_STAMP_SHIFT) | (stamp_index as u64);
    let word = (ts & G17_KICK_TS_MASK) | ((qid as u64 & G17_KICK_QID_MASK) << G17_KICK_QID_SHIFT);
    Ok(word)
}

/// Decode a doorbell word. With `strict`, bits[63:47] must be zero (they have no
/// producer in the recovered host code), else -EINVAL.
pub(crate) fn g17_kick_decode(word: u64, strict: bool) -> Result<G17Kick> {
    if strict && (word & G17_KICK_RESERVED_MASK) != 0 {
        return Err(EINVAL);
    }
    let ts = word & G17_KICK_TS_MASK;
    Ok(G17Kick {
        stamp_index: (word & G17_KICK_STAMP_INDEX_MASK) as u8,
        stamp: ((word >> G17_KICK_STAMP_SHIFT) & G17_KICK_STAMP_MASK) as u32,
        qid: ((word >> G17_KICK_QID_SHIFT) & G17_KICK_QID_MASK) as u8,
        ts,
        mask_sel: ((word >> G17_KICK_MASK_SEL_BIT) & 1) as u8,
    })
}

/// stamp_index = seq mod num_stamps, matching the host's div/msub + and #0xff.
pub(crate) fn g17_kick_stamp_index(seq: u32, num_stamps: u32) -> Result<u8> {
    if num_stamps == 0 {
        return Err(EINVAL);
    }
    Ok(((seq % num_stamps) & 0xff) as u8)
}

/// Byte offset of a queue's firmware record within the queue table.
pub(crate) fn g17_kick_queue_rec_offset(qid: u8) -> Result<u64> {
    if qid > G17_KICK_QID_MAX {
        return Err(ERANGE);
    }
    Ok(G17_KICK_QUEUE_REC_BASE + (qid as u64) * G17_KICK_QUEUE_REC_STRIDE)
}
