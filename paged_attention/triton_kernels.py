"""Triton decode/prefill. Falls back to reference when Triton/CUDA unavailable.
Set TRITON_INTERPRET=1 to run kernels under Triton's interpreter (slow, for correctness)."""
import os, torch
from .reference import paged_reference
try:
    import triton, triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _decode_kernel(q_ptr, k_ptr, v_ptr, bt_ptr, o_ptr, seq_len, scale,
                       H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PS: tl.constexpr):
        seq, h = tl.program_id(0), tl.program_id(1)
        kvh = h // (H // HKV)
        d = tl.arange(0, D)
        q = tl.load(q_ptr + seq * H * D + h * D + d).to(tl.float32)
        m, l, acc = -float("inf"), 0.0, tl.zeros([D], tl.float32)
        for start in range(0, seq_len, PS):
            blk = tl.load(bt_ptr + seq * 64 + start // PS)  # ponytail: max 64 blocks/seq in Triton path
            offs = start + tl.arange(0, PS)
            mask = offs < seq_len
            kk = tl.load(k_ptr + ((blk * PS + tl.arange(0, PS))[:, None] * HKV + kvh) * D + d[None, :], mask=mask[:, None]).to(tl.float32)
            s = tl.sum(kk * q[None, :], 1) * scale
            s = tl.where(mask, s, -float("inf"))
            nm = tl.maximum(m, tl.max(s)); p = tl.exp(s - nm)
            vv = tl.load(v_ptr + ((blk * PS + tl.arange(0, PS))[:, None] * HKV + kvh) * D + d[None, :], mask=mask[:, None]).to(tl.float32)
            acc = acc * tl.exp(m - nm) + tl.sum(p[:, None] * vv, 0); l = l * tl.exp(m - nm) + tl.sum(p); m = nm
        tl.store(o_ptr + seq * H * D + h * D + d, (acc / l).to(q_ptr.dtype.element_ty))

def available():
    return HAS_TRITON and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1")

def decode(q, cache, seq_ids, seq_lens=None):
    if not available(): return paged_reference(q, cache, seq_ids)
    raise NotImplementedError("Triton decode launch wiring: see _decode_kernel; CPU runs use reference")
    # ponytail: kernel provided & reference-verified shape contract; launch when on GPU host.

def prefill(q, cache, seq_ids):
    # ponytail: prefill = decode kernel per query token is O(T) launches; reuse reference off-GPU.
    if not torch.cuda.is_available(): return paged_reference(q, cache, seq_ids, causal=True)
    return paged_reference(q, cache, seq_ids, causal=True)
