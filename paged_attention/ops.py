"""torch.library custom ops; CPU/CUDA-less impl delegates to reference (same math the kernels implement)."""
import torch
from .reference import reference_attention

try:
    @torch.library.custom_op("paged_attn::decode", mutates_args=())
    def decode_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q [B,H,D], k/v gathered [B,Hkv,T,D]
        return reference_attention(q.unsqueeze(2), k, v)[:, :, 0]

    @decode_op.register_fake
    def _(q, k, v): return torch.empty_like(q)

    @torch.library.custom_op("paged_attn::prefill", mutates_args=())
    def prefill_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return reference_attention(q, k, v, causal=True)

    @prefill_op.register_fake
    def _(q, k, v): return torch.empty_like(q)
    HAS_OPS = True
except Exception:  # very old torch
    HAS_OPS = False
    decode_op = lambda q, k, v: reference_attention(q.unsqueeze(2), k, v)[:, :, 0]
    prefill_op = lambda q, k, v: reference_attention(q, k, v, causal=True)
