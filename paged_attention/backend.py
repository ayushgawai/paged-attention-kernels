"""Backend shaped like vLLM/SGLang attention backends: plan() builds metadata, run() executes."""
from dataclasses import dataclass, field
import torch
from . import ops
from .reference import paged_reference

@dataclass
class AttentionMetadata:
    block_tables: torch.Tensor
    seq_lens: list[int]
    is_prefill: bool
    seq_ids: list[int] = field(default_factory=list)

class PagedAttentionBackend:
    def __init__(self, cache, use_triton=False):
        self.cache, self.use_triton = cache, use_triton
        self.metadata = None

    def plan(self, seq_ids, is_prefill=False):
        self.metadata = AttentionMetadata(self.cache.block_table_tensor(seq_ids),
                                          [self.cache.seq_lens[s] for s in seq_ids], is_prefill, list(seq_ids))
        return self.metadata

    def run(self, q, seq_ids=None, is_prefill=False):
        """q: decode [B,H,D] or prefill [B,H,T,D]. KV must already be appended to the cache."""
        seq_ids = seq_ids if seq_ids is not None else self.metadata.seq_ids
        if self.metadata is None or seq_ids != self.metadata.seq_ids: self.plan(seq_ids, is_prefill)
        # Gathered-KV path feeds the custom ops; CUDA JIT kernel consumes block tables directly (see jit.py).
        ks, vs = zip(*(self.cache.gather(s) for s in seq_ids))
        maxlen = max(k.shape[0] for k in ks)
        def pad(x):  # [T,Hkv,D] -> [B,Hkv,maxlen,D], zero-pad (masked by seq_len in real kernels; ref uses per-seq below)
            return torch.stack([torch.cat([t, t.new_zeros(maxlen - t.shape[0], *t.shape[1:])]) for t in x]).permute(0, 2, 1, 3)
        if not is_prefill and all(k.shape[0] == maxlen for k in ks):
            return ops.decode_op(q, pad(ks), pad(vs))
        return paged_reference(q, self.cache, seq_ids, causal=is_prefill)
