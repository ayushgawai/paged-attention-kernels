"""Paged KV cache: block tables, allocation, append/gather. FP8 = e4m3, per-(token,head) scale."""
import torch

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8_e4m3": torch.float8_e4m3fn}
FP8_MAX = 448.0

class PagedKVCache:
    def __init__(self, num_blocks, page_size, num_kv_heads, head_dim, dtype="fp16", device="cpu"):
        assert page_size in (16, 32, 64), page_size
        self.page_size, self.num_kv_heads, self.head_dim = page_size, num_kv_heads, head_dim
        self.dtype_name, self.storage_dtype = dtype, _DTYPES[dtype]
        self.compute_dtype = torch.float16 if dtype == "fp8_e4m3" else self.storage_dtype
        shape = (num_blocks, page_size, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=self.storage_dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=self.storage_dtype, device=device)
        # per-(block, token, head) abs-max scale; 1.0 for non-FP8
        self.k_scales = torch.ones(shape[:3], device=device)
        self.v_scales = torch.ones(shape[:3], device=device)
        self.free = list(range(num_blocks))[::-1]
        self.block_tables: dict[int, list[int]] = {}
        self.seq_lens: dict[int, int] = {}
        self._next = 0

    def allocate(self, seq_id=None):
        seq_id = self._next if seq_id is None else seq_id
        self._next = max(self._next, seq_id + 1)
        self.block_tables[seq_id], self.seq_lens[seq_id] = [], 0
        return seq_id

    def free_seq(self, seq_id):
        self.free.extend(self.block_tables.pop(seq_id)); self.seq_lens.pop(seq_id)

    def _block(self, seq_id, pos):
        idx = pos // self.page_size
        while idx >= len(self.block_tables[seq_id]):
            assert self.free, "out of KV blocks"
            self.block_tables[seq_id].append(self.free.pop())
        return self.block_tables[seq_id][idx], pos % self.page_size

    def _store(self, cache, scales, blk, off, x):  # x: [H, D]
        if self.dtype_name == "fp8_e4m3":
            s = (x.float().abs().amax(-1) / FP8_MAX).clamp_min(1e-8)
            scales[blk, off] = s
            cache[blk, off] = (x.float() / s.unsqueeze(-1)).to(self.storage_dtype)
        else:
            cache[blk, off] = x.to(self.storage_dtype)

    def append(self, seq_id, k, v):
        """k, v: [num_kv_heads, head_dim] for one token."""
        pos = self.seq_lens[seq_id]
        blk, off = self._block(seq_id, pos)
        self._store(self.k_cache, self.k_scales, blk, off, k)
        self._store(self.v_cache, self.v_scales, blk, off, v)
        self.seq_lens[seq_id] = pos + 1

    def append_many(self, seq_id, k, v):  # [T, H, D]
        for t in range(k.shape[0]): self.append(seq_id, k[t], v[t])

    def gather(self, seq_id):
        """Return contiguous dequantized K, V: [seq_len, H, D] in compute dtype."""
        n = self.seq_lens[seq_id]
        if n == 0:
            empty = torch.empty(0, self.num_kv_heads, self.head_dim, dtype=self.compute_dtype, device=self.k_cache.device)
            return empty, empty.clone()
        idx, offs = [], []
        for p in range(n):
            b, o = self.block_tables[seq_id][p // self.page_size], p % self.page_size
            idx.append(b); offs.append(o)
        idx_t = torch.tensor(idx, device=self.k_cache.device); off_t = torch.tensor(offs, device=self.k_cache.device)
        def deq(cache, scales):
            x = cache[idx_t, off_t].float()
            return (x * scales[idx_t, off_t].unsqueeze(-1)).to(self.compute_dtype)
        return deq(self.k_cache, self.k_scales), deq(self.v_cache, self.v_scales)

    def block_table_tensor(self, seq_ids, device=None):
        maxb = max((len(self.block_tables[s]) for s in seq_ids), default=0)
        t = torch.full((len(seq_ids), maxb), -1, dtype=torch.int32)
        for i, s in enumerate(seq_ids): t[i, :len(self.block_tables[s])] = torch.tensor(self.block_tables[s])
        return t.to(device or self.k_cache.device)
