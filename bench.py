#!/usr/bin/env python3
"""Decode/prefill latency + achieved KV bandwidth vs SDPA on gathered KV.
No GPU -> reference path (Triton interpreter not available off-CUDA either); says so."""
import time, torch
import torch.nn.functional as F
from paged_attention import PagedKVCache, paged_reference, reference_attention

def bench(fn, iters=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms

def main():
    gpu = torch.cuda.is_available()
    dev = "cuda" if gpu else "cpu"
    print(f"device={dev} gpu={gpu}" + ("" if gpu else "  NO GPU: running reference/SDPA on CPU, numbers are NOT kernel performance"))
    print(f"{'cfg':<38}{'paged ms':>10}{'sdpa ms':>10}{'KV GB/s':>10}")
    for (B, H, Hkv, D, T) in [(8, 32, 8, 128, 1024), (32, 32, 8, 128, 4096), (4, 32, 8, 128, 512)]:
        c = PagedKVCache(max(64, B * (T // 16 + 1)), 16, Hkv, D, "bf16", dev)
        seqs, ks, vs = [], [], []
        for _ in range(B):
            s = c.allocate(); k, v = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device=dev), torch.randn(T, Hkv, D, dtype=torch.bfloat16, device=dev)
            c.append_many(s, k, v); seqs.append(s); ks.append(k); vs.append(v)
        q = torch.randn(B, H, D, dtype=torch.bfloat16, device=dev)
        kk = torch.stack(ks).permute(0, 2, 1, 3); vv = torch.stack(vs).permute(0, 2, 1, 3)
        ms_p = bench(lambda: paged_reference(q, c, seqs), 5)
        ms_s = bench(lambda: reference_attention(q.unsqueeze(2), kk, vv), 5)
        gbs = (2 * B * T * Hkv * D * 2 / 1e9) / (ms_p / 1e3)
        print(f"decode B{B} H{H} Hkv{Hkv} D{D} T{T:<8}{ms_p:>10.3f}{ms_s:>10.3f}{gbs:>10.2f}")
    # prefill
    B, H, Hkv, D, T = 2, 32, 8, 128, 256
    q = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, Hkv, T, D, dtype=torch.bfloat16, device=dev); v = torch.randn_like(k)
    print(f"prefill B{B} T{T}: sdpa-causal {bench(lambda: reference_attention(q, k, v, causal=True), 5):.3f} ms")

if __name__ == "__main__": main()
