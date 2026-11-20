# paged-attention-kernels

FlashInfer-style attention kernels for LLM serving over a paged KV cache, with a just-in-time kernel code generator.

> Status: CPU/reference path is tested end-to-end. CUDA and Triton kernels are written but **not yet run on a GPU** (developed on a machine without one) — treat them as unvalidated until the GPU tests in `tests/` pass on hardware. Push early, say so plainly.

## Design

```
                plan(seq_ids)                     run(q)
  backend ─────────────────────► AttentionMetadata ──────────┐
  (PagedAttentionBackend)        block_tables, seq_lens      │
                                                             ▼
  ┌─────────────┐   append   ┌──────────────┐  gather   ┌─────────────┐
  │ token K/V   │ ─────────► │ PagedKVCache │ ────────► │ gathered KV │
  └─────────────┘            │ blocks/pages │           └──────┬──────┘
                             │ block tables │                  │
                             │ FP8 scales   │                  ▼
                             └──────┬───────┘        custom ops / SDPA ref
                                    │ block_table              │
                                    ▼                          ▼
                             ┌──────────────┐          output [B,H,(T,)D]
                             │ JIT (jit.py) │ render src/*.cu for
                             │ hash-cached  │ (head_dim,page,dtype,mask)
                             └──────┬───────┘
                                    ▼
                    CUDA decode (split-KV + reduction) / prefill (causal, smem tile)
                    Triton decode kernel (triton_kernels.py)
```

| Module | File | What it does |
|---|---|---|
| Paged KV cache | `paged_attention/cache.py` | Block tables, free-list page allocation, append/gather; page sizes 16/32/64; FP16/BF16 storage and FP8-e4m3 with per-(token, head) abs-max scales |
| CUDA kernels | `src/paged_decode.cu`, `src/paged_prefill.cu` | Decode: GQA, online softmax, split-KV partials + separate reduction kernel. Prefill: causal mask, shared-memory K/V tiling. Templated on head_dim/page_size/dtype/mask |
| Triton | `paged_attention/triton_kernels.py` | Triton decode kernel; reference fallback off-GPU |
| JIT generator | `paged_attention/jit.py` | Renders CUDA templates per specialization, compiles via `torch.utils.cpp_extension.load_inline` (NVRTC-backed toolchain), disk-caches under `~/.cache/paged-attention-kernels` keyed by SHA-256 of the spec |
| Custom ops | `paged_attention/ops.py` | `torch.library` ops `paged_attn::decode` / `paged_attn::prefill` (with fake impls) |
| Backend | `paged_attention/backend.py` | `plan()`/`run()` interface shaped like vLLM/SGLang attention backends, block tables + seq lens metadata |
| Reference + tests | `paged_attention/reference.py`, `tests/` | PyTorch reference (GQA-aware, causal), config sweep over page sizes, dtypes (incl. FP8), head dims, GQA ratios |
| Benchmark | `bench.py` | Decode/prefill latency + achieved KV bandwidth vs `scaled_dot_product_attention` on gathered KV |

## Run

```bash
python -m venv .venv && .venv/bin/pip install torch pytest
.venv/bin/python -m pytest -q        # GPU tests skip cleanly without CUDA
.venv/bin/python bench.py            # says so and uses reference path without a GPU
```

On a CUDA machine, `paged_attention.jit.load("decode", head_dim=128, page_size=16, dtype="bf16")` compiles and caches the specialization; Triton decode is in `triton_kernels.py`.

## Sanity runs (actual output)

Hardware: Apple Silicon arm64 laptop, macOS, **no GPU** — PyTorch 2.15 CPU. These numbers measure the Python reference path, **not** the CUDA/Triton kernels.

```
$ .venv/bin/python -m pytest -q
17 passed, 2 skipped, 1 warning in 0.89s     # 2 skipped = CUDA-only tests

$ .venv/bin/python bench.py
device=cpu gpu=False  NO GPU: running reference/SDPA on CPU, numbers are NOT kernel performance
cfg                                     paged ms   sdpa ms   KV GB/s
decode B8 H32 Hkv8 D128 T1024         9.960     5.038      3.37
decode B32 H32 Hkv8 D128 T4096       158.627   870.609      3.38
decode B4 H32 Hkv8 D128 T512          3.809     1.159      2.20
prefill B2 T256: sdpa-causal 2.997 ms
```

(The `paged` column includes Python gather overhead per sequence; SDPA wins at small sizes for exactly that reason. On GPU the kernels read pages in-place and skip the gather.)

## Honest limitations

- CUDA/Triton kernels are **written but GPU-unvalidated**; the split-KV reduction and FP8 CUDA load paths especially need on-hardware correctness runs.
- Decode kernel uses a simple thread-cooperative dot product (shared-memory reduction), no tensor cores / `cp.async` / warp-per-head tuning — expect well below FlashInfer/FlashAttention throughput.
- FP8 cache uses per-(token, head) scales recomputed at append; no fused dequant in the CUDA path yet (kernels treat `__nv_fp8_e4m3` via `float()` conversion).
- Triton prefill falls back to the reference; Triton decode caps at 64 blocks/sequence.
- Backend `run()` gathers KV for the custom-op path; the direct block-table path is the JIT CUDA launcher.
- No continuous batching, chunked prefill, prefix caching, or CUDA graphs.

## License

MIT — see `LICENSE`.
