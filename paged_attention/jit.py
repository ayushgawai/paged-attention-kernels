"""JIT: render src/*.cu for a (head_dim, page_size, dtype, mask) spec, compile, disk-cache by hash."""
import hashlib, json, os
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
DTYPE = {"fp16": ("__half", "half"), "bf16": ("__nv_bfloat16", "bf16"), "fp8_e4m3": ("__nv_fp8_e4m3", "fp8")}

def spec_key(head_dim, page_size, dtype, mask, kind):
    return hashlib.sha256(json.dumps([head_dim, page_size, dtype, mask, kind], sort_keys=True).encode()).hexdigest()[:16]

def render(kind, head_dim, page_size, dtype, mask="none"):
    """kind: 'decode'|'prefill'. Returns CUDA source with templates instantiated."""
    src = (SRC / f"paged_{kind}.cu").read_text()
    ctype, _ = DTYPE[dtype]
    suffix = f"{head_dim}_{page_size}_{dtype}_{mask}"
    return src.replace("__SPEC_SUFFIX__", suffix).replace("__HEAD_DIM__", str(head_dim)) \
              .replace("__PAGE_SIZE__", str(page_size)).replace("__DTYPE__", ctype) \
              .replace("__CAUSAL__", "1" if mask == "causal" else "0")

_cache = {}
def load(kind, head_dim, page_size, dtype, mask="none", cache_dir=None):
    """Compile via torch load_inline (NVRTC via cuda-python if importable is left to torch's backend).
    Raises RuntimeError without CUDA. Disk cache keyed by spec_key()."""
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("no CUDA: use reference/Triton-interpreter path")
    key = spec_key(head_dim, page_size, dtype, mask, kind)
    if key in _cache: return _cache[key]
    cache_dir = Path(cache_dir or os.environ.get("PAGED_ATTN_CACHE", Path.home() / ".cache/paged-attention-kernels"))
    build = cache_dir / key; build.mkdir(parents=True, exist_ok=True)
    marker = build / "done"
    from torch.utils.cpp_extension import load_inline
    mod = load_inline(name=f"paged_{kind}_{key}", cpp_sources="", cuda_sources=render(kind, head_dim, page_size, dtype, mask),
                      functions=[f"paged_{kind}"], build_directory=str(build), extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"])
    marker.write_text(key); _cache[key] = mod
    return mod
