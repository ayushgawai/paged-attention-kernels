import pytest, torch
from paged_attention import PagedKVCache, PagedAttentionBackend, reference_attention, paged_reference
from paged_attention import jit

torch.manual_seed(0)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")

def fill(cache, T, Hkv, D, dtype=torch.float16):
    s = cache.allocate()
    k, v = torch.randn(T, Hkv, D, dtype=dtype), torch.randn(T, Hkv, D, dtype=dtype)
    cache.append_many(s, k, v)
    return s, k, v

@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("dtype_name,dtol", [("fp16", 1e-3), ("bf16", 1e-2), ("fp8_e4m3", 0.15)])
def test_cache_append_gather(page_size, dtype_name, dtol):
    c = PagedKVCache(64, page_size, 4, 32, dtype_name)
    s, k, v = fill(c, page_size * 2 + 3, 4, 32)
    gk, gv = c.gather(s)
    assert gk.shape == k.shape and torch.allclose(gk.float(), k.float(), atol=dtol, rtol=dtol)
    assert torch.allclose(gv.float(), v.float(), atol=dtol, rtol=dtol)
    assert len(c.block_tables[s]) == 3

def test_cache_free_reuse():
    c = PagedKVCache(2, 16, 2, 16)
    s, _, _ = fill(c, 32, 2, 16); c.free_seq(s); assert len(c.free) == 2

@pytest.mark.parametrize("H,Hkv,D,T", [(8, 2, 32, 37), (4, 4, 64, 16), (8, 1, 128, 65)])
def test_decode_matches_sdpa(H, Hkv, D, T):
    c = PagedKVCache(64, 16, Hkv, D, "fp16")
    s, k, v = fill(c, T, Hkv, D)
    q = torch.randn(1, H, D, dtype=torch.float16)
    got = paged_reference(q, c, [s])
    want = reference_attention(q.unsqueeze(2), k.permute(1, 0, 2)[None], v.permute(1, 0, 2)[None])[:, :, 0]
    assert torch.allclose(got.float(), want.float(), atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_causal(dtype):
    c = PagedKVCache(64, 32, 2, 32, "bf16" if dtype == torch.bfloat16 else "fp16")
    T = 20; s, k, v = fill(c, T, 2, 32, dtype)
    q = torch.randn(1, 4, T, 32, dtype=dtype)
    got = paged_reference(q, c, [s], causal=True)
    want = reference_attention(q, k.permute(1, 0, 2)[None], v.permute(1, 0, 2)[None], causal=True)
    assert torch.allclose(got.float(), want.float(), atol=2e-2, rtol=2e-2)

def test_backend_and_ops():
    c = PagedKVCache(64, 16, 2, 32)
    s, _, _ = fill(c, 33, 2, 32)
    b = PagedAttentionBackend(c); b.plan([s])
    q = torch.randn(1, 8, 32, dtype=torch.float16)
    assert torch.allclose(b.run(q, [s]).float(), paged_reference(q, c, [s]).float(), atol=1e-3)

def test_jit_render_and_key():
    src = jit.render("decode", 64, 16, "bf16", "none")
    assert "64" in src and "paged_decode" in src and jit.spec_key(64, 16, "bf16", "none", "decode") != jit.spec_key(32, 16, "bf16", "none", "decode")
    with pytest.raises(RuntimeError, match="no CUDA"):
        if not torch.cuda.is_available(): jit.load("decode", 64, 16, "bf16")

@requires_cuda
def test_cuda_jit_decode():  # pragma: no cover - GPU only
    from paged_attention.jit import load
    load("decode", 32, 16, "fp16")

@requires_cuda
def test_triton_decode():  # pragma: no cover - GPU only
    import paged_attention.triton_kernels as tk
    assert tk.available()
