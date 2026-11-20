import math, torch
import torch.nn.functional as F

def repeat_kv(x, n_rep):  # [B, Hkv, T, D] -> [B, H, T, D]
    if n_rep == 1: return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)

def reference_attention(q, k, v, causal=False):
    """q [B,H,Tq,D], k/v [B,Hkv,Tk,D]. GQA via repeat. Returns [B,H,Tq,D]."""
    k, v = repeat_kv(k, q.shape[1] // k.shape[1]), repeat_kv(v, q.shape[1] // v.shape[1])
    mask = None
    if causal:
        tq, tk = q.shape[2], k.shape[2]
        mask = torch.full((tq, tk), float("-inf"), device=q.device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=tk - tq + 1)
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask).to(q.dtype)

def paged_reference(q, cache, seq_ids, causal=False):
    """q [B,H,D] (decode) or [B,H,T,D] (prefill, last-dim seqs in cache incl. q tokens)."""
    outs = []
    for i, s in enumerate(seq_ids):
        k, v = cache.gather(s)  # [T,Hkv,D]
        qi = q[i].unsqueeze(-2) if q.dim() == 3 else q[i]  # [H,1,D] or [H,T,D]
        outs.append(reference_attention(qi.unsqueeze(0), k.permute(1,0,2).unsqueeze(0), v.permute(1,0,2).unsqueeze(0), causal=causal)[0])
    out = torch.stack(outs)
    return out[:, :, 0] if q.dim() == 3 else out
