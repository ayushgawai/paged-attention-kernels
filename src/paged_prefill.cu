// Paged prefill: causal mask, shared-memory K/V tile. One block per (seq, q-token, head).
// Template params substituted by paged_attention/jit.py.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#define HEAD_DIM __HEAD_DIM__
#define PAGE_SIZE __PAGE_SIZE__
#define CAUSAL __CAUSAL__
typedef __DTYPE__ dtype_t;
__device__ float to_f(dtype_t x) { return float(x); }

extern "C" __global__ void paged_prefill_kernel__SPEC_SUFFIX__(
    const dtype_t* __restrict__ q, const dtype_t* __restrict__ k, const dtype_t* __restrict__ v,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens, dtype_t* __restrict__ out,
    int num_heads, int num_kv_heads, int max_blocks, int q_len, float scale) {
  int seq = blockIdx.x, qt = blockIdx.y, head = blockIdx.z;
  int kv_head = head / (num_heads / num_kv_heads);
  int len = seq_lens[seq];
  int kv_limit = CAUSAL ? min(len, len - q_len + qt + 1) : len;  // q tokens are the last q_len positions
  __shared__ float tile_k[PAGE_SIZE][HEAD_DIM];
  __shared__ float tile_v[PAGE_SIZE][HEAD_DIM];
  float qv[HEAD_DIM], acc[HEAD_DIM];
  for (int d = 0; d < HEAD_DIM; d++) { qv[d] = to_f(q[(((long)seq * num_heads + head) * q_len + qt) * HEAD_DIM + d]); acc[d] = 0.f; }
  float m = -1e30f, l = 0.f;
  for (int start = 0; start < kv_limit; start += PAGE_SIZE) {
    int n = min(PAGE_SIZE, kv_limit - start);
    for (int i = threadIdx.x; i < n * HEAD_DIM; i += blockDim.x) {
      int t = start + i / HEAD_DIM, d = i % HEAD_DIM;
      int blk = block_table[seq * max_blocks + t / PAGE_SIZE];
      long off = (((long)blk * PAGE_SIZE + t % PAGE_SIZE) * num_kv_heads + kv_head) * HEAD_DIM + d;
      tile_k[i / HEAD_DIM][d] = to_f(k[off]); tile_v[i / HEAD_DIM][d] = to_f(v[off]);
    }
    __syncthreads();
    for (int t = 0; t < n; t++) {
      float dot = 0.f; for (int d = 0; d < HEAD_DIM; d++) dot += qv[d] * tile_k[t][d];
      float score = dot * scale, nm = fmaxf(m, score), p = expf(score - nm), corr = expf(m - nm);
      for (int d = 0; d < HEAD_DIM; d++) acc[d] = acc[d] * corr + p * tile_v[t][d];
      l = l * corr + p; m = nm;
    }
    __syncthreads();
  }
  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x)
    out[(((long)seq * num_heads + head) * q_len + qt) * HEAD_DIM + d] = dtype_t(l > 0 ? acc[d] / l : 0.f);
}

torch::Tensor paged_prefill(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                            torch::Tensor block_table, torch::Tensor seq_lens) {
  int B = q.size(0), H = q.size(1), T = q.size(2), HKV = k.size(2);
  auto out = torch::empty_like(q);
  paged_prefill_kernel__SPEC_SUFFIX__<<<dim3(B, T, H), 128>>>((dtype_t*)q.data_ptr(), (dtype_t*)k.data_ptr(), (dtype_t*)v.data_ptr(),
    block_table.data_ptr<int>(), seq_lens.data_ptr<int>(), (dtype_t*)out.data_ptr(),
    H, HKV, (int)block_table.size(1), T, 1.0f / sqrtf((float)HEAD_DIM));
  return out;
}
