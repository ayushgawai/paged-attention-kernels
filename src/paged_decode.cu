// Paged decode: GQA + online softmax + split-KV. One block per (seq, q-head, split);
// threads cooperate over HEAD_DIM. A reduction kernel merges splits via log-sum-exp.
// Template params are substituted by paged_attention/jit.py (__HEAD_DIM__ etc.).
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#define HEAD_DIM __HEAD_DIM__
#define PAGE_SIZE __PAGE_SIZE__
typedef __DTYPE__ dtype_t;

__device__ float to_f(dtype_t x) {
#if defined(__CUDA_ARCH__)
  return float(x);
#else
  return float(x);
#endif
}

extern "C" __global__ void paged_decode_kernel__SPEC_SUFFIX__(
    const dtype_t* __restrict__ q, const dtype_t* __restrict__ k, const dtype_t* __restrict__ v,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    float* __restrict__ partial_out, float* __restrict__ partial_lse,
    int num_heads, int num_kv_heads, int max_blocks, int num_splits, float scale) {
  int seq = blockIdx.x, head = blockIdx.y, split = blockIdx.z;
  int kv_head = head / (num_heads / num_kv_heads);
  int len = seq_lens[seq];
  int chunk = (len + num_splits - 1) / num_splits;
  int begin = split * chunk, end = min(len, begin + chunk);
  float qv[HEAD_DIM];
  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) qv[d] = to_f(q[((long)seq * num_heads + head) * HEAD_DIM + d]);
  __shared__ float sdata[1024];
  float m = -1e30f, l = 0.f, acc[HEAD_DIM];
  for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.f;
  for (int t = begin; t < end; t++) {
    int blk = block_table[seq * max_blocks + t / PAGE_SIZE];
    long koff = (((long)blk * PAGE_SIZE + t % PAGE_SIZE) * num_kv_heads + kv_head) * HEAD_DIM;
    float dot = 0.f;
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) dot += qv[d] * to_f(k[koff + d]);
    sdata[threadIdx.x] = dot; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; __syncthreads(); }
    float score = sdata[0] * scale; __syncthreads();
    float nm = fmaxf(m, score), p = expf(score - nm), corr = expf(m - nm);
    long voff = koff;
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = acc[d] * corr + p * to_f(v[voff + d]);
    l = l * corr + p; m = nm;
  }
  float* po = partial_out + (((long)seq * num_heads + head) * num_splits + split) * HEAD_DIM;
  if (threadIdx.x == 0) partial_lse[((long)seq * num_heads + head) * num_splits + split] = (l > 0 ? m + logf(l) : -1e30f);
  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) po[d] = (l > 0 ? acc[d] / l : 0.f);
}

extern "C" __global__ void paged_reduce_kernel__SPEC_SUFFIX__(
    const float* __restrict__ partial_out, const float* __restrict__ partial_lse,
    dtype_t* __restrict__ out, int num_heads, int num_splits) {
  int seq = blockIdx.x, head = blockIdx.y;
  float mx = -1e30f;
  for (int s = 0; s < num_splits; s++) mx = fmaxf(mx, partial_lse[((long)seq * num_heads + head) * num_splits + s]);
  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
    float num = 0.f, den = 0.f;
    for (int s = 0; s < num_splits; s++) {
      float w = expf(partial_lse[((long)seq * num_heads + head) * num_splits + s] - mx);
      num += w * partial_out[(((long)seq * num_heads + head) * num_splits + s) * HEAD_DIM + d]; den += w;
    }
    out[((long)seq * num_heads + head) * HEAD_DIM + d] = dtype_t(den > 0 ? num / den : 0.f);
  }
}

// Host launcher signature consumed by load_inline (name: paged_decode).
torch::Tensor paged_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           torch::Tensor block_table, torch::Tensor seq_lens, int64_t num_splits) {
  int B = q.size(0), H = q.size(1), HKV = k.size(2);
  auto out = torch::empty_like(q);
  auto po = torch::empty({B, H, num_splits, HEAD_DIM}, q.options().dtype(torch::kFloat32));
  auto pl = torch::empty({B, H, num_splits}, q.options().dtype(torch::kFloat32));
  dim3 grid(B, H, (unsigned)num_splits);
  paged_decode_kernel__SPEC_SUFFIX__<<<grid, 128>>>((dtype_t*)q.data_ptr(), (dtype_t*)k.data_ptr(), (dtype_t*)v.data_ptr(),
    block_table.data_ptr<int>(), seq_lens.data_ptr<int>(), po.data_ptr<float>(), pl.data_ptr<float>(),
    H, HKV, (int)block_table.size(1), (int)num_splits, 1.0f / sqrtf((float)HEAD_DIM));
  paged_reduce_kernel__SPEC_SUFFIX__<<<dim3(B, H), 128>>>(po.data_ptr<float>(), pl.data_ptr<float>(), (dtype_t*)out.data_ptr(), H, (int)num_splits);
  return out;
}
