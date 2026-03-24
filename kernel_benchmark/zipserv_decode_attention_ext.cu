#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

#include "L_API.cuh"
#include "../csrc/L_Kernel.cuh"

namespace {

constexpr int kTopN = 7;
constexpr int kTileM = 64;
constexpr int kTileK = 64;
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;
constexpr int kSoftmaxThreads = 256;
constexpr int kGroupedQHeadsPerCta = 2;
constexpr int kThreadsPerGroupedQHead = kThreadsPerBlock / kGroupedQHeadsPerCta;
constexpr int kTokenGroupsPerGroupedQHead = kThreadsPerGroupedQHead / 16;
constexpr int kMaxOutputAccumsPerThread = 256 / kThreadsPerGroupedQHead;

static_assert(kThreadsPerBlock % kGroupedQHeadsPerCta == 0, "invalid grouped-head CTA shape");
static_assert(kThreadsPerGroupedQHead % 16 == 0, "grouped-head CTA requires 16-thread token groups");
static_assert(256 % kThreadsPerGroupedQHead == 0, "grouped-head CTA assumes exact head-dim coverage up to 256");

#define CUDA_CHECK(expr) TORCH_CHECK((expr) == cudaSuccess, #expr " failed")

size_t ComputeSharedBytes(int max_high_freq_count, int max_full_count) {
    using Config = TilingConfigBF16TripleBitmap<4, 1, 4>;
    const int bitmap_size = Config::TILE_BITMAP_M_V3 * Config::TILE_BITMAP_K_V3;
    return static_cast<size_t>(kTileM) * (kTileK + PADDING_SHARED_MEM_FOR_DECOMP) * sizeof(__nv_bfloat16) +
           static_cast<size_t>(bitmap_size) * 3 * sizeof(uint64_t) +
           static_cast<size_t>(max_high_freq_count + ((128 - (max_high_freq_count % 128)) % 128)) * sizeof(uint8_t) +
           static_cast<size_t>(max_full_count) * sizeof(__nv_bfloat16);
}

size_t ComputeCompressedBytes(int rows, int cols, int num_global_tiles, int total_hf, int total_full) {
    const int num_tiles = (rows / 8) * (cols / 8);
    const int num_median_tiles = (rows / 16) * (cols / 64);
    return static_cast<size_t>(total_hf) +
           static_cast<size_t>(total_full) * sizeof(__nv_bfloat16) +
           static_cast<size_t>(num_tiles) * 3 * sizeof(uint64_t) +
           static_cast<size_t>(num_tiles) * 2 * sizeof(int) +
           static_cast<size_t>(num_median_tiles) * 2 * sizeof(int) +
           static_cast<size_t>(num_global_tiles + 1) * 2 * sizeof(int) +
           kTopN * sizeof(int);
}

int RoundUpToWarp(int value) {
    const int rounded = ((value + kWarpSize - 1) / kWarpSize) * kWarpSize;
    return std::max(kWarpSize, rounded);
}

__global__ void AnalyzeTopExponentsFusedKernel(const __nv_bfloat16* d_input,
                                               int numel,
                                               int* d_counts,
                                               int* d_top_exponents,
                                               int* d_done_counter,
                                               int expected_blocks) {
    __shared__ int s_counts[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();

    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = global_tid; idx < numel; idx += stride) {
        const uint16_t bits = __bfloat16_as_ushort(d_input[idx]);
        const int exponent = (bits >> 7) & 0xFF;
        atomicAdd(&s_counts[exponent], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        atomicAdd(&d_counts[i], s_counts[i]);
    }
    __threadfence();
    const int done = atomicAdd(d_done_counter, 1) + 1;
    if (done != expected_blocks) {
        return;
    }

    int top_exp[kTopN];
    int top_cnt[kTopN];
    for (int i = 0; i < kTopN; ++i) {
        top_exp[i] = -1;
        top_cnt[i] = -1;
    }
    for (int e = 0; e < 256; ++e) {
        int min_idx = 0;
        for (int i = 1; i < kTopN; ++i) {
            if (top_cnt[i] < top_cnt[min_idx]) min_idx = i;
        }
        const int count = d_counts[e];
        if (count > top_cnt[min_idx]) {
            top_cnt[min_idx] = count;
            top_exp[min_idx] = e;
        }
    }

    int original_top[kTopN];
    int n_found = 0;
    for (int i = 0; i < kTopN; ++i) {
        if (top_exp[i] >= 0) original_top[n_found++] = top_exp[i];
    }
    while (n_found < kTopN) {
        original_top[n_found] = 127 - n_found;
        ++n_found;
    }

    for (int i = 1; i < kTopN; ++i) {
        const int key = original_top[i];
        int j = i - 1;
        while (j >= 0 && original_top[j] > key) {
            original_top[j + 1] = original_top[j];
            --j;
        }
        original_top[j + 1] = key;
    }

    for (int ci = 0; ci < kTopN; ++ci) {
        const int start = original_top[ci];
        if (start < 0 || start + kTopN - 1 > 255) continue;
        bool all_exist = true;
        for (int i = 0; i < kTopN; ++i) {
            if (d_counts[start + i] <= 0) {
                all_exist = false;
                break;
            }
        }
        if (!all_exist) continue;
        for (int i = 0; i < kTopN; ++i) d_top_exponents[i] = start + i;
        return;
    }

    int best_start = original_top[0];
    int max_length = 1;
    int current_length = 1;
    for (int i = 1; i < kTopN; ++i) {
        if (original_top[i] == original_top[i - 1] + 1) {
            ++current_length;
            if (current_length > max_length) {
                max_length = current_length;
                best_start = original_top[i] - current_length + 1;
            }
        } else {
            current_length = 1;
        }
    }
    if (best_start < 0) best_start = 0;
    if (best_start > 255 - kTopN) best_start = 255 - kTopN;
    for (int i = 0; i < kTopN; ++i) d_top_exponents[i] = best_start + i;
}

cudaError_t AnalyzeTopExponentsGPU(cudaStream_t stream,
                                   const __nv_bfloat16* d_input,
                                   int numel,
                                   int* d_exponent_counts,
                                   int* d_top_exponents,
                                   int* d_analysis_done_counter) {
    if (numel <= 0) return cudaErrorInvalidValue;
    cudaError_t ce = cudaMemsetAsync(d_exponent_counts, 0, 256 * sizeof(int), stream);
    if (ce != cudaSuccess) return ce;
    ce = cudaMemsetAsync(d_analysis_done_counter, 0, sizeof(int), stream);
    if (ce != cudaSuccess) return ce;
    const int threads = 256;
    const int blocks = std::max(1, std::min((numel + threads - 1) / threads, 1024));
    AnalyzeTopExponentsFusedKernel<<<blocks, threads, 0, stream>>>(
        d_input, numel, d_exponent_counts, d_top_exponents, d_analysis_done_counter, blocks);
    return cudaGetLastError();
}

template <typename TilingConfig>
__device__ __forceinline__ void LoadZipservGlobalTileToShared(
    const uint8_t* __restrict__ sign_mantissa,
    const __nv_bfloat16* __restrict__ compressed_full,
    const uint64_t* __restrict__ bitmap1,
    const uint64_t* __restrict__ bitmap2,
    const uint64_t* __restrict__ bitmap3,
    const int* __restrict__ tile_offsets_median,
    const int* __restrict__ tile_offsets_global,
    int max_high_freq_count,
    int max_full_count,
    uint8_t start_exp,
    int rows,
    int cols,
    int global_tile_m,
    int global_tile_k,
    __nv_bfloat16 (*smem_output)[64 + PADDING_SHARED_MEM_FOR_DECOMP],
    uint64_t* smem_bitmap1,
    uint64_t* smem_bitmap2,
    uint64_t* smem_bitmap3,
    uint8_t* smem_sign_mantissa,
    __nv_bfloat16* smem_full_values) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int global_tile_idx = global_tile_m * (cols / 64) + global_tile_k;
    const int bitmap_global_offset = ((global_tile_m * 64) >> 3) * (cols >> 3) + global_tile_k * 64;
    const uint64_t* bitmap1_ptr = bitmap1 + bitmap_global_offset;
    const uint64_t* bitmap2_ptr = bitmap2 + bitmap_global_offset;
    const uint64_t* bitmap3_ptr = bitmap3 + bitmap_global_offset;
    const int* global_offset_ptr = tile_offsets_global + global_tile_idx * 2;
    const int global_hf_start = global_offset_ptr[0];
    const int global_full_start = global_offset_ptr[1];
    const int global_hf_count = global_offset_ptr[2] - global_hf_start;
    const int global_full_count = global_offset_ptr[3] - global_full_start;

    CopyTripleBitmapToShared<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
        smem_bitmap1, smem_bitmap2, smem_bitmap3, bitmap1_ptr, bitmap2_ptr, bitmap3_ptr);
    CopyCompressedDataToShared<TilingConfig>(
        smem_sign_mantissa, smem_full_values,
        sign_mantissa + global_hf_start,
        compressed_full + global_full_start,
        global_hf_count,
        global_full_count);
    cp_async_group_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    const int median_tile_idx = global_tile_idx * 4 + warp_id;
    const int* median_offset_ptr = tile_offsets_median + median_tile_idx * 2;
    int warp_hf_start = median_offset_ptr[0];
    int warp_full_start = median_offset_ptr[1];
    uint64_t* smem_bitmap1_warp = smem_bitmap1 + warp_id * 2 * 8;
    uint64_t* smem_bitmap2_warp = smem_bitmap2 + warp_id * 2 * 8;
    uint64_t* smem_bitmap3_warp = smem_bitmap3 + warp_id * 2 * 8;

    DecompressMedianTileToSharedMemory<TilingConfig>(
        smem_sign_mantissa,
        smem_full_values,
        smem_bitmap1_warp,
        smem_bitmap2_warp,
        smem_bitmap3_warp,
        warp_hf_start,
        warp_full_start,
        start_exp,
        smem_output,
        warp_id);
    __syncthreads();
}

template <typename TilingConfig>
__global__ void ZipservDecodeScoresKernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ sign_mantissa,
    const __nv_bfloat16* __restrict__ compressed_full,
    const uint64_t* __restrict__ bitmap1,
    const uint64_t* __restrict__ bitmap2,
    const uint64_t* __restrict__ bitmap3,
    const int* __restrict__ tile_offsets_median,
    const int* __restrict__ tile_offsets_global,
    int rows,
    int cols,
    int max_high_freq_count,
    int max_full_count,
    uint8_t start_exp,
    int logical_kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    float* __restrict__ scores) {
    const int q_head = blockIdx.x;
    const int row_tile = blockIdx.y;
    const int kv_group_size = num_q_heads / num_kv_heads;
    const int kv_head = q_head / kv_group_size;
    const int token_group = threadIdx.x / 16;
    const int lane_in_group = threadIdx.x % 16;
    const int token = row_tile * 8 + token_group;
    const int local_row = kv_head + token_group * 8;

    extern __shared__ __align__(128) __nv_bfloat16 smem_buffer[];
    auto* smem_output = reinterpret_cast<__nv_bfloat16(*)[64 + PADDING_SHARED_MEM_FOR_DECOMP]>(smem_buffer);
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    uint64_t* smem_bitmap1 = reinterpret_cast<uint64_t*>(smem_output + 64);
    uint64_t* smem_bitmap2 = smem_bitmap1 + bitmap_size;
    uint64_t* smem_bitmap3 = smem_bitmap2 + bitmap_size;
    uint8_t* smem_sign_mantissa = reinterpret_cast<uint8_t*>(smem_bitmap3 + bitmap_size);
    const size_t padding = (128 - (max_high_freq_count % 128)) % 128;
    __nv_bfloat16* smem_full_values =
        reinterpret_cast<__nv_bfloat16*>(smem_sign_mantissa + max_high_freq_count + padding);

    __shared__ float token_scores[8];
    if (threadIdx.x < 8) token_scores[threadIdx.x] = 0.0f;
    __syncthreads();

    const int col_tiles = cols / 64;
    for (int col_tile = 0; col_tile < col_tiles; ++col_tile) {
        LoadZipservGlobalTileToShared<TilingConfig>(
            sign_mantissa,
            compressed_full,
            bitmap1,
            bitmap2,
            bitmap3,
            tile_offsets_median,
            tile_offsets_global,
            max_high_freq_count,
            max_full_count,
            start_exp,
            rows,
            cols,
            row_tile,
            col_tile,
            smem_output,
            smem_bitmap1,
            smem_bitmap2,
            smem_bitmap3,
            smem_sign_mantissa,
            smem_full_values);

        float acc = 0.0f;
        if (token < logical_kv_len) {
            const int col0 = lane_in_group + 0;
            const int col1 = lane_in_group + 16;
            const int col2 = lane_in_group + 32;
            const int col3 = lane_in_group + 48;
            const int q_base = q_head * head_dim + col_tile * 64;
            acc += __bfloat162float(q[q_base + col0]) * __bfloat162float(smem_output[local_row][col0]);
            acc += __bfloat162float(q[q_base + col1]) * __bfloat162float(smem_output[local_row][col1]);
            acc += __bfloat162float(q[q_base + col2]) * __bfloat162float(smem_output[local_row][col2]);
            acc += __bfloat162float(q[q_base + col3]) * __bfloat162float(smem_output[local_row][col3]);
        }

        for (int offset = 8; offset > 0; offset /= 2) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
        }
        if (lane_in_group == 0 && token < logical_kv_len) {
            token_scores[token_group] += acc;
        }
        __syncthreads();
    }

    if (threadIdx.x < 8) {
        const int out_token = row_tile * 8 + threadIdx.x;
        if (out_token < logical_kv_len) {
            scores[q_head * logical_kv_len + out_token] = token_scores[threadIdx.x] * sm_scale;
        }
    }
}

template <typename TilingConfig>
__global__ void ZipservDecodeValuesKernel(
    const float* __restrict__ probs,
    const uint8_t* __restrict__ sign_mantissa,
    const __nv_bfloat16* __restrict__ compressed_full,
    const uint64_t* __restrict__ bitmap1,
    const uint64_t* __restrict__ bitmap2,
    const uint64_t* __restrict__ bitmap3,
    const int* __restrict__ tile_offsets_median,
    const int* __restrict__ tile_offsets_global,
    int rows,
    int cols,
    int max_high_freq_count,
    int max_full_count,
    uint8_t start_exp,
    int logical_kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float* __restrict__ output) {
    const int q_head = blockIdx.x;
    const int kv_group_size = num_q_heads / num_kv_heads;
    const int kv_head = q_head / kv_group_size;
    const int dim = threadIdx.x;
    if (dim >= head_dim) return;

    extern __shared__ __align__(128) __nv_bfloat16 smem_buffer[];
    auto* smem_output = reinterpret_cast<__nv_bfloat16(*)[64 + PADDING_SHARED_MEM_FOR_DECOMP]>(smem_buffer);
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    uint64_t* smem_bitmap1 = reinterpret_cast<uint64_t*>(smem_output + 64);
    uint64_t* smem_bitmap2 = smem_bitmap1 + bitmap_size;
    uint64_t* smem_bitmap3 = smem_bitmap2 + bitmap_size;
    uint8_t* smem_sign_mantissa = reinterpret_cast<uint8_t*>(smem_bitmap3 + bitmap_size);
    const size_t padding = (128 - (max_high_freq_count % 128)) % 128;
    __nv_bfloat16* smem_full_values =
        reinterpret_cast<__nv_bfloat16*>(smem_sign_mantissa + max_high_freq_count + padding);

    float acc = 0.0f;
    const int row_tiles = rows / 64;
    const int col_tile = dim / 64;
    const int local_col = dim % 64;

    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        LoadZipservGlobalTileToShared<TilingConfig>(
            sign_mantissa,
            compressed_full,
            bitmap1,
            bitmap2,
            bitmap3,
            tile_offsets_median,
            tile_offsets_global,
            max_high_freq_count,
            max_full_count,
            start_exp,
            rows,
            cols,
            row_tile,
            col_tile,
            smem_output,
            smem_bitmap1,
            smem_bitmap2,
            smem_bitmap3,
            smem_sign_mantissa,
            smem_full_values);

        #pragma unroll
        for (int token_in_tile = 0; token_in_tile < 8; ++token_in_tile) {
            const int token = row_tile * 8 + token_in_tile;
            if (token >= logical_kv_len) break;
            const int local_row = kv_head + token_in_tile * 8;
            const float p = probs[q_head * logical_kv_len + token];
            acc += p * __bfloat162float(smem_output[local_row][local_col]);
        }
        __syncthreads();
    }
    output[q_head * head_dim + dim] = acc;
}

template <typename TilingConfig>
__global__ void ZipservDecodeAttentionOnlineKernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_sign_mantissa,
    const __nv_bfloat16* __restrict__ k_compressed_full,
    const uint64_t* __restrict__ k_bitmap1,
    const uint64_t* __restrict__ k_bitmap2,
    const uint64_t* __restrict__ k_bitmap3,
    const int* __restrict__ k_tile_offsets_median,
    const int* __restrict__ k_tile_offsets_global,
    int k_rows,
    int k_cols,
    int k_max_high_freq_count,
    int k_max_full_count,
    uint8_t k_start_exp,
    const uint8_t* __restrict__ v_sign_mantissa,
    const __nv_bfloat16* __restrict__ v_compressed_full,
    const uint64_t* __restrict__ v_bitmap1,
    const uint64_t* __restrict__ v_bitmap2,
    const uint64_t* __restrict__ v_bitmap3,
    const int* __restrict__ v_tile_offsets_median,
    const int* __restrict__ v_tile_offsets_global,
    int v_rows,
    int v_cols,
    int v_max_high_freq_count,
    int v_max_full_count,
    uint8_t v_start_exp,
    int logical_kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    float* __restrict__ output) {
    const int q_head = blockIdx.x;
    const int kv_group_size = num_q_heads / num_kv_heads;
    const int kv_head = q_head / kv_group_size;
    const int token_group = threadIdx.x / 16;
    const int lane_in_group = threadIdx.x % 16;
    const int token_stride = kTileM / num_kv_heads;
    const int dim0 = threadIdx.x;
    const int dim1 = threadIdx.x + kThreadsPerBlock;

    extern __shared__ __align__(128) __nv_bfloat16 smem_buffer[];
    auto* smem_output = reinterpret_cast<__nv_bfloat16(*)[64 + PADDING_SHARED_MEM_FOR_DECOMP]>(smem_buffer);
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    uint64_t* smem_bitmap1 = reinterpret_cast<uint64_t*>(smem_output + 64);
    uint64_t* smem_bitmap2 = smem_bitmap1 + bitmap_size;
    uint64_t* smem_bitmap3 = smem_bitmap2 + bitmap_size;
    uint8_t* smem_sign_mantissa = reinterpret_cast<uint8_t*>(smem_bitmap3 + bitmap_size);
    const size_t shared_hf_count = static_cast<size_t>(
        k_max_high_freq_count > v_max_high_freq_count ? k_max_high_freq_count : v_max_high_freq_count);
    const size_t shared_full_count = static_cast<size_t>(
        k_max_full_count > v_max_full_count ? k_max_full_count : v_max_full_count);
    const size_t padding = (128 - (shared_hf_count % 128)) % 128;
    __nv_bfloat16* smem_full_values =
        reinterpret_cast<__nv_bfloat16*>(smem_sign_mantissa + shared_hf_count + padding);
    __nv_bfloat16* smem_q_group = smem_full_values + shared_full_count;

    __shared__ float tile_scores[8];
    __shared__ float tile_weights[8];
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float output_scale;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    if (threadIdx.x == 0) {
        running_max = -FLT_MAX;
        running_sum = 0.0f;
        output_scale = 0.0f;
    }
    __syncthreads();

    const int k_col_tiles = k_cols / kTileK;
    const int v_col_tiles = v_cols / kTileK;
    const int row_tiles = (logical_kv_len + token_stride - 1) / token_stride;
    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        if (threadIdx.x < token_stride) {
            tile_scores[threadIdx.x] = 0.0f;
            tile_weights[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int col_tile = 0; col_tile < k_col_tiles; ++col_tile) {
            LoadZipservGlobalTileToShared<TilingConfig>(
                k_sign_mantissa,
                k_compressed_full,
                k_bitmap1,
                k_bitmap2,
                k_bitmap3,
                k_tile_offsets_median,
                k_tile_offsets_global,
                k_max_high_freq_count,
                k_max_full_count,
                k_start_exp,
                k_rows,
                k_cols,
                row_tile,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            const int token = row_tile * token_stride + token_group;
            float partial = 0.0f;
            if (token < logical_kv_len) {
                const int local_row = kv_head + token_group * num_kv_heads;
                const int q_base = q_head * head_dim + col_tile * kTileK;
                partial += __bfloat162float(q[q_base + lane_in_group]) *
                           __bfloat162float(smem_output[local_row][lane_in_group]);
                partial += __bfloat162float(q[q_base + lane_in_group + 16]) *
                           __bfloat162float(smem_output[local_row][lane_in_group + 16]);
                partial += __bfloat162float(q[q_base + lane_in_group + 32]) *
                           __bfloat162float(smem_output[local_row][lane_in_group + 32]);
                partial += __bfloat162float(q[q_base + lane_in_group + 48]) *
                           __bfloat162float(smem_output[local_row][lane_in_group + 48]);
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(0xFFFFFFFF, partial, offset, 16);
            }
            if (lane_in_group == 0 && token < logical_kv_len) {
                tile_scores[token_group] += partial;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            const int tile_token_start = row_tile * token_stride;
            float tile_max = -FLT_MAX;
            for (int token_in_tile = 0; token_in_tile < token_stride; ++token_in_tile) {
                const int token = tile_token_start + token_in_tile;
                if (token >= logical_kv_len) {
                    tile_scores[token_in_tile] = -FLT_MAX;
                    tile_weights[token_in_tile] = 0.0f;
                    continue;
                }
                const float score = tile_scores[token_in_tile] * sm_scale;
                tile_scores[token_in_tile] = score;
                tile_max = fmaxf(tile_max, score);
            }

            const float next_max = fmaxf(running_max, tile_max);
            output_scale = expf(running_max - next_max);
            running_sum *= output_scale;
            for (int token_in_tile = 0; token_in_tile < token_stride; ++token_in_tile) {
                const int token = tile_token_start + token_in_tile;
                if (token >= logical_kv_len) {
                    tile_weights[token_in_tile] = 0.0f;
                    continue;
                }
                const float weight = expf(tile_scores[token_in_tile] - next_max);
                tile_weights[token_in_tile] = weight;
                running_sum += weight;
            }
            running_max = next_max;
        }
        __syncthreads();

        acc0 *= output_scale;
        acc1 *= output_scale;

        for (int col_tile = 0; col_tile < v_col_tiles; ++col_tile) {
            LoadZipservGlobalTileToShared<TilingConfig>(
                v_sign_mantissa,
                v_compressed_full,
                v_bitmap1,
                v_bitmap2,
                v_bitmap3,
                v_tile_offsets_median,
                v_tile_offsets_global,
                v_max_high_freq_count,
                v_max_full_count,
                v_start_exp,
                v_rows,
                v_cols,
                row_tile,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            if (dim0 < head_dim && (dim0 / kTileK) == col_tile) {
                const int local_col0 = dim0 % kTileK;
                for (int token_in_tile = 0; token_in_tile < token_stride; ++token_in_tile) {
                    const int token = row_tile * token_stride + token_in_tile;
                    if (token >= logical_kv_len) break;
                    const int local_row = kv_head + token_in_tile * num_kv_heads;
                    acc0 += tile_weights[token_in_tile] *
                            __bfloat162float(smem_output[local_row][local_col0]);
                }
            }
            if (dim1 < head_dim && (dim1 / kTileK) == col_tile) {
                const int local_col1 = dim1 % kTileK;
                for (int token_in_tile = 0; token_in_tile < token_stride; ++token_in_tile) {
                    const int token = row_tile * token_stride + token_in_tile;
                    if (token >= logical_kv_len) break;
                    const int local_row = kv_head + token_in_tile * num_kv_heads;
                    acc1 += tile_weights[token_in_tile] *
                            __bfloat162float(smem_output[local_row][local_col1]);
                }
            }
            __syncthreads();
        }
    }

    const float inv_sum = 1.0f / running_sum;
    if (dim0 < head_dim) {
        output[q_head * head_dim + dim0] = acc0 * inv_sum;
    }
    if (dim1 < head_dim) {
        output[q_head * head_dim + dim1] = acc1 * inv_sum;
    }
}

template <typename TilingConfig>
__global__ void ZipservDecodeAttentionOnlineBatchedKernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_sign_mantissa,
    const __nv_bfloat16* __restrict__ k_compressed_full,
    const uint64_t* __restrict__ k_bitmap1,
    const uint64_t* __restrict__ k_bitmap2,
    const uint64_t* __restrict__ k_bitmap3,
    const int* __restrict__ k_tile_offsets_median,
    const int* __restrict__ k_tile_offsets_global,
    int k_rows,
    int k_cols,
    int k_batch_stride_rows,
    int k_head_stride_rows,
    int k_max_high_freq_count,
    int k_max_full_count,
    uint8_t k_start_exp,
    const uint8_t* __restrict__ v_sign_mantissa,
    const __nv_bfloat16* __restrict__ v_compressed_full,
    const uint64_t* __restrict__ v_bitmap1,
    const uint64_t* __restrict__ v_bitmap2,
    const uint64_t* __restrict__ v_bitmap3,
    const int* __restrict__ v_tile_offsets_median,
    const int* __restrict__ v_tile_offsets_global,
    int v_rows,
    int v_cols,
    int v_batch_stride_rows,
    int v_head_stride_rows,
    int v_max_high_freq_count,
    int v_max_full_count,
    uint8_t v_start_exp,
    int logical_kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    float* __restrict__ output) {
    const int q_head = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int kv_group_size = num_q_heads / num_kv_heads;
    const int kv_head = q_head / kv_group_size;
    const int token_group = threadIdx.x / 16;
    const int lane_in_group = threadIdx.x % 16;
    const int dim0 = threadIdx.x;
    const int dim1 = threadIdx.x + kThreadsPerBlock;
    const int batch_stride_tiles = k_batch_stride_rows / kTileM;
    const int head_stride_tiles = k_head_stride_rows / kTileM;
    const int v_batch_stride_tiles = v_batch_stride_rows / kTileM;
    const int v_head_stride_tiles = v_head_stride_rows / kTileM;

    extern __shared__ __align__(128) __nv_bfloat16 smem_buffer[];
    auto* smem_output = reinterpret_cast<__nv_bfloat16(*)[64 + PADDING_SHARED_MEM_FOR_DECOMP]>(smem_buffer);
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    uint64_t* smem_bitmap1 = reinterpret_cast<uint64_t*>(smem_output + 64);
    uint64_t* smem_bitmap2 = smem_bitmap1 + bitmap_size;
    uint64_t* smem_bitmap3 = smem_bitmap2 + bitmap_size;
    uint8_t* smem_sign_mantissa = reinterpret_cast<uint8_t*>(smem_bitmap3 + bitmap_size);
    const size_t shared_hf_count = static_cast<size_t>(
        k_max_high_freq_count > v_max_high_freq_count ? k_max_high_freq_count : v_max_high_freq_count);
    const size_t shared_full_count = static_cast<size_t>(
        k_max_full_count > v_max_full_count ? k_max_full_count : v_max_full_count);
    const size_t padding = (128 - (shared_hf_count % 128)) % 128;
    __nv_bfloat16* smem_full_values =
        reinterpret_cast<__nv_bfloat16*>(smem_sign_mantissa + shared_hf_count + padding);
    float* smem_q_group = reinterpret_cast<float*>(smem_full_values + shared_full_count);

    __shared__ float tile_scores[64];
    __shared__ float tile_weights[64];
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float output_scale;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    if (threadIdx.x == 0) {
        running_max = -FLT_MAX;
        running_sum = 0.0f;
        output_scale = 0.0f;
    }
    for (int idx = threadIdx.x; idx < 64; idx += blockDim.x) {
        tile_scores[idx] = 0.0f;
        tile_weights[idx] = 0.0f;
    }
    __syncthreads();

    const int q_batch_stride = num_q_heads * head_dim;
    const int output_batch_stride = num_q_heads * head_dim;
    const int q_head_base = batch_idx * q_batch_stride + q_head * head_dim;
    const int output_head_base = batch_idx * output_batch_stride + q_head * head_dim;
    const int k_col_tiles = k_cols / kTileK;
    const int v_col_tiles = v_cols / kTileK;
    const int row_tiles = (logical_kv_len + kTileM - 1) / kTileM;
    const int k_row_tile_base = batch_idx * batch_stride_tiles + kv_head * head_stride_tiles;
    const int v_row_tile_base = batch_idx * v_batch_stride_tiles + kv_head * v_head_stride_tiles;

    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (int idx = threadIdx.x; idx < 64; idx += blockDim.x) {
            tile_scores[idx] = 0.0f;
            tile_weights[idx] = 0.0f;
        }
        __syncthreads();

        for (int col_tile = 0; col_tile < k_col_tiles; ++col_tile) {
            LoadZipservGlobalTileToShared<TilingConfig>(
                k_sign_mantissa,
                k_compressed_full,
                k_bitmap1,
                k_bitmap2,
                k_bitmap3,
                k_tile_offsets_median,
                k_tile_offsets_global,
                k_max_high_freq_count,
                k_max_full_count,
                k_start_exp,
                k_rows,
                k_cols,
                k_row_tile_base + row_tile,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            #pragma unroll
            for (int token_chunk = 0; token_chunk < 64; token_chunk += 8) {
                const int token_in_tile = token_chunk + token_group;
                const int token = row_tile * kTileM + token_in_tile;
                float partial = 0.0f;
                if (token < logical_kv_len) {
                    const int q_base = q_head_base + col_tile * kTileK;
                    partial += __bfloat162float(q[q_base + lane_in_group]) *
                               __bfloat162float(smem_output[token_in_tile][lane_in_group]);
                    partial += __bfloat162float(q[q_base + lane_in_group + 16]) *
                               __bfloat162float(smem_output[token_in_tile][lane_in_group + 16]);
                    partial += __bfloat162float(q[q_base + lane_in_group + 32]) *
                               __bfloat162float(smem_output[token_in_tile][lane_in_group + 32]);
                    partial += __bfloat162float(q[q_base + lane_in_group + 48]) *
                               __bfloat162float(smem_output[token_in_tile][lane_in_group + 48]);
                }
                for (int offset = 8; offset > 0; offset /= 2) {
                    partial += __shfl_down_sync(0xFFFFFFFF, partial, offset, 16);
                }
                if (lane_in_group == 0 && token < logical_kv_len) {
                    tile_scores[token_in_tile] += partial;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            const int tile_token_start = row_tile * kTileM;
            float tile_max = -FLT_MAX;
            for (int token_in_tile = 0; token_in_tile < 64; ++token_in_tile) {
                const int token = tile_token_start + token_in_tile;
                if (token >= logical_kv_len) {
                    tile_scores[token_in_tile] = -FLT_MAX;
                    tile_weights[token_in_tile] = 0.0f;
                    continue;
                }
                const float score = tile_scores[token_in_tile] * sm_scale;
                tile_scores[token_in_tile] = score;
                tile_max = fmaxf(tile_max, score);
            }

            const float next_max = fmaxf(running_max, tile_max);
            output_scale = expf(running_max - next_max);
            running_sum *= output_scale;
            for (int token_in_tile = 0; token_in_tile < 64; ++token_in_tile) {
                const int token = tile_token_start + token_in_tile;
                if (token >= logical_kv_len) {
                    tile_weights[token_in_tile] = 0.0f;
                    continue;
                }
                const float weight = expf(tile_scores[token_in_tile] - next_max);
                tile_weights[token_in_tile] = weight;
                running_sum += weight;
            }
            running_max = next_max;
        }
        __syncthreads();

        acc0 *= output_scale;
        acc1 *= output_scale;

        for (int col_tile = 0; col_tile < v_col_tiles; ++col_tile) {
            LoadZipservGlobalTileToShared<TilingConfig>(
                v_sign_mantissa,
                v_compressed_full,
                v_bitmap1,
                v_bitmap2,
                v_bitmap3,
                v_tile_offsets_median,
                v_tile_offsets_global,
                v_max_high_freq_count,
                v_max_full_count,
                v_start_exp,
                v_rows,
                v_cols,
                v_row_tile_base + row_tile,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            if (dim0 < head_dim && (dim0 / kTileK) == col_tile) {
                const int local_col0 = dim0 % kTileK;
                #pragma unroll
                for (int token_in_tile = 0; token_in_tile < 64; ++token_in_tile) {
                    const int token = row_tile * kTileM + token_in_tile;
                    if (token >= logical_kv_len) break;
                    acc0 += tile_weights[token_in_tile] *
                            __bfloat162float(smem_output[token_in_tile][local_col0]);
                }
            }
            if (dim1 < head_dim && (dim1 / kTileK) == col_tile) {
                const int local_col1 = dim1 % kTileK;
                #pragma unroll
                for (int token_in_tile = 0; token_in_tile < 64; ++token_in_tile) {
                    const int token = row_tile * kTileM + token_in_tile;
                    if (token >= logical_kv_len) break;
                    acc1 += tile_weights[token_in_tile] *
                            __bfloat162float(smem_output[token_in_tile][local_col1]);
                }
            }
            __syncthreads();
        }
    }

    const float inv_sum = 1.0f / running_sum;
    if (dim0 < head_dim) {
        output[output_head_base + dim0] = acc0 * inv_sum;
    }
    if (dim1 < head_dim) {
        output[output_head_base + dim1] = acc1 * inv_sum;
    }
}

template <typename TilingConfig>
__global__ void ZipservDecodeAttentionOnlineGroupedBatchedKernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_sign_mantissa,
    const __nv_bfloat16* __restrict__ k_compressed_full,
    const uint64_t* __restrict__ k_bitmap1,
    const uint64_t* __restrict__ k_bitmap2,
    const uint64_t* __restrict__ k_bitmap3,
    const int* __restrict__ k_tile_offsets_median,
    const int* __restrict__ k_tile_offsets_global,
    int k_rows,
    int k_cols,
    int k_batch_stride_rows,
    int k_head_stride_rows,
    int k_max_high_freq_count,
    int k_max_full_count,
    uint8_t k_start_exp,
    const uint8_t* __restrict__ v_sign_mantissa,
    const __nv_bfloat16* __restrict__ v_compressed_full,
    const uint64_t* __restrict__ v_bitmap1,
    const uint64_t* __restrict__ v_bitmap2,
    const uint64_t* __restrict__ v_bitmap3,
    const int* __restrict__ v_tile_offsets_median,
    const int* __restrict__ v_tile_offsets_global,
    int v_rows,
    int v_cols,
    int v_batch_stride_rows,
    int v_head_stride_rows,
    int v_max_high_freq_count,
    int v_max_full_count,
    uint8_t v_start_exp,
    int logical_kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    float* __restrict__ output) {
    constexpr int kQGroupSize = 8;
    const int kv_head = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int token_group = threadIdx.x / 16;
    const int lane_in_group = threadIdx.x % 16;
    const int dim0 = threadIdx.x;
    const int dim1 = threadIdx.x + kThreadsPerBlock;
    const int batch_stride_tiles = k_batch_stride_rows / kTileM;
    const int head_stride_tiles = k_head_stride_rows / kTileM;
    const int v_batch_stride_tiles = v_batch_stride_rows / kTileM;
    const int v_head_stride_tiles = v_head_stride_rows / kTileM;
    const int q_head_group_base = kv_head * kQGroupSize;

    extern __shared__ __align__(128) __nv_bfloat16 smem_buffer[];
    auto* smem_output = reinterpret_cast<__nv_bfloat16(*)[64 + PADDING_SHARED_MEM_FOR_DECOMP]>(smem_buffer);
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    uint64_t* smem_bitmap1 = reinterpret_cast<uint64_t*>(smem_output + 64);
    uint64_t* smem_bitmap2 = smem_bitmap1 + bitmap_size;
    uint64_t* smem_bitmap3 = smem_bitmap2 + bitmap_size;
    uint8_t* smem_sign_mantissa = reinterpret_cast<uint8_t*>(smem_bitmap3 + bitmap_size);
    const size_t shared_hf_count = static_cast<size_t>(
        k_max_high_freq_count > v_max_high_freq_count ? k_max_high_freq_count : v_max_high_freq_count);
    const size_t shared_full_count = static_cast<size_t>(
        k_max_full_count > v_max_full_count ? k_max_full_count : v_max_full_count);
    const size_t padding = (128 - (shared_hf_count % 128)) % 128;
    __nv_bfloat16* smem_full_values =
        reinterpret_cast<__nv_bfloat16*>(smem_sign_mantissa + shared_hf_count + padding);
    float* smem_q_group = reinterpret_cast<float*>(smem_full_values + shared_full_count);

    __shared__ float tile_scores[kQGroupSize][64];
    __shared__ float tile_weights[kQGroupSize][64];
    __shared__ float running_max[kQGroupSize];
    __shared__ float running_sum[kQGroupSize];
    __shared__ float output_scale[kQGroupSize];

    float acc0[kQGroupSize];
    float acc1[kQGroupSize];
    #pragma unroll
    for (int qh_local = 0; qh_local < kQGroupSize; ++qh_local) {
        acc0[qh_local] = 0.0f;
        acc1[qh_local] = 0.0f;
    }

    if (threadIdx.x < kQGroupSize) {
        running_max[threadIdx.x] = -FLT_MAX;
        running_sum[threadIdx.x] = 0.0f;
        output_scale[threadIdx.x] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < kQGroupSize * 64; idx += blockDim.x) {
        tile_scores[idx / 64][idx % 64] = 0.0f;
        tile_weights[idx / 64][idx % 64] = 0.0f;
    }
    __syncthreads();

    const int q_batch_stride = num_q_heads * head_dim;
    const int output_batch_stride = num_q_heads * head_dim;
    const int k_col_tiles = k_cols / kTileK;
    const int v_col_tiles = v_cols / kTileK;
    const int row_tiles = (logical_kv_len + kTileM - 1) / kTileM;
    const int k_row_tile_base = batch_idx * batch_stride_tiles + kv_head * head_stride_tiles;
    const int v_row_tile_base = batch_idx * v_batch_stride_tiles + kv_head * v_head_stride_tiles;
    for (int idx = threadIdx.x; idx < kQGroupSize * head_dim; idx += blockDim.x) {
        const int qh_local = idx / head_dim;
        const int d = idx % head_dim;
        const int q_head = q_head_group_base + qh_local;
        smem_q_group[idx] = __bfloat162float(q[batch_idx * q_batch_stride + q_head * head_dim + d]);
    }
    __syncthreads();

    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (int idx = threadIdx.x; idx < kQGroupSize * 64; idx += blockDim.x) {
            tile_scores[idx / 64][idx % 64] = 0.0f;
            tile_weights[idx / 64][idx % 64] = 0.0f;
        }
        __syncthreads();

        for (int col_tile = 0; col_tile < k_col_tiles; ++col_tile) {
            float qv0[kQGroupSize];
            float qv1[kQGroupSize];
            float qv2[kQGroupSize];
            float qv3[kQGroupSize];
            #pragma unroll
            for (int qh_local = 0; qh_local < kQGroupSize; ++qh_local) {
                const int q_base = qh_local * head_dim + col_tile * kTileK;
                qv0[qh_local] = smem_q_group[q_base + lane_in_group];
                qv1[qh_local] = smem_q_group[q_base + lane_in_group + 16];
                qv2[qh_local] = smem_q_group[q_base + lane_in_group + 32];
                qv3[qh_local] = smem_q_group[q_base + lane_in_group + 48];
            }

            LoadZipservGlobalTileToShared<TilingConfig>(
                k_sign_mantissa,
                k_compressed_full,
                k_bitmap1,
                k_bitmap2,
                k_bitmap3,
                k_tile_offsets_median,
                k_tile_offsets_global,
                k_max_high_freq_count,
                k_max_full_count,
                k_start_exp,
                k_rows,
                k_cols,
                k_row_tile_base + row_tile,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            #pragma unroll
            for (int token_chunk = 0; token_chunk < 64; token_chunk += 8) {
                const int token_in_tile = token_chunk + token_group;
                const int token = row_tile * kTileM + token_in_tile;
                float k0 = 0.0f;
                float k1 = 0.0f;
                float k2 = 0.0f;
                float k3 = 0.0f;
                if (token < logical_kv_len) {
                    k0 = __bfloat162float(smem_output[token_in_tile][lane_in_group]);
                    k1 = __bfloat162float(smem_output[token_in_tile][lane_in_group + 16]);
                    k2 = __bfloat162float(smem_output[token_in_tile][lane_in_group + 32]);
                    k3 = __bfloat162float(smem_output[token_in_tile][lane_in_group + 48]);
                }
                #pragma unroll
                for (int qh_local = 0; qh_local < kQGroupSize; ++qh_local) {
                    float partial = 0.0f;
                    if (token < logical_kv_len) {
                        partial = fmaf(qv0[qh_local], k0, partial);
                        partial = fmaf(qv1[qh_local], k1, partial);
                        partial = fmaf(qv2[qh_local], k2, partial);
                        partial = fmaf(qv3[qh_local], k3, partial);
                    }
                    for (int offset = 8; offset > 0; offset /= 2) {
                        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset, 16);
                    }
                    if (lane_in_group == 0 && token < logical_kv_len) {
                        tile_scores[qh_local][token_in_tile] += partial;
                    }
                }
            }
            __syncthreads();
        }

        if (threadIdx.x < kQGroupSize) {
            const int qh_local = threadIdx.x;
            const int tile_token_start = row_tile * kTileM;
            float tile_max = -FLT_MAX;
            for (int token_in_tile = 0; token_in_tile < 64; ++token_in_tile) {
                const int token = tile_token_start + token_in_tile;
                if (token >= logical_kv_len) {
                    tile_scores[qh_local][token_in_tile] = -FLT_MAX;
                    tile_weights[qh_local][token_in_tile] = 0.0f;
                    continue;
                }
                const float score = tile_scores[qh_local][token_in_tile] * sm_scale;
                tile_scores[qh_local][token_in_tile] = score;
                tile_max = fmaxf(tile_max, score);
            }

            const float next_max = fmaxf(running_max[qh_local], tile_max);
            output_scale[qh_local] = expf(running_max[qh_local] - next_max);
            running_sum[qh_local] *= output_scale[qh_local];
            for (int token_in_tile = 0; token_in_tile < 64; ++token_in_tile) {
                const int token = tile_token_start + token_in_tile;
                if (token >= logical_kv_len) {
                    tile_weights[qh_local][token_in_tile] = 0.0f;
                    continue;
                }
                const float weight = expf(tile_scores[qh_local][token_in_tile] - next_max);
                tile_weights[qh_local][token_in_tile] = weight;
                running_sum[qh_local] += weight;
            }
            running_max[qh_local] = next_max;
        }
        __syncthreads();

        #pragma unroll
        for (int qh_local = 0; qh_local < kQGroupSize; ++qh_local) {
            acc0[qh_local] *= output_scale[qh_local];
            acc1[qh_local] *= output_scale[qh_local];
        }

        for (int col_tile = 0; col_tile < v_col_tiles; ++col_tile) {
            LoadZipservGlobalTileToShared<TilingConfig>(
                v_sign_mantissa,
                v_compressed_full,
                v_bitmap1,
                v_bitmap2,
                v_bitmap3,
                v_tile_offsets_median,
                v_tile_offsets_global,
                v_max_high_freq_count,
                v_max_full_count,
                v_start_exp,
                v_rows,
                v_cols,
                v_row_tile_base + row_tile,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            if (dim0 < head_dim && (dim0 / kTileK) == col_tile) {
                const int local_col0 = dim0 % kTileK;
                #pragma unroll
                for (int token_in_tile = 0; token_in_tile < 64; ++token_in_tile) {
                    const int token = row_tile * kTileM + token_in_tile;
                    if (token >= logical_kv_len) break;
                    const float value = __bfloat162float(smem_output[token_in_tile][local_col0]);
                    #pragma unroll
                    for (int qh_local = 0; qh_local < kQGroupSize; ++qh_local) {
                        acc0[qh_local] = fmaf(tile_weights[qh_local][token_in_tile], value, acc0[qh_local]);
                    }
                }
            }
            if (dim1 < head_dim && (dim1 / kTileK) == col_tile) {
                const int local_col1 = dim1 % kTileK;
                #pragma unroll
                for (int token_in_tile = 0; token_in_tile < 64; ++token_in_tile) {
                    const int token = row_tile * kTileM + token_in_tile;
                    if (token >= logical_kv_len) break;
                    const float value = __bfloat162float(smem_output[token_in_tile][local_col1]);
                    #pragma unroll
                    for (int qh_local = 0; qh_local < kQGroupSize; ++qh_local) {
                        acc1[qh_local] = fmaf(tile_weights[qh_local][token_in_tile], value, acc1[qh_local]);
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int qh_local = 0; qh_local < kQGroupSize; ++qh_local) {
        const float inv_sum = 1.0f / running_sum[qh_local];
        const int q_head = q_head_group_base + qh_local;
        const int output_head_base = batch_idx * output_batch_stride + q_head * head_dim;
        if (dim0 < head_dim) {
            output[output_head_base + dim0] = acc0[qh_local] * inv_sum;
        }
        if (dim1 < head_dim) {
            output[output_head_base + dim1] = acc1[qh_local] * inv_sum;
        }
    }
}

template <typename TilingConfig>
__global__ void ZipservDecodeAttentionOnlineGroupedQKernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_sign_mantissa,
    const __nv_bfloat16* __restrict__ k_compressed_full,
    const uint64_t* __restrict__ k_bitmap1,
    const uint64_t* __restrict__ k_bitmap2,
    const uint64_t* __restrict__ k_bitmap3,
    const int* __restrict__ k_tile_offsets_median,
    const int* __restrict__ k_tile_offsets_global,
    int k_rows,
    int k_cols,
    int k_batch_stride_rows,
    int k_head_stride_rows,
    int k_max_high_freq_count,
    int k_max_full_count,
    uint8_t k_start_exp,
    const uint8_t* __restrict__ v_sign_mantissa,
    const __nv_bfloat16* __restrict__ v_compressed_full,
    const uint64_t* __restrict__ v_bitmap1,
    const uint64_t* __restrict__ v_bitmap2,
    const uint64_t* __restrict__ v_bitmap3,
    const int* __restrict__ v_tile_offsets_median,
    const int* __restrict__ v_tile_offsets_global,
    int v_rows,
    int v_cols,
    int v_batch_stride_rows,
    int v_head_stride_rows,
    int v_max_high_freq_count,
    int v_max_full_count,
    uint8_t v_start_exp,
    int logical_kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    float* __restrict__ output) {
    constexpr int kGroupSize = 8;
    constexpr int kTileTokens = 64;
    const int kv_head = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int group_size = num_q_heads / num_kv_heads;
    const int token_group = threadIdx.x / 16;
    const int lane_in_group = threadIdx.x % 16;
    const int dim = threadIdx.x;
    if (dim >= head_dim) {
        return;
    }

    extern __shared__ __align__(128) __nv_bfloat16 smem_buffer[];
    auto* smem_output = reinterpret_cast<__nv_bfloat16(*)[64 + PADDING_SHARED_MEM_FOR_DECOMP]>(smem_buffer);
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    uint64_t* smem_bitmap1 = reinterpret_cast<uint64_t*>(smem_output + 64);
    uint64_t* smem_bitmap2 = smem_bitmap1 + bitmap_size;
    uint64_t* smem_bitmap3 = smem_bitmap2 + bitmap_size;
    uint8_t* smem_sign_mantissa = reinterpret_cast<uint8_t*>(smem_bitmap3 + bitmap_size);
    const size_t shared_hf_count = static_cast<size_t>(
        k_max_high_freq_count > v_max_high_freq_count ? k_max_high_freq_count : v_max_high_freq_count);
    const size_t shared_full_count = static_cast<size_t>(
        k_max_full_count > v_max_full_count ? k_max_full_count : v_max_full_count);
    const size_t padding = (128 - (shared_hf_count % 128)) % 128;
    __nv_bfloat16* smem_full_values =
        reinterpret_cast<__nv_bfloat16*>(smem_sign_mantissa + shared_hf_count + padding);
    uintptr_t smem_q_group_ptr = reinterpret_cast<uintptr_t>(smem_full_values + shared_full_count);
    smem_q_group_ptr = (smem_q_group_ptr + alignof(float) - 1) & ~(static_cast<uintptr_t>(alignof(float)) - 1);
    float* smem_q_group = reinterpret_cast<float*>(smem_q_group_ptr);
    __shared__ float tile_scores[kGroupSize * kTileTokens];
    __shared__ float tile_weights[kGroupSize * kTileTokens];
    __shared__ float running_max[kGroupSize];
    __shared__ float running_sum[kGroupSize];
    __shared__ float output_scale[kGroupSize];

    float acc[kGroupSize];
    #pragma unroll
    for (int q_local = 0; q_local < kGroupSize; ++q_local) {
        acc[q_local] = 0.0f;
    }

    if (threadIdx.x < kGroupSize) {
        running_max[threadIdx.x] = -FLT_MAX;
        running_sum[threadIdx.x] = 0.0f;
        output_scale[threadIdx.x] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < kGroupSize * kTileTokens; idx += blockDim.x) {
        tile_scores[idx] = 0.0f;
        tile_weights[idx] = 0.0f;
    }
    __syncthreads();

    const int q_batch_stride = num_q_heads * head_dim;
    const int output_batch_stride = num_q_heads * head_dim;
    const int q_head_base = kv_head * group_size;
    const int k_row_tile_base = batch_idx * (k_batch_stride_rows / kTileM) + kv_head * (k_head_stride_rows / kTileM);
    const int v_row_tile_base = batch_idx * (v_batch_stride_rows / kTileM) + kv_head * (v_head_stride_rows / kTileM);
    const int k_col_tiles = k_cols / kTileK;
    const int v_col_tiles = v_cols / kTileK;
    const int row_tiles = (logical_kv_len + kTileTokens - 1) / kTileTokens;

    for (int idx = threadIdx.x; idx < group_size * head_dim; idx += blockDim.x) {
        const int q_local = idx / head_dim;
        const int d = idx % head_dim;
        const int q_index =
            batch_idx * q_batch_stride + (q_head_base + q_local) * head_dim + d;
        smem_q_group[idx] = __bfloat162float(q[q_index]);
    }
    __syncthreads();

    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (int idx = threadIdx.x; idx < kGroupSize * kTileTokens; idx += blockDim.x) {
            tile_scores[idx] = 0.0f;
            tile_weights[idx] = 0.0f;
        }
        __syncthreads();

        for (int col_tile = 0; col_tile < k_col_tiles; ++col_tile) {
            float qv0[kGroupSize];
            float qv1[kGroupSize];
            float qv2[kGroupSize];
            float qv3[kGroupSize];
            #pragma unroll
            for (int q_local = 0; q_local < kGroupSize; ++q_local) {
                const int q_offset = q_local * head_dim + col_tile * kTileK;
                qv0[q_local] = smem_q_group[q_offset + lane_in_group];
                qv1[q_local] = smem_q_group[q_offset + lane_in_group + 16];
                qv2[q_local] = smem_q_group[q_offset + lane_in_group + 32];
                qv3[q_local] = smem_q_group[q_offset + lane_in_group + 48];
            }

            LoadZipservGlobalTileToShared<TilingConfig>(
                k_sign_mantissa,
                k_compressed_full,
                k_bitmap1,
                k_bitmap2,
                k_bitmap3,
                k_tile_offsets_median,
                k_tile_offsets_global,
                k_max_high_freq_count,
                k_max_full_count,
                k_start_exp,
                k_rows,
                k_cols,
                k_row_tile_base + row_tile,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            #pragma unroll
            for (int token_chunk = 0; token_chunk < kTileTokens; token_chunk += 8) {
                const int token_in_tile = token_chunk + token_group;
                const int token = row_tile * kTileTokens + token_in_tile;
                float k0 = 0.0f;
                float k1 = 0.0f;
                float k2 = 0.0f;
                float k3 = 0.0f;
                if (token < logical_kv_len) {
                    k0 = __bfloat162float(smem_output[token_in_tile][lane_in_group]);
                    k1 = __bfloat162float(smem_output[token_in_tile][lane_in_group + 16]);
                    k2 = __bfloat162float(smem_output[token_in_tile][lane_in_group + 32]);
                    k3 = __bfloat162float(smem_output[token_in_tile][lane_in_group + 48]);
                }
                #pragma unroll
                for (int q_local = 0; q_local < kGroupSize; ++q_local) {
                    float partial = 0.0f;
                    if (token < logical_kv_len) {
                        partial = fmaf(qv0[q_local], k0, partial);
                        partial = fmaf(qv1[q_local], k1, partial);
                        partial = fmaf(qv2[q_local], k2, partial);
                        partial = fmaf(qv3[q_local], k3, partial);
                    }
                    for (int offset = 8; offset > 0; offset /= 2) {
                        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset, 16);
                    }
                    if (lane_in_group == 0 && token < logical_kv_len) {
                        tile_scores[q_local * kTileTokens + token_in_tile] += partial;
                    }
                }
            }
            __syncthreads();
        }

        {
            const int q_local = token_group;
            const int softmax_lane = lane_in_group;
            const int tile_token_start = row_tile * kTileTokens;
            float tile_max = -FLT_MAX;
            #pragma unroll
            for (int token_in_tile = softmax_lane; token_in_tile < kTileTokens; token_in_tile += 16) {
                const int token = tile_token_start + token_in_tile;
                const int idx = q_local * kTileTokens + token_in_tile;
                if (token >= logical_kv_len) {
                    tile_scores[idx] = -FLT_MAX;
                    tile_weights[idx] = 0.0f;
                    continue;
                }
                const float score = tile_scores[idx] * sm_scale;
                tile_scores[idx] = score;
                tile_max = fmaxf(tile_max, score);
            }
            for (int offset = 8; offset > 0; offset /= 2) {
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, offset, 16));
            }

            const float prev_max = running_max[q_local];
            const float prev_sum = running_sum[q_local];
            float next_max = tile_max;
            if (softmax_lane == 0) {
                next_max = fmaxf(prev_max, tile_max);
                const float scale = expf(prev_max - next_max);
                output_scale[q_local] = scale;
                running_sum[q_local] = prev_sum * scale;
            }
            next_max = __shfl_sync(0xFFFFFFFF, next_max, 0, 16);

            float partial_sum = 0.0f;
            #pragma unroll
            for (int token_in_tile = softmax_lane; token_in_tile < kTileTokens; token_in_tile += 16) {
                const int token = tile_token_start + token_in_tile;
                const int idx = q_local * kTileTokens + token_in_tile;
                if (token >= logical_kv_len) {
                    tile_weights[idx] = 0.0f;
                    continue;
                }
                const float weight = expf(tile_scores[idx] - next_max);
                tile_weights[idx] = weight;
                partial_sum += weight;
            }
            for (int offset = 8; offset > 0; offset /= 2) {
                partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset, 16);
            }
            if (softmax_lane == 0) {
                running_sum[q_local] += partial_sum;
                running_max[q_local] = next_max;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int q_local = 0; q_local < kGroupSize; ++q_local) {
            acc[q_local] *= output_scale[q_local];
        }

        for (int col_tile = 0; col_tile < v_col_tiles; ++col_tile) {
            LoadZipservGlobalTileToShared<TilingConfig>(
                v_sign_mantissa,
                v_compressed_full,
                v_bitmap1,
                v_bitmap2,
                v_bitmap3,
                v_tile_offsets_median,
                v_tile_offsets_global,
                v_max_high_freq_count,
                v_max_full_count,
                v_start_exp,
                v_rows,
                v_cols,
                v_row_tile_base + row_tile,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            if ((dim / kTileK) == col_tile) {
                const int local_col = dim % kTileK;
                #pragma unroll
                for (int token_in_tile = 0; token_in_tile < kTileTokens; ++token_in_tile) {
                    const int token = row_tile * kTileTokens + token_in_tile;
                    if (token >= logical_kv_len) break;
                    const float value = __bfloat162float(smem_output[token_in_tile][local_col]);
                    #pragma unroll
                    for (int q_local = 0; q_local < kGroupSize; ++q_local) {
                        acc[q_local] = fmaf(tile_weights[q_local * kTileTokens + token_in_tile], value, acc[q_local]);
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int q_local = 0; q_local < kGroupSize; ++q_local) {
        const int out_base = batch_idx * output_batch_stride + (q_head_base + q_local) * head_dim;
        output[out_base + dim] = acc[q_local] / running_sum[q_local];
    }
}

template <typename TilingConfig>
__global__ void ZipservDecodeAttentionOnlineGroupedPairKernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_sign_mantissa,
    const __nv_bfloat16* __restrict__ k_compressed_full,
    const uint64_t* __restrict__ k_bitmap1,
    const uint64_t* __restrict__ k_bitmap2,
    const uint64_t* __restrict__ k_bitmap3,
    const int* __restrict__ k_tile_offsets_median,
    const int* __restrict__ k_tile_offsets_global,
    int k_rows,
    int k_cols,
    int k_batch_stride_rows,
    int k_head_stride_rows,
    int k_max_high_freq_count,
    int k_max_full_count,
    uint8_t k_start_exp,
    const uint8_t* __restrict__ v_sign_mantissa,
    const __nv_bfloat16* __restrict__ v_compressed_full,
    const uint64_t* __restrict__ v_bitmap1,
    const uint64_t* __restrict__ v_bitmap2,
    const uint64_t* __restrict__ v_bitmap3,
    const int* __restrict__ v_tile_offsets_median,
    const int* __restrict__ v_tile_offsets_global,
    int v_rows,
    int v_cols,
    int v_batch_stride_rows,
    int v_head_stride_rows,
    int v_max_high_freq_count,
    int v_max_full_count,
    uint8_t v_start_exp,
    int logical_kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    float* __restrict__ output) {
    const int kv_head = blockIdx.x;
    const int q_head_group = blockIdx.y * kGroupedQHeadsPerCta;
    const int batch_idx = blockIdx.z;
    const int kv_group_size = num_q_heads / num_kv_heads;
    const int q_head_base = kv_head * kv_group_size + q_head_group;
    const int remaining_q_heads = kv_group_size - q_head_group;
    const int active_q_heads =
        remaining_q_heads <= 0 ? 0 : (remaining_q_heads < kGroupedQHeadsPerCta ? remaining_q_heads : kGroupedQHeadsPerCta);
    const int tid = threadIdx.x;
    const int q_head_slot = tid / kThreadsPerGroupedQHead;
    const int local_tid = tid % kThreadsPerGroupedQHead;
    const int token_group = local_tid / 16;
    const int lane_in_group = local_tid % 16;
    const bool q_head_active = q_head_slot < active_q_heads;
    const int q_head = q_head_base + q_head_slot;

    extern __shared__ __align__(128) __nv_bfloat16 smem_buffer[];
    auto* smem_output = reinterpret_cast<__nv_bfloat16(*)[64 + PADDING_SHARED_MEM_FOR_DECOMP]>(smem_buffer);
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    uint64_t* smem_bitmap1 = reinterpret_cast<uint64_t*>(smem_output + kTileM);
    uint64_t* smem_bitmap2 = smem_bitmap1 + bitmap_size;
    uint64_t* smem_bitmap3 = smem_bitmap2 + bitmap_size;
    uint8_t* smem_sign_mantissa = reinterpret_cast<uint8_t*>(smem_bitmap3 + bitmap_size);
    const size_t shared_hf_count = static_cast<size_t>(
        k_max_high_freq_count > v_max_high_freq_count ? k_max_high_freq_count : v_max_high_freq_count);
    const size_t padding = (128 - (shared_hf_count % 128)) % 128;
    __nv_bfloat16* smem_full_values =
        reinterpret_cast<__nv_bfloat16*>(smem_sign_mantissa + shared_hf_count + padding);

    __shared__ float tile_scores[kGroupedQHeadsPerCta][kTileM];
    __shared__ float tile_weights[kGroupedQHeadsPerCta][kTileM];
    __shared__ float running_max[kGroupedQHeadsPerCta];
    __shared__ float running_sum[kGroupedQHeadsPerCta];
    __shared__ float output_scale[kGroupedQHeadsPerCta];
    __shared__ float inv_running_sum[kGroupedQHeadsPerCta];
    __shared__ float softmax_warp_reduce[kGroupedQHeadsPerCta][2];

    if (tid < kGroupedQHeadsPerCta) {
        running_max[tid] = -INFINITY;
        running_sum[tid] = 0.0f;
        output_scale[tid] = 0.0f;
        inv_running_sum[tid] = 0.0f;
    }
    __syncthreads();

    float acc[kMaxOutputAccumsPerThread] = {0.0f};
    const int q_batch_stride = num_q_heads * head_dim;
    const int output_batch_stride = num_q_heads * head_dim;
    const int q_head_base_offset = batch_idx * q_batch_stride;
    const int output_batch_base = batch_idx * output_batch_stride;
    const int k_col_tiles = k_cols / kTileK;
    const int v_col_tiles = v_cols / kTileK;
    const int row_tiles = (logical_kv_len + kTileM - 1) / kTileM;
    const int batch_stride_tiles = k_batch_stride_rows / kTileM;
    const int head_stride_tiles = k_head_stride_rows / kTileM;
    const int v_batch_stride_tiles = v_batch_stride_rows / kTileM;
    const int v_head_stride_tiles = v_head_stride_rows / kTileM;
    const int k_row_tile_base = batch_idx * batch_stride_tiles + kv_head * head_stride_tiles;
    const int v_row_tile_base = batch_idx * v_batch_stride_tiles + kv_head * v_head_stride_tiles;
    const __nv_bfloat16* q_ptr = q_head_active ? q + q_head_base_offset + q_head * head_dim : nullptr;

    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        if (tid < active_q_heads * kTileM) {
            const int init_head = tid / kTileM;
            const int init_token = tid % kTileM;
            tile_scores[init_head][init_token] = 0.0f;
            tile_weights[init_head][init_token] = 0.0f;
        }
        __syncthreads();

        const int row_token_base = row_tile * kTileM;

        for (int col_tile = 0; col_tile < k_col_tiles; ++col_tile) {
            LoadZipservGlobalTileToShared<TilingConfig>(
                k_sign_mantissa,
                k_compressed_full,
                k_bitmap1,
                k_bitmap2,
                k_bitmap3,
                k_tile_offsets_median,
                k_tile_offsets_global,
                k_max_high_freq_count,
                k_max_full_count,
                k_start_exp,
                k_rows,
                k_cols,
                k_row_tile_base + row_tile,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            if (q_head_active) {
                const int q_base = col_tile * kTileK + lane_in_group;
                const float q0 = __bfloat162float(q_ptr[q_base + 0]);
                const float q1 = __bfloat162float(q_ptr[q_base + 16]);
                const float q2 = __bfloat162float(q_ptr[q_base + 32]);
                const float q3 = __bfloat162float(q_ptr[q_base + 48]);
                #pragma unroll
                for (int token_block = 0; token_block < (kTileM / kTokenGroupsPerGroupedQHead); ++token_block) {
                    const int token_in_tile = token_block * kTokenGroupsPerGroupedQHead + token_group;
                    const int token = row_token_base + token_in_tile;
                    float dot = 0.0f;
                    if (token < logical_kv_len) {
                        dot = fmaf(q0, __bfloat162float(smem_output[token_in_tile][lane_in_group + 0]), dot);
                        dot = fmaf(q1, __bfloat162float(smem_output[token_in_tile][lane_in_group + 16]), dot);
                        dot = fmaf(q2, __bfloat162float(smem_output[token_in_tile][lane_in_group + 32]), dot);
                        dot = fmaf(q3, __bfloat162float(smem_output[token_in_tile][lane_in_group + 48]), dot);
                    }
                    for (int offset = 8; offset > 0; offset >>= 1) {
                        dot += __shfl_down_sync(0xFFFFFFFFu, dot, offset, 16);
                    }
                    if (lane_in_group == 0 && token < logical_kv_len) {
                        tile_scores[q_head_slot][token_in_tile] += dot;
                    }
                }
            }
            __syncthreads();
        }

        if (q_head_active) {
            const int token_in_tile = local_tid;
            const int token = row_token_base + token_in_tile;
            const int warp_id = local_tid >> 5;
            const int lane = local_tid & 31;

            float score = -INFINITY;
            if (token < logical_kv_len) {
                score = tile_scores[q_head_slot][token_in_tile] * sm_scale;
            }
            tile_scores[q_head_slot][token_in_tile] = score;

            float warp_max = score;
            for (int offset = 16; offset > 0; offset >>= 1) {
                warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFFu, warp_max, offset));
            }
            if (lane == 0) {
                softmax_warp_reduce[q_head_slot][warp_id] = warp_max;
            }
        }
        __syncthreads();

        if (local_tid == 0 && q_head_active) {
            const float tile_max = fmaxf(softmax_warp_reduce[q_head_slot][0], softmax_warp_reduce[q_head_slot][1]);
            const float next_max =
                running_sum[q_head_slot] > 0.0f ? fmaxf(running_max[q_head_slot], tile_max) : tile_max;
            const float scale_prev =
                running_sum[q_head_slot] > 0.0f ? __expf(running_max[q_head_slot] - next_max) : 0.0f;
            output_scale[q_head_slot] = scale_prev;
            running_max[q_head_slot] = next_max;
            running_sum[q_head_slot] *= scale_prev;
        }
        __syncthreads();

        if (q_head_active) {
            const int token_in_tile = local_tid;
            const int token = row_token_base + token_in_tile;
            const int warp_id = local_tid >> 5;
            const int lane = local_tid & 31;

            float weight = 0.0f;
            if (token < logical_kv_len) {
                weight = __expf(tile_scores[q_head_slot][token_in_tile] - running_max[q_head_slot]);
            }
            tile_weights[q_head_slot][token_in_tile] = weight;

            float warp_sum = weight;
            for (int offset = 16; offset > 0; offset >>= 1) {
                warp_sum += __shfl_down_sync(0xFFFFFFFFu, warp_sum, offset);
            }
            if (lane == 0) {
                softmax_warp_reduce[q_head_slot][warp_id] = warp_sum;
            }
        }
        __syncthreads();

        if (local_tid == 0 && q_head_active) {
            running_sum[q_head_slot] += softmax_warp_reduce[q_head_slot][0] + softmax_warp_reduce[q_head_slot][1];
        }
        __syncthreads();

        if (q_head_active) {
            #pragma unroll
            for (int acc_idx = 0; acc_idx < kMaxOutputAccumsPerThread; ++acc_idx) {
                acc[acc_idx] *= output_scale[q_head_slot];
            }
        }

        for (int col_tile = 0; col_tile < v_col_tiles; ++col_tile) {
            LoadZipservGlobalTileToShared<TilingConfig>(
                v_sign_mantissa,
                v_compressed_full,
                v_bitmap1,
                v_bitmap2,
                v_bitmap3,
                v_tile_offsets_median,
                v_tile_offsets_global,
                v_max_high_freq_count,
                v_max_full_count,
                v_start_exp,
                v_rows,
                v_cols,
                v_row_tile_base + row_tile,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            if (q_head_active) {
                const int col_start = col_tile * kTileK;
                #pragma unroll
                for (int acc_idx = 0; acc_idx < kMaxOutputAccumsPerThread; ++acc_idx) {
                    const int dim = local_tid + acc_idx * kThreadsPerGroupedQHead;
                    if (dim >= col_start && dim < col_start + kTileK && dim < head_dim) {
                        const int local_col = dim - col_start;
                        #pragma unroll
                        for (int token_in_tile = 0; token_in_tile < kTileM; ++token_in_tile) {
                            const int token = row_token_base + token_in_tile;
                            if (token >= logical_kv_len) {
                                break;
                            }
                            acc[acc_idx] = fmaf(
                                tile_weights[q_head_slot][token_in_tile],
                                __bfloat162float(smem_output[token_in_tile][local_col]),
                                acc[acc_idx]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    if (local_tid == 0 && q_head_active) {
        inv_running_sum[q_head_slot] = 1.0f / running_sum[q_head_slot];
    }
    __syncthreads();

    if (q_head_active) {
        const int output_head_base = output_batch_base + q_head * head_dim;
        #pragma unroll
        for (int acc_idx = 0; acc_idx < kMaxOutputAccumsPerThread; ++acc_idx) {
            const int dim = local_tid + acc_idx * kThreadsPerGroupedQHead;
            if (dim < head_dim) {
                output[output_head_base + dim] = acc[acc_idx] * inv_running_sum[q_head_slot];
            }
        }
    }
}

__global__ void RowSoftmaxKernel(
    const float* __restrict__ scores,
    float* __restrict__ probs,
    int rows,
    int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float reduce[kSoftmaxThreads];

    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        thread_max = fmaxf(thread_max, scores[row * cols + col]);
    }
    reduce[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = reduce[0];

    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float val = expf(scores[row * cols + col] - row_max);
        probs[row * cols + col] = val;
        thread_sum += val;
    }
    reduce[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduce[0];
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        probs[row * cols + col] *= inv_sum;
    }
}

}  // namespace

std::vector<torch::Tensor> compress_zipserv_cuda(
    torch::Tensor input,
    int64_t logical_rows,
    int64_t logical_cols) {
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    TORCH_CHECK(rows % 64 == 0, "rows must be padded to a multiple of 64");
    TORCH_CHECK(cols % 64 == 0, "cols must be padded to a multiple of 64");

    auto device = input.device();
    auto stream = at::cuda::getDefaultCUDAStream(device.index()).stream();
    const int num_tiles = (rows / 8) * (cols / 8);
    const int num_median_tiles = (rows / 16) * (cols / 64);
    const int num_global_tiles = (rows / 64) * (cols / 64);
    const int64_t max_elems = static_cast<int64_t>(rows) * cols;
    const int max_elem_per_gtile = 64 * 64;
    const int max_sm_per_gtile = max_elem_per_gtile + 15;
    const int max_full_per_gtile = max_elem_per_gtile + 7;

    auto sign_mantissa = torch::empty({max_elems}, input.options().dtype(torch::kUInt8));
    auto compressed_full = torch::empty({max_elems}, input.options().dtype(torch::kBFloat16));
    auto bitmap1 = torch::empty({num_tiles}, input.options().dtype(torch::kUInt64));
    auto bitmap2 = torch::empty({num_tiles}, input.options().dtype(torch::kUInt64));
    auto bitmap3 = torch::empty({num_tiles}, input.options().dtype(torch::kUInt64));
    auto tile_offsets = torch::empty({num_tiles * 2}, input.options().dtype(torch::kInt32));
    auto tile_offsets_median = torch::empty({num_median_tiles * 2}, input.options().dtype(torch::kInt32));
    auto tile_offsets_global = torch::empty({(num_global_tiles + 1) * 2}, input.options().dtype(torch::kInt32));
    auto temp_sm = torch::empty({static_cast<int64_t>(num_global_tiles) * max_sm_per_gtile},
                                input.options().dtype(torch::kUInt8));
    auto temp_full = torch::empty({static_cast<int64_t>(num_global_tiles) * max_full_per_gtile},
                                  input.options().dtype(torch::kBFloat16));
    auto gt_hf_count = torch::empty({num_global_tiles}, input.options().dtype(torch::kInt32));
    auto gt_full_count = torch::empty({num_global_tiles}, input.options().dtype(torch::kInt32));
    auto hf_offsets = torch::empty({num_global_tiles + 1}, input.options().dtype(torch::kInt32));
    auto full_offsets = torch::empty({num_global_tiles + 1}, input.options().dtype(torch::kInt32));
    auto top_exponents = torch::empty({kTopN}, input.options().dtype(torch::kInt32));
    auto exponent_counts = torch::empty({256}, input.options().dtype(torch::kInt32));
    auto phase_state = torch::empty({4}, input.options().dtype(torch::kInt32));
    auto max_hf = torch::empty({1}, input.options().dtype(torch::kInt32));
    auto max_full = torch::empty({1}, input.options().dtype(torch::kInt32));
    auto total_hf = torch::empty({1}, input.options().dtype(torch::kInt32));
    auto total_full = torch::empty({1}, input.options().dtype(torch::kInt32));

    CUDA_CHECK(AnalyzeTopExponentsGPU(
        stream,
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        static_cast<int>(max_elems),
        exponent_counts.data_ptr<int>(),
        top_exponents.data_ptr<int>(),
        phase_state.data_ptr<int>()));

    CUDA_CHECK(InitBF16MatrixTripleBitmap_GPU(
        stream,
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        rows,
        cols,
        top_exponents.data_ptr<int>(),
        sign_mantissa.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(compressed_full.data_ptr<at::BFloat16>()),
        bitmap1.data_ptr<uint64_t>(),
        bitmap2.data_ptr<uint64_t>(),
        bitmap3.data_ptr<uint64_t>(),
        tile_offsets.data_ptr<int>(),
        tile_offsets_median.data_ptr<int>(),
        tile_offsets_global.data_ptr<int>(),
        temp_sm.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(temp_full.data_ptr<at::BFloat16>()),
        gt_hf_count.data_ptr<int>(),
        gt_full_count.data_ptr<int>(),
        hf_offsets.data_ptr<int>(),
        full_offsets.data_ptr<int>(),
        max_hf.data_ptr<int>(),
        max_full.data_ptr<int>(),
        total_hf.data_ptr<int>(),
        total_full.data_ptr<int>()));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    int host_top[kTopN];
    int host_max_hf = 0;
    int host_max_full = 0;
    int host_total_hf = 0;
    int host_total_full = 0;
    CUDA_CHECK(cudaMemcpy(host_top, top_exponents.data_ptr<int>(), sizeof(host_top), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_max_hf, max_hf.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_max_full, max_full.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_total_hf, total_hf.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_total_full, total_full.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost));

    const int start_exp = host_top[0] - 1;
    const int64_t comp_bytes = static_cast<int64_t>(
        ComputeCompressedBytes(rows, cols, num_global_tiles, host_total_hf, host_total_full));
    auto meta = torch::tensor(
        {static_cast<int64_t>(rows),
         static_cast<int64_t>(cols),
         logical_rows,
         logical_cols,
         static_cast<int64_t>(host_max_hf),
         static_cast<int64_t>(host_max_full),
         static_cast<int64_t>(start_exp),
         static_cast<int64_t>(host_total_hf),
         static_cast<int64_t>(host_total_full),
         static_cast<int64_t>(num_global_tiles),
         comp_bytes},
        torch::TensorOptions().dtype(torch::kInt64));

    return {sign_mantissa, compressed_full, bitmap1, bitmap2, bitmap3, tile_offsets_median, tile_offsets_global, meta};
}

torch::Tensor decompress_zipserv_into_cuda(
    torch::Tensor output,
    torch::Tensor sign_mantissa,
    torch::Tensor compressed_full,
    torch::Tensor bitmap1,
    torch::Tensor bitmap2,
    torch::Tensor bitmap3,
    torch::Tensor tile_offsets_median,
    torch::Tensor tile_offsets_global,
    int64_t rows,
    int64_t cols,
    int64_t max_high_freq_count,
    int64_t max_full_count,
    int64_t start_exp);

torch::Tensor decompress_zipserv_cuda(
    torch::Tensor sign_mantissa,
    torch::Tensor compressed_full,
    torch::Tensor bitmap1,
    torch::Tensor bitmap2,
    torch::Tensor bitmap3,
    torch::Tensor tile_offsets_median,
    torch::Tensor tile_offsets_global,
    int64_t rows,
    int64_t cols,
    int64_t max_high_freq_count,
    int64_t max_full_count,
    int64_t start_exp) {
    auto output = torch::empty({rows, cols}, compressed_full.options().dtype(torch::kBFloat16));
    return decompress_zipserv_into_cuda(
        output,
        sign_mantissa,
        compressed_full,
        bitmap1,
        bitmap2,
        bitmap3,
        tile_offsets_median,
        tile_offsets_global,
        rows,
        cols,
        max_high_freq_count,
        max_full_count,
        start_exp);
}

torch::Tensor decompress_zipserv_into_cuda(
    torch::Tensor output,
    torch::Tensor sign_mantissa,
    torch::Tensor compressed_full,
    torch::Tensor bitmap1,
    torch::Tensor bitmap2,
    torch::Tensor bitmap3,
    torch::Tensor tile_offsets_median,
    torch::Tensor tile_offsets_global,
    int64_t rows,
    int64_t cols,
    int64_t max_high_freq_count,
    int64_t max_full_count,
    int64_t start_exp) {
    auto stream = at::cuda::getDefaultCUDAStream(sign_mantissa.device().index()).stream();
    CUDA_CHECK(BF16TripleBitmap_Decompress_API(
        stream,
        sign_mantissa.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(compressed_full.data_ptr<at::BFloat16>()),
        bitmap1.data_ptr<uint64_t>(),
        bitmap2.data_ptr<uint64_t>(),
        bitmap3.data_ptr<uint64_t>(),
        tile_offsets_median.data_ptr<int>(),
        tile_offsets_global.data_ptr<int>(),
        static_cast<int>(max_high_freq_count),
        static_cast<int>(max_full_count),
        static_cast<uint8_t>(start_exp),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        static_cast<int>(rows),
        static_cast<int>(cols)));
    return output;
}

torch::Tensor zipserv_decode_attention_cuda(
    torch::Tensor q,
    torch::Tensor k_sign_mantissa,
    torch::Tensor k_compressed_full,
    torch::Tensor k_bitmap1,
    torch::Tensor k_bitmap2,
    torch::Tensor k_bitmap3,
    torch::Tensor k_tile_offsets_median,
    torch::Tensor k_tile_offsets_global,
    int64_t k_rows,
    int64_t k_cols,
    int64_t k_max_high_freq_count,
    int64_t k_max_full_count,
    int64_t k_start_exp,
    torch::Tensor v_sign_mantissa,
    torch::Tensor v_compressed_full,
    torch::Tensor v_bitmap1,
    torch::Tensor v_bitmap2,
    torch::Tensor v_bitmap3,
    torch::Tensor v_tile_offsets_median,
    torch::Tensor v_tile_offsets_global,
    int64_t v_rows,
    int64_t v_cols,
    int64_t v_max_high_freq_count,
    int64_t v_max_full_count,
    int64_t v_start_exp,
    int64_t logical_kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    double sm_scale) {
    TORCH_CHECK(k_rows == v_rows, "K/V rows must match");
    TORCH_CHECK(k_cols == v_cols, "K/V cols must match");
    TORCH_CHECK(k_cols == head_dim, "compressed K cols must match head_dim");
    TORCH_CHECK(v_cols == head_dim, "compressed V cols must match head_dim");
    TORCH_CHECK(k_rows % kTileM == 0, "compressed K rows must be padded to a multiple of 64");
    TORCH_CHECK(k_cols % kTileK == 0, "compressed K cols must be padded to a multiple of 64");
    TORCH_CHECK(v_rows % kTileM == 0, "compressed V rows must be padded to a multiple of 64");
    TORCH_CHECK(v_cols % kTileK == 0, "compressed V cols must be padded to a multiple of 64");
    TORCH_CHECK(num_q_heads > 0, "num_q_heads must be > 0");
    TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be > 0");
    TORCH_CHECK(head_dim > 0, "head_dim must be > 0");
    TORCH_CHECK(head_dim <= 256, "zipserv native decode attention currently requires head_dim <= 256");
    TORCH_CHECK(logical_kv_len > 0, "logical_kv_len must be > 0");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(num_kv_heads == 8, "zipserv native decode attention currently requires num_kv_heads == 8");
    TORCH_CHECK(logical_kv_len <= (k_rows / num_kv_heads), "logical_kv_len exceeds compressed row capacity");

    using Config = TilingConfigBF16TripleBitmap<4, 1, 4>;
    auto stream = at::cuda::getDefaultCUDAStream(q.device().index()).stream();
    auto scores = torch::empty(
        {num_q_heads, logical_kv_len},
        q.options().dtype(torch::kFloat32));
    auto probs = torch::empty_like(scores);
    auto output = torch::empty(
        {num_q_heads, head_dim},
        q.options().dtype(torch::kFloat32));

    const int tokens_per_tile = kTileM / num_kv_heads;
    const dim3 score_grid(
        static_cast<unsigned int>(num_q_heads),
        static_cast<unsigned int>((logical_kv_len + tokens_per_tile - 1) / tokens_per_tile));
    const size_t score_shared_bytes = ComputeSharedBytes(
        static_cast<int>(k_max_high_freq_count),
        static_cast<int>(k_max_full_count));
    ZipservDecodeScoresKernel<Config><<<score_grid, kThreadsPerBlock, score_shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        k_sign_mantissa.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(k_compressed_full.data_ptr<at::BFloat16>()),
        k_bitmap1.data_ptr<uint64_t>(),
        k_bitmap2.data_ptr<uint64_t>(),
        k_bitmap3.data_ptr<uint64_t>(),
        k_tile_offsets_median.data_ptr<int>(),
        k_tile_offsets_global.data_ptr<int>(),
        static_cast<int>(k_rows),
        static_cast<int>(k_cols),
        static_cast<int>(k_max_high_freq_count),
        static_cast<int>(k_max_full_count),
        static_cast<uint8_t>(k_start_exp),
        static_cast<int>(logical_kv_len),
        static_cast<int>(num_q_heads),
        static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim),
        static_cast<float>(sm_scale),
        scores.data_ptr<float>());
    CUDA_CHECK(cudaGetLastError());

    RowSoftmaxKernel<<<static_cast<unsigned int>(num_q_heads), kSoftmaxThreads, 0, stream>>>(
        scores.data_ptr<float>(),
        probs.data_ptr<float>(),
        static_cast<int>(num_q_heads),
        static_cast<int>(logical_kv_len));
    CUDA_CHECK(cudaGetLastError());

    const int value_threads = std::min(256, RoundUpToWarp(static_cast<int>(head_dim)));
    const size_t value_shared_bytes = ComputeSharedBytes(
        static_cast<int>(v_max_high_freq_count),
        static_cast<int>(v_max_full_count));
    ZipservDecodeValuesKernel<Config><<<static_cast<unsigned int>(num_q_heads), value_threads, value_shared_bytes, stream>>>(
        probs.data_ptr<float>(),
        v_sign_mantissa.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(v_compressed_full.data_ptr<at::BFloat16>()),
        v_bitmap1.data_ptr<uint64_t>(),
        v_bitmap2.data_ptr<uint64_t>(),
        v_bitmap3.data_ptr<uint64_t>(),
        v_tile_offsets_median.data_ptr<int>(),
        v_tile_offsets_global.data_ptr<int>(),
        static_cast<int>(v_rows),
        static_cast<int>(v_cols),
        static_cast<int>(v_max_high_freq_count),
        static_cast<int>(v_max_full_count),
        static_cast<uint8_t>(v_start_exp),
        static_cast<int>(logical_kv_len),
        static_cast<int>(num_q_heads),
        static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim),
        output.data_ptr<float>());
    CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor zipserv_decode_attention_online_cuda(
    torch::Tensor q,
    torch::Tensor k_sign_mantissa,
    torch::Tensor k_compressed_full,
    torch::Tensor k_bitmap1,
    torch::Tensor k_bitmap2,
    torch::Tensor k_bitmap3,
    torch::Tensor k_tile_offsets_median,
    torch::Tensor k_tile_offsets_global,
    int64_t k_rows,
    int64_t k_cols,
    int64_t k_batch_stride_rows,
    int64_t k_head_stride_rows,
    int64_t k_max_high_freq_count,
    int64_t k_max_full_count,
    int64_t k_start_exp,
    torch::Tensor v_sign_mantissa,
    torch::Tensor v_compressed_full,
    torch::Tensor v_bitmap1,
    torch::Tensor v_bitmap2,
    torch::Tensor v_bitmap3,
    torch::Tensor v_tile_offsets_median,
    torch::Tensor v_tile_offsets_global,
    int64_t v_rows,
    int64_t v_cols,
    int64_t v_batch_stride_rows,
    int64_t v_head_stride_rows,
    int64_t v_max_high_freq_count,
    int64_t v_max_full_count,
    int64_t v_start_exp,
    int64_t logical_kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    double sm_scale) {
    TORCH_CHECK(k_rows == v_rows, "K/V rows must match");
    TORCH_CHECK(k_cols == v_cols, "K/V cols must match");
    TORCH_CHECK(k_cols == head_dim, "compressed K cols must match head_dim");
    TORCH_CHECK(v_cols == head_dim, "compressed V cols must match head_dim");
    TORCH_CHECK(k_rows % kTileM == 0, "compressed K rows must be padded to a multiple of 64");
    TORCH_CHECK(k_cols % kTileK == 0, "compressed K cols must be padded to a multiple of 64");
    TORCH_CHECK(v_rows % kTileM == 0, "compressed V rows must be padded to a multiple of 64");
    TORCH_CHECK(v_cols % kTileK == 0, "compressed V cols must be padded to a multiple of 64");
    TORCH_CHECK(num_q_heads > 0, "num_q_heads must be > 0");
    TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be > 0");
    TORCH_CHECK(head_dim > 0, "head_dim must be > 0");
    TORCH_CHECK(head_dim <= 256, "zipserv fused native decode attention currently requires head_dim <= 256");
    TORCH_CHECK(logical_kv_len > 0, "logical_kv_len must be > 0");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(num_kv_heads == 8, "zipserv fused native decode attention currently requires num_kv_heads == 8");
    TORCH_CHECK(k_batch_stride_rows > 0 && v_batch_stride_rows > 0, "batch strides must be > 0");
    TORCH_CHECK(k_head_stride_rows > 0 && v_head_stride_rows > 0, "head strides must be > 0");
    TORCH_CHECK(k_rows % k_batch_stride_rows == 0, "K rows must be divisible by k_batch_stride_rows");
    TORCH_CHECK(v_rows % v_batch_stride_rows == 0, "V rows must be divisible by v_batch_stride_rows");
    TORCH_CHECK(k_head_stride_rows * num_kv_heads >= logical_kv_len * num_kv_heads,
                "logical_kv_len exceeds K row capacity for the provided stride metadata");
    TORCH_CHECK(v_head_stride_rows * num_kv_heads >= logical_kv_len * num_kv_heads,
                "logical_kv_len exceeds V row capacity for the provided stride metadata");
    TORCH_CHECK(logical_kv_len <= (k_rows / num_kv_heads), "logical_kv_len exceeds compressed row capacity");
    TORCH_CHECK(k_batch_stride_rows % kTileM == 0, "K batch stride rows must be a multiple of 64");
    TORCH_CHECK(v_batch_stride_rows % kTileM == 0, "V batch stride rows must be a multiple of 64");
    TORCH_CHECK(k_head_stride_rows % kTileM == 0, "K head stride rows must be a multiple of 64");
    TORCH_CHECK(v_head_stride_rows % kTileM == 0, "V head stride rows must be a multiple of 64");
    using Config = TilingConfigBF16TripleBitmap<4, 1, 4>;
    auto stream = at::cuda::getDefaultCUDAStream(q.device().index()).stream();
    const int shared_hf = std::max(static_cast<int>(k_max_high_freq_count), static_cast<int>(v_max_high_freq_count));
    const int shared_full = std::max(static_cast<int>(k_max_full_count), static_cast<int>(v_max_full_count));
    const size_t shared_bytes = ComputeSharedBytes(shared_hf, shared_full);
    if (q.dim() == 2) {
        auto output = torch::empty(
            {num_q_heads, head_dim},
            q.options().dtype(torch::kFloat32));
        ZipservDecodeAttentionOnlineKernel<Config>
            <<<static_cast<unsigned int>(num_q_heads), kThreadsPerBlock, shared_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                k_sign_mantissa.data_ptr<uint8_t>(),
                reinterpret_cast<const __nv_bfloat16*>(k_compressed_full.data_ptr<at::BFloat16>()),
                k_bitmap1.data_ptr<uint64_t>(),
                k_bitmap2.data_ptr<uint64_t>(),
                k_bitmap3.data_ptr<uint64_t>(),
                k_tile_offsets_median.data_ptr<int>(),
                k_tile_offsets_global.data_ptr<int>(),
                static_cast<int>(k_rows),
                static_cast<int>(k_cols),
                static_cast<int>(k_max_high_freq_count),
                static_cast<int>(k_max_full_count),
                static_cast<uint8_t>(k_start_exp),
                v_sign_mantissa.data_ptr<uint8_t>(),
                reinterpret_cast<const __nv_bfloat16*>(v_compressed_full.data_ptr<at::BFloat16>()),
                v_bitmap1.data_ptr<uint64_t>(),
                v_bitmap2.data_ptr<uint64_t>(),
                v_bitmap3.data_ptr<uint64_t>(),
                v_tile_offsets_median.data_ptr<int>(),
                v_tile_offsets_global.data_ptr<int>(),
                static_cast<int>(v_rows),
                static_cast<int>(v_cols),
                static_cast<int>(v_max_high_freq_count),
                static_cast<int>(v_max_full_count),
                static_cast<uint8_t>(v_start_exp),
                static_cast<int>(logical_kv_len),
                static_cast<int>(num_q_heads),
                static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim),
                static_cast<float>(sm_scale),
                output.data_ptr<float>());
        CUDA_CHECK(cudaGetLastError());
        return output;
    }

    const int batch_size = static_cast<int>(q.size(0));
    TORCH_CHECK(k_rows == batch_size * k_batch_stride_rows, "K rows must match batch_size * k_batch_stride_rows");
    TORCH_CHECK(v_rows == batch_size * v_batch_stride_rows, "V rows must match batch_size * v_batch_stride_rows");
    auto output = torch::empty(
        {batch_size, num_q_heads, head_dim},
        q.options().dtype(torch::kFloat32));
    if (batch_size == 1) {
        const dim3 grid(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(batch_size));
        ZipservDecodeAttentionOnlineBatchedKernel<Config>
            <<<grid, kThreadsPerBlock, shared_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                k_sign_mantissa.data_ptr<uint8_t>(),
                reinterpret_cast<const __nv_bfloat16*>(k_compressed_full.data_ptr<at::BFloat16>()),
                k_bitmap1.data_ptr<uint64_t>(),
                k_bitmap2.data_ptr<uint64_t>(),
                k_bitmap3.data_ptr<uint64_t>(),
                k_tile_offsets_median.data_ptr<int>(),
                k_tile_offsets_global.data_ptr<int>(),
                static_cast<int>(k_rows),
                static_cast<int>(k_cols),
                static_cast<int>(k_batch_stride_rows),
                static_cast<int>(k_head_stride_rows),
                static_cast<int>(k_max_high_freq_count),
                static_cast<int>(k_max_full_count),
                static_cast<uint8_t>(k_start_exp),
                v_sign_mantissa.data_ptr<uint8_t>(),
                reinterpret_cast<const __nv_bfloat16*>(v_compressed_full.data_ptr<at::BFloat16>()),
                v_bitmap1.data_ptr<uint64_t>(),
                v_bitmap2.data_ptr<uint64_t>(),
                v_bitmap3.data_ptr<uint64_t>(),
                v_tile_offsets_median.data_ptr<int>(),
                v_tile_offsets_global.data_ptr<int>(),
                static_cast<int>(v_rows),
                static_cast<int>(v_cols),
                static_cast<int>(v_batch_stride_rows),
                static_cast<int>(v_head_stride_rows),
                static_cast<int>(v_max_high_freq_count),
                static_cast<int>(v_max_full_count),
                static_cast<uint8_t>(v_start_exp),
                static_cast<int>(logical_kv_len),
                static_cast<int>(num_q_heads),
                static_cast<int>(num_kv_heads),
                static_cast<int>(head_dim),
                static_cast<float>(sm_scale),
                output.data_ptr<float>());
        CUDA_CHECK(cudaGetLastError());
        return output;
    }

    TORCH_CHECK(num_q_heads / num_kv_heads == 8,
                "Grouped batched online decode currently requires q_group_size == 8");
    const size_t grouped_shared_bytes =
        shared_bytes + (alignof(float) - 1) +
        static_cast<size_t>(num_q_heads / num_kv_heads) * static_cast<size_t>(head_dim) * sizeof(float);
    const dim3 grid(static_cast<unsigned int>(num_kv_heads), static_cast<unsigned int>(batch_size));
    ZipservDecodeAttentionOnlineGroupedQKernel<Config>
        <<<grid, kThreadsPerBlock, grouped_shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            k_sign_mantissa.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(k_compressed_full.data_ptr<at::BFloat16>()),
            k_bitmap1.data_ptr<uint64_t>(),
            k_bitmap2.data_ptr<uint64_t>(),
            k_bitmap3.data_ptr<uint64_t>(),
            k_tile_offsets_median.data_ptr<int>(),
            k_tile_offsets_global.data_ptr<int>(),
            static_cast<int>(k_rows),
            static_cast<int>(k_cols),
            static_cast<int>(k_batch_stride_rows),
            static_cast<int>(k_head_stride_rows),
            static_cast<int>(k_max_high_freq_count),
            static_cast<int>(k_max_full_count),
            static_cast<uint8_t>(k_start_exp),
            v_sign_mantissa.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(v_compressed_full.data_ptr<at::BFloat16>()),
            v_bitmap1.data_ptr<uint64_t>(),
            v_bitmap2.data_ptr<uint64_t>(),
            v_bitmap3.data_ptr<uint64_t>(),
            v_tile_offsets_median.data_ptr<int>(),
            v_tile_offsets_global.data_ptr<int>(),
            static_cast<int>(v_rows),
            static_cast<int>(v_cols),
            static_cast<int>(v_batch_stride_rows),
            static_cast<int>(v_head_stride_rows),
            static_cast<int>(v_max_high_freq_count),
            static_cast<int>(v_max_full_count),
            static_cast<uint8_t>(v_start_exp),
            static_cast<int>(logical_kv_len),
            static_cast<int>(num_q_heads),
            static_cast<int>(num_kv_heads),
            static_cast<int>(head_dim),
            static_cast<float>(sm_scale),
            output.data_ptr<float>());
    CUDA_CHECK(cudaGetLastError());
    return output;
}
