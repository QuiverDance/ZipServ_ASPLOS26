#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include "L_API.cuh"
#include "../csrc/L_Kernel.cuh"

namespace {

constexpr int kTileM = 64;
constexpr int kTileK = 64;
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;
constexpr int kMaxHeadDimTiles = 4;

#define CUDA_CHECK(expr) TORCH_CHECK((expr) == cudaSuccess, #expr " failed")

size_t ComputeSharedBytes(int max_high_freq_count, int max_full_count) {
    using Config = TilingConfigBF16TripleBitmap<4, 1, 4>;
    const int bitmap_size = Config::TILE_BITMAP_M_V3 * Config::TILE_BITMAP_K_V3;
    return static_cast<size_t>(kTileM) * (kTileK + PADDING_SHARED_MEM_FOR_DECOMP) * sizeof(__nv_bfloat16) +
           static_cast<size_t>(bitmap_size) * 3 * sizeof(uint64_t) +
           static_cast<size_t>(max_high_freq_count + ((128 - (max_high_freq_count % 128)) % 128)) * sizeof(uint8_t) +
           static_cast<size_t>(max_full_count) * sizeof(__nv_bfloat16);
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
__global__ void ZipservFlashAttnDecodeKernel(
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
    const int num_head_tiles = head_dim / 64;
    const int tokens_per_row_tile = kTileM / num_kv_heads;

    extern __shared__ __align__(128) __nv_bfloat16 smem_buffer[];
    auto* smem_output = reinterpret_cast<__nv_bfloat16(*)[64 + PADDING_SHARED_MEM_FOR_DECOMP]>(smem_buffer);
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    uint64_t* smem_bitmap1 = reinterpret_cast<uint64_t*>(smem_output + 64);
    uint64_t* smem_bitmap2 = smem_bitmap1 + bitmap_size;
    uint64_t* smem_bitmap3 = smem_bitmap2 + bitmap_size;
    uint8_t* smem_sign_mantissa = reinterpret_cast<uint8_t*>(smem_bitmap3 + bitmap_size);
    const size_t max_high_freq_count = static_cast<size_t>(
        k_max_high_freq_count > v_max_high_freq_count ? k_max_high_freq_count : v_max_high_freq_count);
    const size_t max_high_freq_padding = (128 - (max_high_freq_count % 128)) % 128;
    __nv_bfloat16* smem_full_values =
        reinterpret_cast<__nv_bfloat16*>(smem_sign_mantissa + max_high_freq_count + max_high_freq_padding);

    __shared__ float tile_scores[8];
    __shared__ float tile_exp[8];
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float alpha_scale;

    float numerators[kMaxHeadDimTiles] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (threadIdx.x == 0) {
        running_max = -FLT_MAX;
        running_sum = 0.0f;
        alpha_scale = 0.0f;
    }
    __syncthreads();

    const int row_tiles = (logical_kv_len + tokens_per_row_tile - 1) / tokens_per_row_tile;
    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        if (threadIdx.x < tokens_per_row_tile) {
            tile_scores[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int col_tile = 0; col_tile < num_head_tiles; ++col_tile) {
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

            float acc = 0.0f;
            const int token = row_tile * tokens_per_row_tile + token_group;
            const int local_row = kv_head + token_group * num_kv_heads;
            if (token_group < tokens_per_row_tile && token < logical_kv_len) {
                const int q_base = q_head * head_dim + col_tile * 64;
                const int col0 = lane_in_group + 0;
                const int col1 = lane_in_group + 16;
                const int col2 = lane_in_group + 32;
                const int col3 = lane_in_group + 48;
                acc += __bfloat162float(q[q_base + col0]) * __bfloat162float(smem_output[local_row][col0]);
                acc += __bfloat162float(q[q_base + col1]) * __bfloat162float(smem_output[local_row][col1]);
                acc += __bfloat162float(q[q_base + col2]) * __bfloat162float(smem_output[local_row][col2]);
                acc += __bfloat162float(q[q_base + col3]) * __bfloat162float(smem_output[local_row][col3]);
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, 16);
            }
            if (lane_in_group == 0 && token_group < tokens_per_row_tile && token < logical_kv_len) {
                tile_scores[token_group] += acc * sm_scale;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float tile_max = -FLT_MAX;
            for (int i = 0; i < tokens_per_row_tile; ++i) {
                const int token = row_tile * tokens_per_row_tile + i;
                if (token < logical_kv_len) {
                    tile_max = fmaxf(tile_max, tile_scores[i]);
                } else {
                    tile_exp[i] = 0.0f;
                }
            }
            const float next_max = fmaxf(running_max, tile_max);
            const float alpha = running_sum > 0.0f ? expf(running_max - next_max) : 0.0f;
            float next_sum = running_sum * alpha;
            for (int i = 0; i < tokens_per_row_tile; ++i) {
                const int token = row_tile * tokens_per_row_tile + i;
                const float p = token < logical_kv_len ? expf(tile_scores[i] - next_max) : 0.0f;
                tile_exp[i] = p;
                next_sum += p;
            }
            running_max = next_max;
            running_sum = next_sum;
            alpha_scale = alpha;
        }
        __syncthreads();

        for (int col_tile = 0; col_tile < num_head_tiles; ++col_tile) {
            numerators[col_tile] *= alpha_scale;
        }

        for (int col_tile = 0; col_tile < num_head_tiles; ++col_tile) {
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

            if (threadIdx.x < 64) {
                const int dim = col_tile * 64 + threadIdx.x;
                if (dim < head_dim) {
                    float acc = numerators[col_tile];
                    #pragma unroll
                    for (int token_in_tile = 0; token_in_tile < 8; ++token_in_tile) {
                        if (token_in_tile >= tokens_per_row_tile) {
                            break;
                        }
                        const int token = row_tile * tokens_per_row_tile + token_in_tile;
                        if (token >= logical_kv_len) {
                            break;
                        }
                        const int local_row = kv_head + token_in_tile * num_kv_heads;
                        acc += tile_exp[token_in_tile] * __bfloat162float(smem_output[local_row][threadIdx.x]);
                    }
                    numerators[col_tile] = acc;
                }
            }
            __syncthreads();
        }
    }

    const float inv_sum = running_sum > 0.0f ? (1.0f / running_sum) : 0.0f;
    if (threadIdx.x < 64) {
        for (int col_tile = 0; col_tile < num_head_tiles; ++col_tile) {
            const int dim = col_tile * 64 + threadIdx.x;
            if (dim < head_dim) {
                output[q_head * head_dim + dim] = numerators[col_tile] * inv_sum;
            }
        }
    }
}

}  // namespace

torch::Tensor zipserv_flashattn_decode_attention_cuda(
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
    TORCH_CHECK(num_kv_heads == 8, "zipserv flashattn currently requires num_kv_heads == 8");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(head_dim > 0 && head_dim <= 256, "zipserv flashattn requires 0 < head_dim <= 256");
    TORCH_CHECK(head_dim % 64 == 0, "zipserv flashattn requires head_dim to be a multiple of 64");
    TORCH_CHECK(head_dim / 64 <= kMaxHeadDimTiles, "zipserv flashattn supports head_dim up to 256");
    TORCH_CHECK(logical_kv_len > 0, "logical_kv_len must be > 0");
    TORCH_CHECK(logical_kv_len <= (k_rows / num_kv_heads), "logical_kv_len exceeds compressed row capacity");

    using Config = TilingConfigBF16TripleBitmap<4, 1, 4>;
    auto stream = at::cuda::getDefaultCUDAStream(q.device().index()).stream();
    auto output = torch::empty(
        {num_q_heads, head_dim},
        q.options().dtype(torch::kFloat32));
    const size_t shared_bytes = std::max(
        ComputeSharedBytes(static_cast<int>(k_max_high_freq_count), static_cast<int>(k_max_full_count)),
        ComputeSharedBytes(static_cast<int>(v_max_high_freq_count), static_cast<int>(v_max_full_count)));
    ZipservFlashAttnDecodeKernel<Config><<<static_cast<unsigned int>(num_q_heads), kThreadsPerBlock, shared_bytes, stream>>>(
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
