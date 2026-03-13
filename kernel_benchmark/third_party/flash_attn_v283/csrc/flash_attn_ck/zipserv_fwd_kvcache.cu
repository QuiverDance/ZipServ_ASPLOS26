/******************************************************************************
 * Copyright (c) 2026, ZipServ Authors.
 ******************************************************************************/

#include "zipserv_fwd_kvcache.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "L_Kernel.cuh"
#include "MatMulUtilities.cuh"

#define ZIPSERV_CUDA_CHECK(expr)                                                   \
    do {                                                                           \
        cudaError_t err__ = (expr);                                                \
        TORCH_CHECK(err__ == cudaSuccess, #expr " failed: ", cudaGetErrorString(err__)); \
    } while (0)

namespace FLASH_NAMESPACE {
namespace {

constexpr int kTileRows = 64;
constexpr int kTileCols = 64;
constexpr int kThreadsPerBlock = 128;
constexpr int kTokensPerTile = 64;
constexpr int kQHeadsPerCta = 2;
constexpr int kThreadsPerQHead = kThreadsPerBlock / kQHeadsPerCta;
constexpr int kTokenGroupsPerQHead = kThreadsPerQHead / 16;
constexpr int kMaxOutputAccumsPerThread = 256 / kThreadsPerQHead;

static_assert(kThreadsPerBlock % kQHeadsPerCta == 0, "invalid grouped-head CTA shape");
static_assert(kThreadsPerQHead % 16 == 0, "grouped-head CTA requires 16-thread token groups");
static_assert(256 % kThreadsPerQHead == 0, "grouped-head CTA assumes exact head-dim coverage up to 256");

using ZipservConfig = TilingConfigBF16TripleBitmap<4, 1, 4>;

struct ZipservCompressedParams {
    const uint8_t *sign_mantissa;
    const __nv_bfloat16 *compressed_full;
    const uint64_t *bitmap1;
    const uint64_t *bitmap2;
    const uint64_t *bitmap3;
    const int *tile_offsets_median;
    const int *tile_offsets_global;
    int rows;
    int cols;
    int max_high_freq_count;
    int max_full_count;
    uint8_t start_exp;
};

struct ZipservFlashDecodeParams {
    const __nv_bfloat16 *q;
    __nv_bfloat16 *out;
    float *lse;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int padded_head_dim;
    int logical_kv_len;
    float softmax_scale;
    ZipservCompressedParams k;
    ZipservCompressedParams v;
};

void check_zipserv_compressed(
    const at::Tensor &sign_mantissa,
    const at::Tensor &compressed_full,
    const at::Tensor &bitmap1,
    const at::Tensor &bitmap2,
    const at::Tensor &bitmap3,
    const at::Tensor &tile_offsets_median,
    const at::Tensor &tile_offsets_global,
    const at::Device &device) {
    CHECK_DEVICE(sign_mantissa);
    CHECK_DEVICE(compressed_full);
    CHECK_DEVICE(bitmap1);
    CHECK_DEVICE(bitmap2);
    CHECK_DEVICE(bitmap3);
    CHECK_DEVICE(tile_offsets_median);
    CHECK_DEVICE(tile_offsets_global);
    CHECK_CONTIGUOUS(sign_mantissa);
    CHECK_CONTIGUOUS(compressed_full);
    CHECK_CONTIGUOUS(bitmap1);
    CHECK_CONTIGUOUS(bitmap2);
    CHECK_CONTIGUOUS(bitmap3);
    CHECK_CONTIGUOUS(tile_offsets_median);
    CHECK_CONTIGUOUS(tile_offsets_global);
    TORCH_CHECK(sign_mantissa.device() == device, "compressed tensor device mismatch");
    TORCH_CHECK(compressed_full.device() == device, "compressed tensor device mismatch");
    TORCH_CHECK(bitmap1.device() == device, "compressed tensor device mismatch");
    TORCH_CHECK(bitmap2.device() == device, "compressed tensor device mismatch");
    TORCH_CHECK(bitmap3.device() == device, "compressed tensor device mismatch");
    TORCH_CHECK(tile_offsets_median.device() == device, "compressed tensor device mismatch");
    TORCH_CHECK(tile_offsets_global.device() == device, "compressed tensor device mismatch");
    TORCH_CHECK(sign_mantissa.dtype() == torch::kUInt8, "sign_mantissa must be uint8");
    TORCH_CHECK(compressed_full.dtype() == torch::kBFloat16, "compressed_full must be bfloat16");
    TORCH_CHECK(bitmap1.dtype() == torch::kUInt64, "bitmap1 must be uint64");
    TORCH_CHECK(bitmap2.dtype() == torch::kUInt64, "bitmap2 must be uint64");
    TORCH_CHECK(bitmap3.dtype() == torch::kUInt64, "bitmap3 must be uint64");
    TORCH_CHECK(tile_offsets_median.dtype() == torch::kInt32, "tile_offsets_median must be int32");
    TORCH_CHECK(tile_offsets_global.dtype() == torch::kInt32, "tile_offsets_global must be int32");
}

size_t compute_shared_bytes(int max_high_freq_count, int max_full_count) {
    const int bitmap_size = ZipservConfig::TILE_BITMAP_M_V3 * ZipservConfig::TILE_BITMAP_K_V3;
    const size_t padded_high_freq =
        static_cast<size_t>(max_high_freq_count + ((128 - (max_high_freq_count % 128)) % 128));
    return static_cast<size_t>(kTileRows) * (kTileCols + PADDING_SHARED_MEM_FOR_DECOMP) * sizeof(__nv_bfloat16) +
           static_cast<size_t>(bitmap_size) * 3 * sizeof(uint64_t) +
           padded_high_freq * sizeof(uint8_t) +
           static_cast<size_t>(max_full_count) * sizeof(__nv_bfloat16);
}

template <typename TilingConfig>
__device__ __forceinline__ void load_zipserv_global_tile_to_shared(
    const ZipservCompressedParams &comp,
    int global_tile_m,
    int global_tile_k,
    __nv_bfloat16 (*smem_output)[64 + PADDING_SHARED_MEM_FOR_DECOMP],
    uint64_t *smem_bitmap1,
    uint64_t *smem_bitmap2,
    uint64_t *smem_bitmap3,
    uint8_t *smem_sign_mantissa,
    __nv_bfloat16 *smem_full_values) {
    const int warp_id = threadIdx.x / 32;
    const int global_tile_idx = global_tile_m * (comp.cols / 64) + global_tile_k;
    const int bitmap_global_offset = ((global_tile_m * 64) >> 3) * (comp.cols >> 3) + global_tile_k * 64;
    const uint64_t *bitmap1_ptr = comp.bitmap1 + bitmap_global_offset;
    const uint64_t *bitmap2_ptr = comp.bitmap2 + bitmap_global_offset;
    const uint64_t *bitmap3_ptr = comp.bitmap3 + bitmap_global_offset;
    const int *global_offset_ptr = comp.tile_offsets_global + global_tile_idx * 2;
    const int global_hf_start = global_offset_ptr[0];
    const int global_full_start = global_offset_ptr[1];
    const int global_hf_count = global_offset_ptr[2] - global_hf_start;
    const int global_full_count = global_offset_ptr[3] - global_full_start;

    CopyTripleBitmapToShared<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
        smem_bitmap1, smem_bitmap2, smem_bitmap3, bitmap1_ptr, bitmap2_ptr, bitmap3_ptr);
    CopyCompressedDataToShared<TilingConfig>(
        smem_sign_mantissa,
        smem_full_values,
        comp.sign_mantissa + global_hf_start,
        comp.compressed_full + global_full_start,
        global_hf_count,
        global_full_count);
    cp_async_group_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    const int median_tile_idx = global_tile_idx * 4 + warp_id;
    const int *median_offset_ptr = comp.tile_offsets_median + median_tile_idx * 2;
    const int warp_hf_start = median_offset_ptr[0];
    const int warp_full_start = median_offset_ptr[1];
    uint64_t *smem_bitmap1_warp = smem_bitmap1 + warp_id * 2 * 8;
    uint64_t *smem_bitmap2_warp = smem_bitmap2 + warp_id * 2 * 8;
    uint64_t *smem_bitmap3_warp = smem_bitmap3 + warp_id * 2 * 8;

    DecompressMedianTileToSharedMemory<TilingConfig>(
        smem_sign_mantissa,
        smem_full_values,
        smem_bitmap1_warp,
        smem_bitmap2_warp,
        smem_bitmap3_warp,
        warp_hf_start,
        warp_full_start,
        comp.start_exp,
        smem_output,
        warp_id);
    __syncthreads();
}

template <typename TilingConfig>
__global__ void zipserv_flash_decode_kernel(ZipservFlashDecodeParams params) {
    const int kv_group_size = params.num_q_heads / params.num_kv_heads;
    const int kv_head = blockIdx.x;
    // Each CTA serves one kv head and up to two q heads that share its compressed K/V tiles.
    const int q_head_group = blockIdx.y * kQHeadsPerCta;
    const int q_head_base = kv_head * kv_group_size + q_head_group;
    const int active_q_heads = max(0, min(kQHeadsPerCta, kv_group_size - q_head_group));
    const int tid = threadIdx.x;
    const int q_head_slot = tid / kThreadsPerQHead;
    const int local_tid = tid % kThreadsPerQHead;
    const int token_group = local_tid / 16;
    const int lane_in_group = local_tid % 16;
    const bool q_head_active = q_head_slot < active_q_heads;
    const int q_head = q_head_base + q_head_slot;

    extern __shared__ __align__(128) __nv_bfloat16 smem_buffer[];
    auto *smem_output = reinterpret_cast<__nv_bfloat16(*)[64 + PADDING_SHARED_MEM_FOR_DECOMP]>(smem_buffer);
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    uint64_t *smem_bitmap1 = reinterpret_cast<uint64_t *>(smem_output + kTileRows);
    uint64_t *smem_bitmap2 = smem_bitmap1 + bitmap_size;
    uint64_t *smem_bitmap3 = smem_bitmap2 + bitmap_size;
    uint8_t *smem_sign_mantissa = reinterpret_cast<uint8_t *>(smem_bitmap3 + bitmap_size);
    const int shared_max_high_freq_count = max(params.k.max_high_freq_count, params.v.max_high_freq_count);
    const size_t padding = (128 - (shared_max_high_freq_count % 128)) % 128;
    __nv_bfloat16 *smem_full_values =
        reinterpret_cast<__nv_bfloat16 *>(smem_sign_mantissa + shared_max_high_freq_count + padding);

    __shared__ float tile_scores[kQHeadsPerCta][kTokensPerTile];
    __shared__ float tile_weights[kQHeadsPerCta][kTokensPerTile];
    __shared__ float running_m[kQHeadsPerCta];
    __shared__ float running_l[kQHeadsPerCta];
    __shared__ float prev_scale[kQHeadsPerCta];
    __shared__ float inv_l[kQHeadsPerCta];

    if (tid < kQHeadsPerCta) {
        running_m[tid] = -INFINITY;
        running_l[tid] = 0.0f;
        prev_scale[tid] = 0.0f;
        inv_l[tid] = 0.0f;
    }
    __syncthreads();

    float acc[kMaxOutputAccumsPerThread] = {0.0f};
    const __nv_bfloat16 *q_ptr =
        q_head_active ? params.q + static_cast<int64_t>(q_head) * params.padded_head_dim : nullptr;
    const int q_col_tiles = params.padded_head_dim / 64;
    const int v_col_tiles = params.v.cols / 64;
    const int kv_tiles_per_head = params.k.rows / (params.num_kv_heads * kTileRows);
    const int kv_tiles = (params.logical_kv_len + kTokensPerTile - 1) / kTokensPerTile;
    const int kv_tile_base = kv_head * kv_tiles_per_head;

    for (int row_tile = 0; row_tile < kv_tiles; ++row_tile) {
        if (tid < active_q_heads * kTokensPerTile) {
            const int init_head = tid / kTokensPerTile;
            const int init_token = tid % kTokensPerTile;
            tile_scores[init_head][init_token] = 0.0f;
            tile_weights[init_head][init_token] = 0.0f;
        }
        __syncthreads();

        const int global_tile_m = kv_tile_base + row_tile;
        const int row_token_base = row_tile * kTokensPerTile;

        for (int col_tile = 0; col_tile < q_col_tiles; ++col_tile) {
            load_zipserv_global_tile_to_shared<TilingConfig>(
                params.k,
                global_tile_m,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            if (q_head_active) {
                const int q_base = col_tile * 64 + lane_in_group;
                const float q0 = __bfloat162float(q_ptr[q_base + 0]);
                const float q1 = __bfloat162float(q_ptr[q_base + 16]);
                const float q2 = __bfloat162float(q_ptr[q_base + 32]);
                const float q3 = __bfloat162float(q_ptr[q_base + 48]);
                #pragma unroll
                for (int token_block = 0; token_block < (kTokensPerTile / kTokenGroupsPerQHead); ++token_block) {
                    const int token_in_tile = token_block * kTokenGroupsPerQHead + token_group;
                    const int token = row_token_base + token_in_tile;
                    float dot = 0.0f;
                    if (token < params.logical_kv_len) {
                        dot += q0 *
                               __bfloat162float(smem_output[token_in_tile][lane_in_group + 0]);
                        dot += q1 *
                               __bfloat162float(smem_output[token_in_tile][lane_in_group + 16]);
                        dot += q2 *
                               __bfloat162float(smem_output[token_in_tile][lane_in_group + 32]);
                        dot += q3 *
                               __bfloat162float(smem_output[token_in_tile][lane_in_group + 48]);
                    }

                    for (int offset = 8; offset > 0; offset >>= 1) {
                        dot += __shfl_down_sync(0xFFFFFFFFu, dot, offset, 16);
                    }
                    if (lane_in_group == 0 && token < params.logical_kv_len) {
                        tile_scores[q_head_slot][token_in_tile] += dot;
                    }
                }
            }
            __syncthreads();
        }

        if (local_tid == 0 && q_head_active) {
            float tile_max = -INFINITY;
            for (int i = 0; i < kTokensPerTile; ++i) {
                const int tile_token = row_token_base + i;
                const float score = tile_token < params.logical_kv_len
                    ? tile_scores[q_head_slot][i] * params.softmax_scale
                    : -INFINITY;
                tile_scores[q_head_slot][i] = score;
                tile_max = fmaxf(tile_max, score);
            }
            const float new_m =
                running_l[q_head_slot] > 0.0f ? fmaxf(running_m[q_head_slot], tile_max) : tile_max;
            const float scale_prev =
                running_l[q_head_slot] > 0.0f ? __expf(running_m[q_head_slot] - new_m) : 0.0f;
            float tile_sum = 0.0f;
            for (int i = 0; i < kTokensPerTile; ++i) {
                const int tile_token = row_token_base + i;
                const float w = tile_token < params.logical_kv_len
                    ? __expf(tile_scores[q_head_slot][i] - new_m)
                    : 0.0f;
                tile_weights[q_head_slot][i] = w;
                tile_sum += w;
            }
            prev_scale[q_head_slot] = scale_prev;
            running_m[q_head_slot] = new_m;
            running_l[q_head_slot] = scale_prev * running_l[q_head_slot] + tile_sum;
        }
        __syncthreads();

        if (q_head_active) {
            #pragma unroll
            for (int acc_idx = 0; acc_idx < kMaxOutputAccumsPerThread; ++acc_idx) {
                acc[acc_idx] *= prev_scale[q_head_slot];
            }
        }

        for (int col_tile = 0; col_tile < v_col_tiles; ++col_tile) {
            load_zipserv_global_tile_to_shared<TilingConfig>(
                params.v,
                global_tile_m,
                col_tile,
                smem_output,
                smem_bitmap1,
                smem_bitmap2,
                smem_bitmap3,
                smem_sign_mantissa,
                smem_full_values);

            if (q_head_active) {
                const int col_start = col_tile * 64;
                #pragma unroll
                for (int acc_idx = 0; acc_idx < kMaxOutputAccumsPerThread; ++acc_idx) {
                    const int dim = local_tid + acc_idx * kThreadsPerQHead;
                    if (dim >= col_start && dim < col_start + 64 && dim < params.head_dim) {
                        const int local_col = dim - col_start;
                        #pragma unroll
                        for (int token_in_tile = 0; token_in_tile < kTokensPerTile; ++token_in_tile) {
                            const int tile_token = row_token_base + token_in_tile;
                            if (tile_token >= params.logical_kv_len) {
                                break;
                            }
                            acc[acc_idx] += tile_weights[q_head_slot][token_in_tile] *
                                __bfloat162float(smem_output[token_in_tile][local_col]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    if (local_tid == 0 && q_head_active) {
        inv_l[q_head_slot] = 1.0f / running_l[q_head_slot];
        params.lse[q_head] = running_m[q_head_slot] + logf(running_l[q_head_slot]);
    }
    __syncthreads();

    if (q_head_active) {
        #pragma unroll
        for (int acc_idx = 0; acc_idx < kMaxOutputAccumsPerThread; ++acc_idx) {
            const int dim = local_tid + acc_idx * kThreadsPerQHead;
            if (dim < params.head_dim) {
                params.out[static_cast<int64_t>(q_head) * params.head_dim + dim] =
                    __float2bfloat16(acc[acc_idx] * inv_l[q_head_slot]);
            }
        }
    }
}

}  // namespace

std::vector<at::Tensor> mha_fwd_kvcache_zipserv(
    at::Tensor &q,
    const at::Tensor &k_sign_mantissa,
    const at::Tensor &k_compressed_full,
    const at::Tensor &k_bitmap1,
    const at::Tensor &k_bitmap2,
    const at::Tensor &k_bitmap3,
    const at::Tensor &k_tile_offsets_median,
    const at::Tensor &k_tile_offsets_global,
    int64_t k_rows,
    int64_t k_cols,
    int64_t k_max_high_freq_count,
    int64_t k_max_full_count,
    int64_t k_start_exp,
    const at::Tensor &v_sign_mantissa,
    const at::Tensor &v_compressed_full,
    const at::Tensor &v_bitmap1,
    const at::Tensor &v_bitmap2,
    const at::Tensor &v_bitmap3,
    const at::Tensor &v_tile_offsets_median,
    const at::Tensor &v_tile_offsets_global,
    int64_t v_rows,
    int64_t v_cols,
    int64_t v_max_high_freq_count,
    int64_t v_max_full_count,
    int64_t v_start_exp,
    int64_t logical_kv_len,
    int64_t num_kv_heads,
    float softmax_scale) {
    CHECK_DEVICE(q);
    CHECK_CONTIGUOUS(q);
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(q.dim() == 4, "q must have shape (batch, seqlen_q, nheads, headdim)");
    TORCH_CHECK(q.size(0) == 1, "zipserv flash decode currently requires batch_size == 1");
    TORCH_CHECK(q.size(1) == 1, "zipserv flash decode currently requires q_len == 1");
    TORCH_CHECK(q.stride(-1) == 1, "q last dimension must be contiguous");

    const auto device = q.device();
    check_zipserv_compressed(
        k_sign_mantissa,
        k_compressed_full,
        k_bitmap1,
        k_bitmap2,
        k_bitmap3,
        k_tile_offsets_median,
        k_tile_offsets_global,
        device);
    check_zipserv_compressed(
        v_sign_mantissa,
        v_compressed_full,
        v_bitmap1,
        v_bitmap2,
        v_bitmap3,
        v_tile_offsets_median,
        v_tile_offsets_global,
        device);

    TORCH_CHECK(k_rows == v_rows, "K/V rows must match");
    TORCH_CHECK(k_cols == v_cols, "K/V cols must match");
    TORCH_CHECK(k_rows % kTileRows == 0, "compressed K rows must be padded to a multiple of 64");
    TORCH_CHECK(k_cols % kTileCols == 0, "compressed K cols must be padded to a multiple of 64");
    TORCH_CHECK(v_rows % kTileRows == 0, "compressed V rows must be padded to a multiple of 64");
    TORCH_CHECK(v_cols % kTileCols == 0, "compressed V cols must be padded to a multiple of 64");
    TORCH_CHECK(logical_kv_len > 0, "logical_kv_len must be > 0");
    TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be > 0");
    TORCH_CHECK(num_kv_heads == 8, "zipserv flash decode currently requires num_kv_heads == 8");

    const int64_t num_q_heads = q.size(2);
    const int64_t head_dim = q.size(3);
    TORCH_CHECK(head_dim > 0, "head_dim must be > 0");
    TORCH_CHECK(head_dim <= 256, "zipserv flash decode currently requires head_dim <= 256");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(k_rows % (num_kv_heads * kTileRows) == 0, "compressed K rows must be head-major padded by 64 rows per kv head");
    TORCH_CHECK(logical_kv_len * num_kv_heads <= k_rows, "logical_kv_len exceeds compressed row capacity");
    TORCH_CHECK(k_cols >= head_dim, "compressed K cols must be >= head_dim");
    TORCH_CHECK(v_cols >= head_dim, "compressed V cols must be >= head_dim");

    at::Tensor q_padded = q;
    if (head_dim != k_cols) {
        q_padded = torch::nn::functional::pad(
            q,
            torch::nn::functional::PadFuncOptions({0, static_cast<int64_t>(k_cols - head_dim)}));
    }
    auto q_flat = q_padded.view({num_q_heads, k_cols}).contiguous();

    at::Tensor out_flat = torch::empty({num_q_heads, head_dim}, q.options());
    at::Tensor lse_flat = torch::empty({num_q_heads}, q.options().dtype(torch::kFloat32));

    const at::cuda::CUDAGuard device_guard{device};
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(device.index()).stream();

    ZipservFlashDecodeParams params;
    params.q = reinterpret_cast<const __nv_bfloat16 *>(q_flat.data_ptr<at::BFloat16>());
    params.out = reinterpret_cast<__nv_bfloat16 *>(out_flat.data_ptr<at::BFloat16>());
    params.lse = lse_flat.data_ptr<float>();
    params.num_q_heads = static_cast<int>(num_q_heads);
    params.num_kv_heads = static_cast<int>(num_kv_heads);
    params.head_dim = static_cast<int>(head_dim);
    params.padded_head_dim = static_cast<int>(k_cols);
    params.logical_kv_len = static_cast<int>(logical_kv_len);
    params.softmax_scale = softmax_scale;
    params.k = {
        k_sign_mantissa.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16 *>(k_compressed_full.data_ptr<at::BFloat16>()),
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
    };
    params.v = {
        v_sign_mantissa.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16 *>(v_compressed_full.data_ptr<at::BFloat16>()),
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
    };

    const int max_high_freq_count = std::max(static_cast<int>(k_max_high_freq_count), static_cast<int>(v_max_high_freq_count));
    const int max_full_count = std::max(static_cast<int>(k_max_full_count), static_cast<int>(v_max_full_count));
    const int kv_group_size = static_cast<int>(num_q_heads / num_kv_heads);
    const int q_head_groups = (kv_group_size + kQHeadsPerCta - 1) / kQHeadsPerCta;
    const size_t shared_bytes = compute_shared_bytes(max_high_freq_count, max_full_count);
    ZIPSERV_CUDA_CHECK(cudaFuncSetAttribute(
        zipserv_flash_decode_kernel<ZipservConfig>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    zipserv_flash_decode_kernel<ZipservConfig>
        <<<dim3(
               static_cast<unsigned int>(num_kv_heads),
               static_cast<unsigned int>(q_head_groups),
               1),
           kThreadsPerBlock,
           shared_bytes,
           stream>>>(params);
    ZIPSERV_CUDA_CHECK(cudaGetLastError());

    return {
        out_flat.view({1, 1, num_q_heads, head_dim}),
        lse_flat.view({1, num_q_heads, 1}),
    };
}

}  // namespace FLASH_NAMESPACE
