#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include "flash.h"
#include "flash_fwd_kernel.h"

namespace zipserv_tile_bench {

using KernelTraits = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
using Element = typename KernelTraits::Element;

constexpr int kThreads = KernelTraits::kNThreads;
constexpr int kBlockM = KernelTraits::kBlockM;
constexpr int kBlockN = KernelTraits::kBlockN;
constexpr int kHeadDim = KernelTraits::kHeadDim;
constexpr int kTileRows = 64;
constexpr int kTileCols = 64;
constexpr int kNumMeasuredStages = 11;
constexpr int kTimingBracketIdx = kNumMeasuredStages;
constexpr int kNumOutputs = kNumMeasuredStages + 1;

size_t AlignUp(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

size_t PaddedHighFreqCount(int max_high_freq_count) {
    return static_cast<size_t>(max_high_freq_count + ((128 - (max_high_freq_count % 128)) % 128));
}

size_t ScratchBufferBytes(int max_high_freq_count, int max_full_count) {
    constexpr size_t kBitmapWords =
        FLASH_NAMESPACE::ZipservFlashConfig::TILE_BITMAP_M_V3 *
        FLASH_NAMESPACE::ZipservFlashConfig::TILE_BITMAP_K_V3;
    constexpr size_t kMetadataBytes =
        static_cast<size_t>(
            (2 * FLASH_NAMESPACE::kZipservWarpsPerTile +
             FLASH_NAMESPACE::kZipservWarpsPerTile * FLASH_NAMESPACE::kZipservSmallTilesPerWarp) *
            sizeof(int));
    const size_t payload_bytes =
        kBitmapWords * 3 * sizeof(uint64_t) +
        PaddedHighFreqCount(max_high_freq_count) +
        static_cast<size_t>(max_full_count) * sizeof(__nv_bfloat16) +
        kMetadataBytes;
    return AlignUp(payload_bytes, 128);
}

void CheckTensor(
    const at::Tensor& tensor,
    at::ScalarType expected_dtype,
    const at::Device& device,
    const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == expected_dtype, name, " has an unexpected dtype");
    TORCH_CHECK(tensor.device() == device, name, " must be on the same device as the other tensors");
}

FLASH_NAMESPACE::ZipservCompressedKVParams MakeCompressedParams(
    const at::Tensor& sign_mantissa,
    const at::Tensor& compressed_full,
    const at::Tensor& bitmap1,
    const at::Tensor& bitmap2,
    const at::Tensor& bitmap3,
    const at::Tensor& tile_offsets_median,
    const at::Tensor& tile_offsets_global,
    int rows,
    int cols,
    int batch_stride_rows,
    int head_stride_rows,
    int max_high_freq_count,
    int max_full_count,
    int start_exp,
    int layout) {
    return {
        sign_mantissa.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(compressed_full.data_ptr<at::BFloat16>()),
        bitmap1.data_ptr<uint64_t>(),
        bitmap2.data_ptr<uint64_t>(),
        bitmap3.data_ptr<uint64_t>(),
        tile_offsets_median.data_ptr<int>(),
        tile_offsets_global.data_ptr<int>(),
        rows,
        cols,
        batch_stride_rows,
        head_stride_rows,
        max_high_freq_count,
        max_full_count,
        start_exp,
        layout,
    };
}

template <typename TensorDst>
__device__ void DecompressTileToDst(
    const FLASH_NAMESPACE::ZipservCompressedKVParams& comp,
    const FLASH_NAMESPACE::ZipservGlobalTileInfo& tile_info,
    const FLASH_NAMESPACE::ZipservScratchBufferView& scratch,
    TensorDst& dst_smem,
    int row_base,
    int col_base,
    int valid_rows,
    int valid_cols,
    int num_heads_k,
    int head_in_tile) {
    if (comp.layout == FLASH_NAMESPACE::ZIPSERV_LAYOUT_TOKEN_MAJOR) {
        if (valid_rows == kTileRows / max(1, num_heads_k) && valid_cols == kTileCols) {
            FLASH_NAMESPACE::zipserv_decompress_scratch_to_dst_token_major<cutlass::bfloat16_t, /*FullTile=*/true>(
                comp,
                tile_info,
                scratch,
                dst_smem,
                row_base,
                col_base,
                valid_rows,
                valid_cols,
                num_heads_k,
                head_in_tile);
        } else {
            FLASH_NAMESPACE::zipserv_decompress_scratch_to_dst_token_major<cutlass::bfloat16_t, /*FullTile=*/false>(
                comp,
                tile_info,
                scratch,
                dst_smem,
                row_base,
                col_base,
                valid_rows,
                valid_cols,
                num_heads_k,
                head_in_tile);
        }
    } else if (valid_rows == kTileRows && valid_cols == kTileCols) {
        FLASH_NAMESPACE::zipserv_decompress_scratch_to_dst<cutlass::bfloat16_t, /*FullTile=*/true>(
            comp,
            tile_info,
            scratch,
            dst_smem,
            row_base,
            col_base,
            valid_rows,
            valid_cols);
    } else {
        FLASH_NAMESPACE::zipserv_decompress_scratch_to_dst<cutlass::bfloat16_t, /*FullTile=*/false>(
            comp,
            tile_info,
            scratch,
            dst_smem,
            row_base,
            col_base,
            valid_rows,
            valid_cols);
    }
}

template <typename TensorDst>
__device__ void LoadTwoTiles(
    const FLASH_NAMESPACE::ZipservCompressedKVParams& comp,
    int batch_idx,
    int kv_head,
    int block_row,
    int total_valid_rows,
    int block_rows,
    char* zipserv_smem,
    TensorDst& dst_smem,
    bool measure,
    unsigned long long* accum,
    int load1_idx,
    unsigned long long* stage_start) {
    using namespace FLASH_NAMESPACE;

    ZipservBlockLoadState load_state{};

    __syncthreads();
    if (measure && threadIdx.x == 0) {
        *stage_start = clock64();
    }
    __syncthreads();
    load_state = zipserv_begin_block_load(
        comp,
        batch_idx,
        kv_head,
        block_row,
        total_valid_rows,
        block_rows,
        zipserv_smem);
    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        accum[load1_idx + 0] += clock64() - *stage_start;
    }
    __syncthreads();

    if (!load_state.valid) {
        return;
    }

    auto current_scratch = zipserv_get_scratch_buffer(comp, zipserv_smem, 0);
    const ZipservGlobalTileInfo current_tile_info = load_state.first_tile_info;
    const int current_valid_rows =
        zipserv_valid_rows(load_state.rows_to_load, /*row_start=*/0, load_state.rows_per_global_tile);
    const int current_valid_cols = min(kTileCols, comp.cols);

    __syncthreads();
    if (measure && threadIdx.x == 0) {
        *stage_start = clock64();
    }
    __syncthreads();
    DecompressTileToDst(
        comp,
        current_tile_info,
        current_scratch,
        dst_smem,
        /*row_base=*/0,
        /*col_base=*/0,
        current_valid_rows,
        current_valid_cols,
        load_state.num_heads_k,
        load_state.head_in_tile);
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        accum[load1_idx + 1] += clock64() - *stage_start;
    }
    __syncthreads();

    if (load_state.total_tiles <= 1) {
        return;
    }

    constexpr int second_tile_idx = 1;
    const int second_row_tile = second_tile_idx / load_state.max_col_tiles;
    const int second_col_tile = second_tile_idx % load_state.max_col_tiles;
    ZipservGlobalTileInfo second_tile_info{};
    auto second_scratch = zipserv_get_scratch_buffer(comp, zipserv_smem, 1);

    __syncthreads();
    if (measure && threadIdx.x == 0) {
        *stage_start = clock64();
    }
    __syncthreads();
    zipserv_prefetch_global_tile_to_shared(
        comp,
        load_state.global_tile_m_base + second_row_tile,
        second_col_tile,
        second_scratch,
        second_tile_info);
    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        accum[load1_idx + 2] += clock64() - *stage_start;
    }
    __syncthreads();

    const int second_row_base = second_row_tile * load_state.rows_per_global_tile;
    const int second_valid_rows =
        zipserv_valid_rows(load_state.rows_to_load, second_row_base, load_state.rows_per_global_tile);
    const int second_col_base = second_col_tile * kTileCols;
    const int second_valid_cols = min(kTileCols, comp.cols - second_col_base);

    __syncthreads();
    if (measure && threadIdx.x == 0) {
        *stage_start = clock64();
    }
    __syncthreads();
    DecompressTileToDst(
        comp,
        second_tile_info,
        second_scratch,
        dst_smem,
        second_row_base,
        second_col_base,
        second_valid_rows,
        second_valid_cols,
        load_state.num_heads_k,
        load_state.head_in_tile);
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        accum[load1_idx + 3] += clock64() - *stage_start;
    }
    __syncthreads();
}

__device__ void AccumulateTimingBracketCycles(
    bool measure,
    unsigned long long* accum,
    unsigned long long* stage_start) {
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        *stage_start = clock64();
    }
    __syncthreads();
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        accum[kTimingBracketIdx] += clock64() - *stage_start;
    }
    __syncthreads();
}

template <typename TensorQ, typename TensorCoords>
__device__ void InitializeSyntheticQ(TensorQ& tQsQ, TensorCoords& tQcQ) {
    #pragma unroll
    for (int i = 0; i < cute::size<0>(tQsQ); ++i) {
        #pragma unroll
        for (int j = 0; j < cute::size<1>(tQsQ); ++j) {
            #pragma unroll
            for (int k = 0; k < cute::size<2>(tQsQ); ++k) {
                const auto coord = tQcQ(i, j, k);
                const int row = static_cast<int>(cute::get<0>(coord));
                const int col = static_cast<int>(cute::get<1>(coord));
                const float value = static_cast<float>(((row * 17 + col * 5 + 3) % 19) - 9) / 8.0f;
                tQsQ(i, j, k) = Element(value);
            }
        }
    }
}

template <typename TensorQ, typename TensorK, typename TensorVtNoSwizzle, typename TensorVt, typename TensorAccO>
__device__ void RunFlashDecodeBlock(
    TensorQ& sQ,
    TensorK& sK,
    int block_row,
    int actual_seqlen_k,
    int actual_seqlen_q,
    TensorVtNoSwizzle& sVtNoSwizzle,
    TensorVt& sVt,
    TensorAccO& acc_o,
    FLASH_NAMESPACE::Softmax<2 * decltype(cute::size<1>(acc_o))::value>& softmax,
    bool is_first,
    float* qk_sink,
    float* pv_sink,
    float* out_sink,
    bool write_sink,
    bool measure,
    unsigned long long* accum,
    unsigned long long* stage_start) {
    using namespace FLASH_NAMESPACE;

    typename KernelTraits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(threadIdx.x);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename KernelTraits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor acc_s = partition_fragment_C(tiled_mma, cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockN>>{});
    clear(acc_s);
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        *stage_start = clock64();
    }
    __syncthreads();
    FLASH_NAMESPACE::gemm</*A_in_regs=*/KernelTraits::Is_Q_in_regs>(
        acc_s,
        tSrQ,
        tSrK,
        tSsQ,
        tSsK,
        tiled_mma,
        smem_tiled_copy_Q,
        smem_tiled_copy_K,
        smem_thr_copy_Q,
        smem_thr_copy_K);
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        accum[8] += clock64() - *stage_start;
    }

    FLASH_NAMESPACE::Mask</*Is_causal=*/false, /*Is_local=*/false, /*Has_alibi=*/false> mask(
        /*max_seqlen_k=*/actual_seqlen_k,
        /*max_seqlen_q=*/actual_seqlen_q,
        /*window_size_left=*/-1,
        /*window_size_right=*/-1,
        /*alibi_slope=*/0.0f);
    constexpr float kSoftmaxScaleLog2 = float(M_LOG2E) * M_SQRT1_2 / 8.0f;  // log2(e) / sqrt(128)
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        *stage_start = clock64();
    }
    __syncthreads();
    mask.template apply_mask</*Causal_mask=*/false, /*Is_even_MN=*/false>(
        acc_s,
        block_row,
        (threadIdx.x / 32) * 16 + (threadIdx.x % 32) / 4,
        KernelTraits::kNWarps * 16);
    if (is_first) {
        softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/false>(acc_s, acc_o, kSoftmaxScaleLog2);
    } else {
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(acc_s, acc_o, kSoftmaxScaleLog2);
    }
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        accum[9] += clock64() - *stage_start;
    }

    Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
    Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename KernelTraits::TiledMma>(rP.layout()));
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        *stage_start = clock64();
    }
    __syncthreads();
    FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    __syncthreads();
    if (measure && threadIdx.x == 0) {
        accum[10] += clock64() - *stage_start;
    }

    if (write_sink) {
        float qk_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < cute::size<0>(acc_s); ++i) {
            #pragma unroll
            for (int j = 0; j < cute::size<1>(acc_s); ++j) {
                #pragma unroll
                for (int k = 0; k < cute::size<2>(acc_s); ++k) {
                    qk_sum += acc_s(i, j, k);
                }
            }
        }
        float pv_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < cute::size<0>(acc_o); ++i) {
            #pragma unroll
            for (int j = 0; j < cute::size<1>(acc_o); ++j) {
                #pragma unroll
                for (int k = 0; k < cute::size<2>(acc_o); ++k) {
                    pv_sum += acc_o(i, j, k);
                }
            }
        }
        qk_sink[threadIdx.x] = qk_sum;
        pv_sink[threadIdx.x] = pv_sum;
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                qk_sink[threadIdx.x] += qk_sink[threadIdx.x + stride];
                pv_sink[threadIdx.x] += pv_sink[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out_sink[0] = qk_sink[0];
            out_sink[1] = pv_sink[0];
        }
    }
}

__global__ void MeasureZipservTileStagesKernel(
    FLASH_NAMESPACE::ZipservCompressedKVParams k_comp,
    FLASH_NAMESPACE::ZipservCompressedKVParams v_comp,
    int batch_idx,
    int kv_head,
    int target_block_row,
    int actual_seqlen_q,
    int logical_seqlen_k,
    int block_rows,
    int warmup,
    int iters,
    float* out_sink,
    unsigned long long* out_cycles) {
    extern __shared__ __align__(128) unsigned char shared_bytes[];

    Element* smem_ptr = reinterpret_cast<Element*>(shared_bytes);
    auto sQ = cute::make_tensor(cute::make_smem_ptr(smem_ptr), typename KernelTraits::SmemLayoutQ{});
    auto sK = cute::make_tensor(
        sQ.data() + (KernelTraits::Share_Q_K_smem ? 0 : cute::size(sQ)),
        typename KernelTraits::SmemLayoutKV{});
    auto sV = cute::make_tensor(sK.data() + cute::size(sK), typename KernelTraits::SmemLayoutKV{});
    auto sVt = cute::make_tensor(sV.data(), typename KernelTraits::SmemLayoutVtransposed{});
    auto sVtNoSwizzle = cute::make_tensor(sV.data().get(), typename KernelTraits::SmemLayoutVtransposedNoSwizzle{});
    char* zipserv_smem = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(shared_bytes + KernelTraits::kSmemSize + 127) & ~uintptr_t(127)));

    __shared__ unsigned long long accum[kNumOutputs];
    __shared__ unsigned long long stage_start;
    __shared__ float qk_sink[kThreads];
    __shared__ float pv_sink[kThreads];

    if (threadIdx.x < kNumOutputs) {
        accum[threadIdx.x] = 0;
    }
    __syncthreads();

    typename KernelTraits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(threadIdx.x);
    auto tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    auto cQ = cute::make_identity_tensor(cute::make_shape(cute::size<0>(sQ), cute::size<1>(sQ)));
    auto tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    InitializeSyntheticQ(tQsQ, tQcQ);
    #pragma unroll
    for (int i = 0; i < cute::size<0>(tQsQ); ++i) {
        #pragma unroll
        for (int j = 0; j < cute::size<1>(tQsQ); ++j) {
            #pragma unroll
            for (int k = 0; k < cute::size<2>(tQsQ); ++k) {
                const auto coord = tQcQ(i, j, k);
                const int row = static_cast<int>(cute::get<0>(coord));
                if (row >= actual_seqlen_q) {
                    tQsQ(i, j, k) = Element(0.0f);
                }
            }
        }
    }
    __syncthreads();

    const int total_loops = warmup + iters;
    const int n_block_max = (logical_seqlen_k + kBlockN - 1) / kBlockN;
    const int target_block_idx = target_block_row / kBlockN;
    for (int iter = 0; iter < total_loops; ++iter) {
        typename KernelTraits::TiledMma tiled_mma;
        Tensor acc_o = partition_fragment_C(tiled_mma, cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>{});
        clear(acc_o);
        constexpr int kSoftmaxRows = 2 * decltype(cute::size<1>(acc_o))::value;
        FLASH_NAMESPACE::Softmax<kSoftmaxRows> softmax;

        for (int n_block = n_block_max - 1; n_block >= 0; --n_block) {
            const int block_row = n_block * kBlockN;
            const int total_valid_rows = max(0, logical_seqlen_k - block_row);
            const bool measure = n_block == target_block_idx;
            AccumulateTimingBracketCycles(measure, accum, &stage_start);
            LoadTwoTiles(
                k_comp,
                batch_idx,
                kv_head,
                block_row,
                total_valid_rows,
                block_rows,
                zipserv_smem,
                sK,
                measure,
                accum,
                /*load1_idx=*/0,
                &stage_start);
            LoadTwoTiles(
                v_comp,
                batch_idx,
                kv_head,
                block_row,
                total_valid_rows,
                block_rows,
                zipserv_smem,
                sV,
                measure,
                accum,
                /*load1_idx=*/4,
                &stage_start);
            RunFlashDecodeBlock(
                sQ,
                sK,
                block_row,
                logical_seqlen_k,
                actual_seqlen_q,
                sVtNoSwizzle,
                sVt,
                acc_o,
                softmax,
                /*is_first=*/n_block == n_block_max - 1,
                qk_sink,
                pv_sink,
                out_sink,
                (iter + 1 == total_loops) && measure,
                measure,
                accum,
                &stage_start);
        }
        if (iter + 1 == warmup) {
            __syncthreads();
            if (threadIdx.x < kNumOutputs) {
                accum[threadIdx.x] = 0;
            }
            __syncthreads();
        }
    }

    if (threadIdx.x < kNumOutputs) {
        out_cycles[threadIdx.x] = accum[threadIdx.x] / max(1, iters);
    }
}

torch::Tensor measure_zipserv_tile_stages(
    const at::Tensor& k_sign_mantissa,
    const at::Tensor& k_compressed_full,
    const at::Tensor& k_bitmap1,
    const at::Tensor& k_bitmap2,
    const at::Tensor& k_bitmap3,
    const at::Tensor& k_tile_offsets_median,
    const at::Tensor& k_tile_offsets_global,
    int64_t k_rows,
    int64_t k_cols,
    int64_t k_batch_stride_rows,
    int64_t k_head_stride_rows,
    int64_t k_max_high_freq_count,
    int64_t k_max_full_count,
    int64_t k_start_exp,
    int64_t k_layout,
    const at::Tensor& v_sign_mantissa,
    const at::Tensor& v_compressed_full,
    const at::Tensor& v_bitmap1,
    const at::Tensor& v_bitmap2,
    const at::Tensor& v_bitmap3,
    const at::Tensor& v_tile_offsets_median,
    const at::Tensor& v_tile_offsets_global,
    int64_t v_rows,
    int64_t v_cols,
    int64_t v_batch_stride_rows,
    int64_t v_head_stride_rows,
    int64_t v_max_high_freq_count,
    int64_t v_max_full_count,
    int64_t v_start_exp,
    int64_t v_layout,
    int64_t batch_idx,
    int64_t kv_head,
    int64_t block_row,
    int64_t actual_seqlen_q,
    int64_t block_rows,
    int64_t logical_seqlen_k,
    int64_t warmup,
    int64_t iters) {
    TORCH_CHECK(warmup >= 0, "warmup must be non-negative");
    TORCH_CHECK(iters > 0, "iters must be positive");
    TORCH_CHECK(block_rows > 0 && block_rows <= kBlockN, "block_rows must be in (0, 64]");
    TORCH_CHECK(k_cols > 0 && k_cols <= kHeadDim, "K cols must be in (0, 128]");
    TORCH_CHECK(v_cols > 0 && v_cols <= kHeadDim, "V cols must be in (0, 128]");
    TORCH_CHECK(k_rows > 0 && v_rows > 0, "rows must be positive");
    TORCH_CHECK(actual_seqlen_q > 0 && actual_seqlen_q <= kBlockM, "actual_seqlen_q must be in (0, 64]");
    TORCH_CHECK(logical_seqlen_k > 0, "logical_seqlen_k must be positive");
    TORCH_CHECK(block_row >= 0 && block_row < logical_seqlen_k, "block_row must refer to a valid K/V block");

    const at::Device device = k_sign_mantissa.device();
    CheckTensor(k_sign_mantissa, torch::kUInt8, device, "k_sign_mantissa");
    CheckTensor(k_compressed_full, torch::kBFloat16, device, "k_compressed_full");
    CheckTensor(k_bitmap1, torch::kUInt64, device, "k_bitmap1");
    CheckTensor(k_bitmap2, torch::kUInt64, device, "k_bitmap2");
    CheckTensor(k_bitmap3, torch::kUInt64, device, "k_bitmap3");
    CheckTensor(k_tile_offsets_median, torch::kInt32, device, "k_tile_offsets_median");
    CheckTensor(k_tile_offsets_global, torch::kInt32, device, "k_tile_offsets_global");
    CheckTensor(v_sign_mantissa, torch::kUInt8, device, "v_sign_mantissa");
    CheckTensor(v_compressed_full, torch::kBFloat16, device, "v_compressed_full");
    CheckTensor(v_bitmap1, torch::kUInt64, device, "v_bitmap1");
    CheckTensor(v_bitmap2, torch::kUInt64, device, "v_bitmap2");
    CheckTensor(v_bitmap3, torch::kUInt64, device, "v_bitmap3");
    CheckTensor(v_tile_offsets_median, torch::kInt32, device, "v_tile_offsets_median");
    CheckTensor(v_tile_offsets_global, torch::kInt32, device, "v_tile_offsets_global");

    const auto guard = c10::cuda::CUDAGuard(device);

    const auto k_comp = MakeCompressedParams(
        k_sign_mantissa,
        k_compressed_full,
        k_bitmap1,
        k_bitmap2,
        k_bitmap3,
        k_tile_offsets_median,
        k_tile_offsets_global,
        static_cast<int>(k_rows),
        static_cast<int>(k_cols),
        static_cast<int>(k_batch_stride_rows),
        static_cast<int>(k_head_stride_rows),
        static_cast<int>(k_max_high_freq_count),
        static_cast<int>(k_max_full_count),
        static_cast<int>(k_start_exp),
        static_cast<int>(k_layout));
    const auto v_comp = MakeCompressedParams(
        v_sign_mantissa,
        v_compressed_full,
        v_bitmap1,
        v_bitmap2,
        v_bitmap3,
        v_tile_offsets_median,
        v_tile_offsets_global,
        static_cast<int>(v_rows),
        static_cast<int>(v_cols),
        static_cast<int>(v_batch_stride_rows),
        static_cast<int>(v_head_stride_rows),
        static_cast<int>(v_max_high_freq_count),
        static_cast<int>(v_max_full_count),
        static_cast<int>(v_start_exp),
        static_cast<int>(v_layout));

    constexpr size_t kBaseSmemBytes = (KernelTraits::kSmemSize + 127) & ~size_t(127);
    const size_t shared_bytes =
        kBaseSmemBytes +
        2 * std::max(
                ScratchBufferBytes(static_cast<int>(k_max_high_freq_count), static_cast<int>(k_max_full_count)),
                ScratchBufferBytes(static_cast<int>(v_max_high_freq_count), static_cast<int>(v_max_full_count)));

    int current_device = -1;
    cudaError_t ce = cudaGetDevice(&current_device);
    TORCH_CHECK(ce == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(ce));
    cudaDeviceProp prop{};
    ce = cudaGetDeviceProperties(&prop, current_device);
    TORCH_CHECK(ce == cudaSuccess, "cudaGetDeviceProperties failed: ", cudaGetErrorString(ce));
    TORCH_CHECK(
        shared_bytes <= static_cast<size_t>(prop.sharedMemPerBlockOptin),
        "requested shared memory (", shared_bytes, " bytes) exceeds device opt-in limit (",
        prop.sharedMemPerBlockOptin, " bytes)");

    auto out_sink = torch::empty({2}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto out_cycles = torch::empty({kNumOutputs}, torch::TensorOptions().device(device).dtype(torch::kInt64));
    auto stream = at::cuda::getDefaultCUDAStream(device.index()).stream();
    ce = cudaFuncSetAttribute(
        MeasureZipservTileStagesKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    TORCH_CHECK(ce == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(ce));

    MeasureZipservTileStagesKernel<<<1, kThreads, shared_bytes, stream>>>(
        k_comp,
        v_comp,
        static_cast<int>(batch_idx),
        static_cast<int>(kv_head),
        static_cast<int>(block_row),
        static_cast<int>(actual_seqlen_q),
        static_cast<int>(logical_seqlen_k),
        static_cast<int>(block_rows),
        static_cast<int>(warmup),
        static_cast<int>(iters),
        out_sink.data_ptr<float>(),
        reinterpret_cast<unsigned long long*>(out_cycles.data_ptr<int64_t>()));
    ce = cudaGetLastError();
    TORCH_CHECK(ce == cudaSuccess, "MeasureZipservTileStagesKernel launch failed: ", cudaGetErrorString(ce));
    ce = cudaStreamSynchronize(stream);
    TORCH_CHECK(ce == cudaSuccess, "cudaStreamSynchronize failed: ", cudaGetErrorString(ce));

    auto out_cpu = out_cycles.to(torch::kCPU, torch::kFloat64);
    const double bracket_cycles = out_cpu[kTimingBracketIdx].item<double>();
    const double cycles_to_us = 1000.0 / static_cast<double>(prop.clockRate);
    for (int idx = 0; idx < kNumMeasuredStages; ++idx) {
        const double raw_cycles = out_cpu[idx].item<double>();
        const double pure_cycles = std::max(0.0, raw_cycles - bracket_cycles);
        out_cpu[idx].fill_(pure_cycles * cycles_to_us);
    }
    out_cpu[kTimingBracketIdx].fill_(bracket_cycles * cycles_to_us);
    return out_cpu;
}


}  // namespace zipserv_tile_bench

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "measure_zipserv_tile_stages",
        &zipserv_tile_bench::measure_zipserv_tile_stages,
        "Measure ZipServ fused tile load/decomp stages");
}
