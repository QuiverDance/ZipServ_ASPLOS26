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

#define CUDA_CHECK(expr) TORCH_CHECK((expr) == cudaSuccess, #expr " failed")

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
