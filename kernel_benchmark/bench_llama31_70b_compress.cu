#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "L_API.cuh"
#include "bench_manifest_utils.h"
#include "csv_writer.h"
#include "cuda_timer.h"
#include "./utils.h"

struct ProgramOptions {
    std::string manifest_path;
    int warmup = 10;
    int iters = 100;
    int device = 0;
    std::string out_csv = "llama31_70b_compress_results.csv";
    std::string layers_spec;
    std::string filter_spec;
    bool flush_l2 = false;
};

struct ManifestTensor {
    int layer = -1;
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    std::string path;
    size_t orig_bytes = 0;
};

struct BenchRow {
    int layer = -1;
    std::string name;
    std::string shape;
    std::string dtype;
    size_t orig_bytes = 0;
    size_t comp_bytes = 0;
    double compress_ms = 0.0;
    double decompress_ms = 0.0;
    double h2d_ms = 0.0;
    double h2d_decompress_ms = 0.0;
    double compress_gbps = 0.0;
    double decompress_gbps = 0.0;
    std::string status = "ok";
    std::string note;
};

struct CompressedBuffers {
    uint8_t* sign_mantissa = nullptr;
    __nv_bfloat16* compressed_full = nullptr;
    uint64_t* bitmap1 = nullptr;
    uint64_t* bitmap2 = nullptr;
    uint64_t* bitmap3 = nullptr;
    int* tile_offsets = nullptr;
    int* tile_offsets_median = nullptr;
    int* tile_offsets_global = nullptr;

    int num_global_tiles = 0;
    int max_high_freq_count = 0;
    int max_full_count = 0;
    int high_freq_count = 0;
    int full_count = 0;
    uint8_t start_exp = 0;
    size_t comp_bytes = 0;
};

struct DeviceCompressedBuffers {
    uint8_t* sign_mantissa = nullptr;
    __nv_bfloat16* compressed_full = nullptr;
    uint64_t* bitmap1 = nullptr;
    uint64_t* bitmap2 = nullptr;
    uint64_t* bitmap3 = nullptr;
    int* tile_offsets_median = nullptr;
    int* tile_offsets_global = nullptr;
    __nv_bfloat16* output = nullptr;
};

void PrintUsage() {
    std::cout
        << "Usage: ./bench_llama31_70b_compress --manifest <path> [options]\n"
        << "Options:\n"
        << "  --manifest <path>   Manifest JSONL path (required)\n"
        << "  --warmup <int>      Warmup iterations per tensor (default: 10)\n"
        << "  --iters <int>       Benchmark iterations per tensor (default: 100)\n"
        << "  --device <int>      CUDA device index (default: 0)\n"
        << "  --out <path>        Output CSV path (default: llama31_70b_compress_results.csv)\n"
        << "  --layers <spec>     Layer filter, e.g. 0-79 or 0,1,2\n"
        << "  --filter <spec>     Comma-separated tensor name substrings, e.g. q_proj,up_proj\n"
        << "  --flush-l2          Flush L2 cache before each decompression iteration\n"
        << "  --help              Show this help\n";
}

bool ParseLayerSpec(const std::string& spec, std::set<int>* layers, std::string* error) {
    return bench_common::ParseLayerSpec(spec, layers, error);
}

bool ParseOptions(int argc, char** argv, ProgramOptions* opts) {
    if (argc <= 1) {
        PrintUsage();
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            PrintUsage();
            return false;
        } else if (arg == "--manifest" && i + 1 < argc) {
            opts->manifest_path = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts->warmup = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            opts->iters = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            opts->device = std::atoi(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            opts->out_csv = argv[++i];
        } else if (arg == "--layers" && i + 1 < argc) {
            opts->layers_spec = argv[++i];
        } else if (arg == "--filter" && i + 1 < argc) {
            opts->filter_spec = argv[++i];
        } else if (arg == "--flush-l2") {
            opts->flush_l2 = true;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            PrintUsage();
            return false;
        }
    }

    if (opts->manifest_path.empty()) {
        std::cerr << "Error: --manifest is required.\n";
        PrintUsage();
        return false;
    }
    if (opts->warmup < 0) {
        std::cerr << "Error: --warmup must be >= 0.\n";
        return false;
    }
    if (opts->iters <= 0) {
        std::cerr << "Error: --iters must be > 0.\n";
        return false;
    }
    if (opts->device < 0) {
        std::cerr << "Error: --device must be >= 0.\n";
        return false;
    }
    return true;
}

bool LoadManifest(const std::string& path, std::vector<ManifestTensor>* tensors, std::string* error) {
    std::vector<bench_common::ManifestTensorInfo> parsed;
    if (!bench_common::LoadManifestJsonl(path, &parsed, error)) {
        return false;
    }
    tensors->clear();
    tensors->reserve(parsed.size());
    for (size_t i = 0; i < parsed.size(); ++i) {
        ManifestTensor tensor;
        tensor.layer = parsed[i].layer;
        tensor.name = parsed[i].name;
        tensor.shape = parsed[i].shape;
        tensor.dtype = parsed[i].dtype;
        tensor.path = parsed[i].path;
        tensor.orig_bytes = parsed[i].nbytes;
        tensors->push_back(tensor);
    }
    return true;
}

bool TensorPassesNameFilter(const std::string& name, const std::vector<std::string>& filters) {
    return bench_common::TensorNameMatchesAnyFilter(name, filters);
}

std::string ShapeToString(const std::vector<int>& shape) {
    return bench_common::ShapeToString(shape);
}

size_t FileSize(const std::string& path) {
    return bench_common::FileSize(path);
}

bool ReadBinaryFile(const std::string& path, std::vector<uint8_t>* data, std::string* error) {
    return bench_common::ReadBinaryFile(path, data, error);
}

bool ConvertRawToBF16(const ManifestTensor& tensor, const std::vector<uint8_t>& raw, std::vector<__nv_bfloat16>* out, std::string* error) {
    if (tensor.shape.size() != 2) {
        *error = "tensor is not 2D";
        return false;
    }
    const int rows = tensor.shape[0];
    const int cols = tensor.shape[1];
    if (rows <= 0 || cols <= 0) {
        *error = "shape has non-positive dimension";
        return false;
    }

    const size_t numel = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t expected_bytes = numel * sizeof(uint16_t);
    if (raw.size() != expected_bytes) {
        std::ostringstream oss;
        oss << "file bytes mismatch: expected " << expected_bytes << ", got " << raw.size();
        *error = oss.str();
        return false;
    }

    out->resize(numel);
    if (tensor.dtype == "bf16") {
        std::memcpy(out->data(), raw.data(), expected_bytes);
        return true;
    }
    *error = "unsupported dtype: " + tensor.dtype + " (expected bf16)";
    return false;
}

void FreeCompressedBuffers(CompressedBuffers* buffers) {
    free(buffers->sign_mantissa);
    free(buffers->compressed_full);
    free(buffers->bitmap1);
    free(buffers->bitmap2);
    free(buffers->bitmap3);
    free(buffers->tile_offsets);
    free(buffers->tile_offsets_median);
    free(buffers->tile_offsets_global);
    buffers->sign_mantissa = nullptr;
    buffers->compressed_full = nullptr;
    buffers->bitmap1 = nullptr;
    buffers->bitmap2 = nullptr;
    buffers->bitmap3 = nullptr;
    buffers->tile_offsets = nullptr;
    buffers->tile_offsets_median = nullptr;
    buffers->tile_offsets_global = nullptr;
    buffers->num_global_tiles = 0;
    buffers->max_high_freq_count = 0;
    buffers->max_full_count = 0;
    buffers->high_freq_count = 0;
    buffers->full_count = 0;
    buffers->start_exp = 0;
    buffers->comp_bytes = 0;
}

bool RunCompressionOnce(
    __nv_bfloat16* matrix,
    int rows,
    int cols,
    const int* top_exponents,
    CompressedBuffers* out,
    std::string* error) {
    FreeCompressedBuffers(out);

    const int tile_m = 8;
    const int tile_m_median = 16;
    const int tile_m_global = 64;
    const int tile_k = 8;
    const int tile_k_median = 64;
    const int tile_k_global = 64;

    int max_high_freq_count = 0;
    int max_full_count = 0;
    int num_global_tiles = InitBF16MatrixTripleBitmap(
        matrix,
        rows,
        cols,
        tile_m,
        tile_m_median,
        tile_m_global,
        tile_k,
        tile_k_median,
        tile_k_global,
        top_exponents,
        &out->sign_mantissa,
        &out->compressed_full,
        &out->bitmap1,
        &out->bitmap2,
        &out->bitmap3,
        &out->tile_offsets,
        &out->tile_offsets_median,
        &out->tile_offsets_global,
        max_high_freq_count,
        max_full_count);
    if (num_global_tiles <= 0) {
        *error = "InitBF16MatrixTripleBitmap failed";
        FreeCompressedBuffers(out);
        return false;
    }
    out->num_global_tiles = num_global_tiles;
    out->max_high_freq_count = max_high_freq_count;
    out->max_full_count = max_full_count;
    out->start_exp = static_cast<uint8_t>(top_exponents[0] - 1);
    out->high_freq_count = out->tile_offsets_global[num_global_tiles * 2];
    out->full_count = out->tile_offsets_global[num_global_tiles * 2 + 1];

    const int num_tiles = (rows / tile_m) * (cols / tile_k);
    const int num_median_tiles = (rows / tile_m_median) * (cols / tile_k_median);
    out->comp_bytes = static_cast<size_t>(out->high_freq_count) * sizeof(uint8_t) +
                      static_cast<size_t>(out->full_count) * sizeof(__nv_bfloat16) +
                      static_cast<size_t>(num_tiles) * sizeof(uint64_t) * 3 +
                      static_cast<size_t>(num_median_tiles) * 2 * sizeof(int) +
                      static_cast<size_t>(num_global_tiles + 1) * 2 * sizeof(int);
    return true;
}

void FreeDeviceCompressedBuffers(DeviceCompressedBuffers* buffers) {
    cudaFree(buffers->sign_mantissa);
    cudaFree(buffers->compressed_full);
    cudaFree(buffers->bitmap1);
    cudaFree(buffers->bitmap2);
    cudaFree(buffers->bitmap3);
    cudaFree(buffers->tile_offsets_median);
    cudaFree(buffers->tile_offsets_global);
    cudaFree(buffers->output);
    buffers->sign_mantissa = nullptr;
    buffers->compressed_full = nullptr;
    buffers->bitmap1 = nullptr;
    buffers->bitmap2 = nullptr;
    buffers->bitmap3 = nullptr;
    buffers->tile_offsets_median = nullptr;
    buffers->tile_offsets_global = nullptr;
    buffers->output = nullptr;
}

bool PrepareDeviceCompressedBuffers(
    const CompressedBuffers& host,
    int rows,
    int cols,
    DeviceCompressedBuffers* device,
    std::string* error) {
    FreeDeviceCompressedBuffers(device);

    const int num_tiles = (rows / 8) * (cols / 8);
    const int num_median_tiles = (rows / 16) * (cols / 64);
    const size_t out_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(__nv_bfloat16);

    cudaError_t cuda_error = cudaSuccess;
    cuda_error = cudaMalloc(reinterpret_cast<void**>(&device->sign_mantissa), host.high_freq_count * sizeof(uint8_t));
    if (cuda_error != cudaSuccess) {
        *error = "cudaMalloc sign_mantissa failed";
        return false;
    }
    cuda_error = cudaMalloc(reinterpret_cast<void**>(&device->compressed_full), host.full_count * sizeof(__nv_bfloat16));
    if (cuda_error != cudaSuccess) {
        *error = "cudaMalloc compressed_full failed";
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    cuda_error = cudaMalloc(reinterpret_cast<void**>(&device->bitmap1), num_tiles * sizeof(uint64_t));
    if (cuda_error != cudaSuccess) {
        *error = "cudaMalloc bitmap1 failed";
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    cuda_error = cudaMalloc(reinterpret_cast<void**>(&device->bitmap2), num_tiles * sizeof(uint64_t));
    if (cuda_error != cudaSuccess) {
        *error = "cudaMalloc bitmap2 failed";
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    cuda_error = cudaMalloc(reinterpret_cast<void**>(&device->bitmap3), num_tiles * sizeof(uint64_t));
    if (cuda_error != cudaSuccess) {
        *error = "cudaMalloc bitmap3 failed";
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    cuda_error = cudaMalloc(reinterpret_cast<void**>(&device->tile_offsets_median), num_median_tiles * 2 * sizeof(int));
    if (cuda_error != cudaSuccess) {
        *error = "cudaMalloc tile_offsets_median failed";
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    cuda_error = cudaMalloc(
        reinterpret_cast<void**>(&device->tile_offsets_global),
        (host.num_global_tiles + 1) * 2 * sizeof(int));
    if (cuda_error != cudaSuccess) {
        *error = "cudaMalloc tile_offsets_global failed";
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    cuda_error = cudaMalloc(reinterpret_cast<void**>(&device->output), out_bytes);
    if (cuda_error != cudaSuccess) {
        *error = "cudaMalloc output failed";
        FreeDeviceCompressedBuffers(device);
        return false;
    }

    cudaMemcpy(device->sign_mantissa, host.sign_mantissa, host.high_freq_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device->compressed_full, host.compressed_full, host.full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(device->bitmap1, host.bitmap1, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device->bitmap2, host.bitmap2, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device->bitmap3, host.bitmap3, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device->tile_offsets_median, host.tile_offsets_median, num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(
        device->tile_offsets_global,
        host.tile_offsets_global,
        (host.num_global_tiles + 1) * 2 * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemset(device->output, 0, out_bytes);
    checkLastCudaError(__LINE__);

    return true;
}

void FlushL2CacheIfEnabled(bool enabled);

void flush_l2_cache() {
    static void* d_flush_buffer = nullptr;
    static size_t flush_buffer_size = 0;
    static bool initialized = false;

    if (!initialized) {
        int current_device = 0;
        int l2_cache_size = 0;
        cudaGetDevice(&current_device);
        cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, current_device);
        flush_buffer_size = static_cast<size_t>(l2_cache_size) * 2;
        cudaMalloc(&d_flush_buffer, flush_buffer_size);
        if (d_flush_buffer == nullptr) {
            std::cerr << "Failed to allocate L2 flush buffer.\n";
            std::exit(EXIT_FAILURE);
        }
        initialized = true;
    }

    cudaMemsetAsync(d_flush_buffer, 0, flush_buffer_size);
    cudaDeviceSynchronize();
}

void FlushL2CacheIfEnabled(bool enabled) {
    if (enabled) {
        flush_l2_cache();
    }
}

bool CopyCompressedToDeviceAsync(
    const CompressedBuffers& host,
    int rows,
    int cols,
    const DeviceCompressedBuffers& device,
    cudaStream_t stream,
    std::string* error) {
    const int num_tiles = (rows / 8) * (cols / 8);
    const int num_median_tiles = (rows / 16) * (cols / 64);

    cudaError_t copy_error = cudaSuccess;
    copy_error = cudaMemcpyAsync(
        device.sign_mantissa,
        host.sign_mantissa,
        host.high_freq_count * sizeof(uint8_t),
        cudaMemcpyHostToDevice,
        stream);
    if (copy_error != cudaSuccess) {
        *error = std::string("H2D sign_mantissa failed: ") + cudaGetErrorString(copy_error);
        return false;
    }

    copy_error = cudaMemcpyAsync(
        device.compressed_full,
        host.compressed_full,
        host.full_count * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice,
        stream);
    if (copy_error != cudaSuccess) {
        *error = std::string("H2D compressed_full failed: ") + cudaGetErrorString(copy_error);
        return false;
    }

    copy_error = cudaMemcpyAsync(
        device.bitmap1,
        host.bitmap1,
        num_tiles * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        stream);
    if (copy_error != cudaSuccess) {
        *error = std::string("H2D bitmap1 failed: ") + cudaGetErrorString(copy_error);
        return false;
    }

    copy_error = cudaMemcpyAsync(
        device.bitmap2,
        host.bitmap2,
        num_tiles * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        stream);
    if (copy_error != cudaSuccess) {
        *error = std::string("H2D bitmap2 failed: ") + cudaGetErrorString(copy_error);
        return false;
    }

    copy_error = cudaMemcpyAsync(
        device.bitmap3,
        host.bitmap3,
        num_tiles * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        stream);
    if (copy_error != cudaSuccess) {
        *error = std::string("H2D bitmap3 failed: ") + cudaGetErrorString(copy_error);
        return false;
    }

    copy_error = cudaMemcpyAsync(
        device.tile_offsets_median,
        host.tile_offsets_median,
        num_median_tiles * 2 * sizeof(int),
        cudaMemcpyHostToDevice,
        stream);
    if (copy_error != cudaSuccess) {
        *error = std::string("H2D tile_offsets_median failed: ") + cudaGetErrorString(copy_error);
        return false;
    }

    copy_error = cudaMemcpyAsync(
        device.tile_offsets_global,
        host.tile_offsets_global,
        (host.num_global_tiles + 1) * 2 * sizeof(int),
        cudaMemcpyHostToDevice,
        stream);
    if (copy_error != cudaSuccess) {
        *error = std::string("H2D tile_offsets_global failed: ") + cudaGetErrorString(copy_error);
        return false;
    }

    return true;
}

bool BenchmarkDecompress(
    const CompressedBuffers& host,
    int rows,
    int cols,
    const ProgramOptions& options,
    double* h2d_ms,
    double* decompress_ms,
    double* h2d_decompress_ms,
    std::string* error) {
    DeviceCompressedBuffers device;
    if (!PrepareDeviceCompressedBuffers(host, rows, cols, &device, error)) {
        return false;
    }

    for (int i = 0; i < options.warmup; ++i) {
        FlushL2CacheIfEnabled(options.flush_l2);
        cudaError_t decompress_error = BF16TripleBitmap_Decompress_API(
            0,
            device.sign_mantissa,
            device.compressed_full,
            device.bitmap1,
            device.bitmap2,
            device.bitmap3,
            device.tile_offsets_median,
            device.tile_offsets_global,
            host.max_high_freq_count,
            host.max_full_count,
            host.start_exp,
            device.output,
            rows,
            cols);
        if (decompress_error != cudaSuccess) {
            *error = std::string("BF16TripleBitmap_Decompress_API warmup failed: ") +
                     cudaGetErrorString(decompress_error);
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
    }
    cudaDeviceSynchronize();
    checkLastCudaError(__LINE__);

    bench_common::CudaEventTimer timer;

    float total_ms = 0.0f;
    for (int i = 0; i < options.iters; ++i) {
        FlushL2CacheIfEnabled(options.flush_l2);
        if (!timer.RecordStart(0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        cudaError_t decompress_error = BF16TripleBitmap_Decompress_API(
            0,
            device.sign_mantissa,
            device.compressed_full,
            device.bitmap1,
            device.bitmap2,
            device.bitmap3,
            device.tile_offsets_median,
            device.tile_offsets_global,
            host.max_high_freq_count,
            host.max_full_count,
            host.start_exp,
            device.output,
            rows,
            cols);
        if (decompress_error != cudaSuccess) {
            *error = std::string("BF16TripleBitmap_Decompress_API benchmark failed: ") +
                     cudaGetErrorString(decompress_error);
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        if (!timer.RecordStop(0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        if (!timer.SyncStop(error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        float iter_ms = 0.0f;
        if (!timer.ElapsedMs(&iter_ms, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        total_ms += iter_ms;
    }
    *decompress_ms = static_cast<double>(total_ms) / static_cast<double>(options.iters);

    for (int i = 0; i < options.warmup; ++i) {
        if (!CopyCompressedToDeviceAsync(host, rows, cols, device, 0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
    }
    cudaDeviceSynchronize();
    checkLastCudaError(__LINE__);

    float total_h2d_ms = 0.0f;
    for (int i = 0; i < options.iters; ++i) {
        if (!timer.RecordStart(0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        if (!CopyCompressedToDeviceAsync(host, rows, cols, device, 0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        if (!timer.RecordStop(0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        if (!timer.SyncStop(error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        float iter_h2d_ms = 0.0f;
        if (!timer.ElapsedMs(&iter_h2d_ms, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        total_h2d_ms += iter_h2d_ms;
    }
    *h2d_ms = static_cast<double>(total_h2d_ms) / static_cast<double>(options.iters);

    for (int i = 0; i < options.warmup; ++i) {
        FlushL2CacheIfEnabled(options.flush_l2);
        if (!CopyCompressedToDeviceAsync(host, rows, cols, device, 0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        cudaError_t decompress_error = BF16TripleBitmap_Decompress_API(
            0,
            device.sign_mantissa,
            device.compressed_full,
            device.bitmap1,
            device.bitmap2,
            device.bitmap3,
            device.tile_offsets_median,
            device.tile_offsets_global,
            host.max_high_freq_count,
            host.max_full_count,
            host.start_exp,
            device.output,
            rows,
            cols);
        if (decompress_error != cudaSuccess) {
            *error = std::string("BF16TripleBitmap_Decompress_API H2D+decomp warmup failed: ") +
                     cudaGetErrorString(decompress_error);
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
    }
    cudaDeviceSynchronize();
    checkLastCudaError(__LINE__);

    float total_h2d_decompress_ms = 0.0f;
    for (int i = 0; i < options.iters; ++i) {
        FlushL2CacheIfEnabled(options.flush_l2);
        if (!timer.RecordStart(0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        if (!CopyCompressedToDeviceAsync(host, rows, cols, device, 0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        cudaError_t decompress_error = BF16TripleBitmap_Decompress_API(
            0,
            device.sign_mantissa,
            device.compressed_full,
            device.bitmap1,
            device.bitmap2,
            device.bitmap3,
            device.tile_offsets_median,
            device.tile_offsets_global,
            host.max_high_freq_count,
            host.max_full_count,
            host.start_exp,
            device.output,
            rows,
            cols);
        if (decompress_error != cudaSuccess) {
            *error = std::string("BF16TripleBitmap_Decompress_API H2D+decomp benchmark failed: ") +
                     cudaGetErrorString(decompress_error);
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        if (!timer.RecordStop(0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        if (!timer.SyncStop(error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        float iter_h2d_decompress_ms = 0.0f;
        if (!timer.ElapsedMs(&iter_h2d_decompress_ms, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        total_h2d_decompress_ms += iter_h2d_decompress_ms;
    }
    *h2d_decompress_ms = static_cast<double>(total_h2d_decompress_ms) / static_cast<double>(options.iters);

    FreeDeviceCompressedBuffers(&device);
    return true;
}

double BytesPerSecondToGBps(size_t bytes, double milliseconds) {
    if (milliseconds <= 0.0) {
        return 0.0;
    }
    const double seconds = milliseconds / 1000.0;
    return static_cast<double>(bytes) / seconds / 1e9;
}

double CompressionRatioX(size_t orig_bytes, size_t comp_bytes) {
    if (orig_bytes == 0 || comp_bytes == 0) {
        return 0.0;
    }
    return static_cast<double>(orig_bytes) / static_cast<double>(comp_bytes);
}

std::string FormatHumanBytes(size_t bytes) {
    static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream oss;
    if (unit == 0) {
        oss << static_cast<size_t>(value) << kUnits[unit];
        return oss.str();
    }
    const int precision = (value >= 100.0) ? 1 : ((value >= 10.0) ? 2 : 3);
    oss << std::fixed << std::setprecision(precision) << value << kUnits[unit];
    return oss.str();
}

std::string FormatMilliseconds(double milliseconds) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << milliseconds << "ms";
    return oss.str();
}

std::string FormatHumanThroughput(double gbps) {
    static const char* kUnits[] = {"B/s", "KB/s", "MB/s", "GB/s", "TB/s"};
    double value = gbps * 1e9;
    int unit = 0;
    while (value >= 1000.0 && unit < 4) {
        value /= 1000.0;
        ++unit;
    }
    const int precision = (value >= 100.0) ? 1 : ((value >= 10.0) ? 2 : 3);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value << kUnits[unit];
    return oss.str();
}

std::string TruncateString(const std::string& text, size_t max_len) {
    if (text.size() <= max_len) {
        return text;
    }
    if (max_len <= 3) {
        return text.substr(0, max_len);
    }
    return text.substr(0, max_len - 3) + "...";
}

std::string DisplayStatus(const std::string& status) {
    if (status == "ok") {
        return "OK";
    }
    if (status == "skipped") {
        return "SKIP";
    }
    if (status == "failed") {
        return "FAIL";
    }
    return status;
}

void PrintTableRule() {
    std::cout << std::string(206, '-') << "\n";
}

void PrintTableHeader() {
    PrintTableRule();
    std::cout << std::left
              << std::setw(7) << "Status"
              << std::setw(7) << "Layer"
              << std::setw(44) << "Tensor"
              << std::setw(15) << "Shape"
              << std::setw(7) << "DType"
              << std::setw(12) << "Orig"
              << std::setw(12) << "Comp"
              << std::setw(8) << "Ratio"
              << std::setw(12) << "Comp(ms)"
              << std::setw(10) << "H2D(ms)"
              << std::setw(12) << "Decomp(ms)"
              << std::setw(16) << "H2D+Decomp(ms)"
              << std::setw(14) << "CompBW"
              << std::setw(14) << "DecompBW"
              << "Note\n";
    PrintTableRule();
}

std::string FormatRatioX(size_t orig_bytes, size_t comp_bytes) {
    if (orig_bytes == 0 || comp_bytes == 0) {
        return "-";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << CompressionRatioX(orig_bytes, comp_bytes) << "x";
    return oss.str();
}

void PrintTableRow(const BenchRow& row) {
    const bool has_metrics = (row.status == "ok");
    const std::string comp_size = (row.comp_bytes > 0) ? FormatHumanBytes(row.comp_bytes) : "-";
    const std::string ratio_text = (row.comp_bytes > 0) ? FormatRatioX(row.orig_bytes, row.comp_bytes) : "-";
    const std::string comp_time = has_metrics ? FormatMilliseconds(row.compress_ms) : "-";
    const std::string h2d_time = has_metrics ? FormatMilliseconds(row.h2d_ms) : "-";
    const std::string decomp_time = has_metrics ? FormatMilliseconds(row.decompress_ms) : "-";
    const std::string h2d_decomp_time = has_metrics ? FormatMilliseconds(row.h2d_decompress_ms) : "-";
    const std::string comp_bw = has_metrics ? FormatHumanThroughput(row.compress_gbps) : "-";
    const std::string decomp_bw = has_metrics ? FormatHumanThroughput(row.decompress_gbps) : "-";

    std::cout << std::left
              << std::setw(7) << DisplayStatus(row.status)
              << std::setw(7) << row.layer
              << std::setw(44) << TruncateString(row.name, 43)
              << std::setw(15) << row.shape
              << std::setw(7) << row.dtype
              << std::setw(12) << FormatHumanBytes(row.orig_bytes)
              << std::setw(12) << comp_size
              << std::setw(8) << ratio_text
              << std::setw(12) << comp_time
              << std::setw(10) << h2d_time
              << std::setw(12) << decomp_time
              << std::setw(16) << h2d_decomp_time
              << std::setw(14) << comp_bw
              << std::setw(14) << decomp_bw
              << TruncateString(row.note, 32)
              << "\n";
}

std::string EscapeCSV(const std::string& field) {
    return bench_common::EscapeCSV(field);
}

void WriteCSVHeader(std::ofstream* ofs) {
    (*ofs) << "Layer,Tensor,Shape,DType,Orignal_Size,Compressed_Size,Ratio,Comp(ms),H2D(ms),Decomp(ms),H2D+Decomp(ms),CompBW,DecompBW\n";
}

void WriteCSVRow(std::ofstream* ofs, const BenchRow& row) {
    const bool has_metrics = (row.status == "ok");
    const std::string comp_size = (row.comp_bytes > 0) ? FormatHumanBytes(row.comp_bytes) : "-";
    const std::string ratio_text = (row.comp_bytes > 0) ? FormatRatioX(row.orig_bytes, row.comp_bytes) : "-";
    const std::string comp_time = has_metrics ? FormatMilliseconds(row.compress_ms) : "-";
    const std::string h2d_time = has_metrics ? FormatMilliseconds(row.h2d_ms) : "-";
    const std::string decomp_time = has_metrics ? FormatMilliseconds(row.decompress_ms) : "-";
    const std::string h2d_decomp_time = has_metrics ? FormatMilliseconds(row.h2d_decompress_ms) : "-";
    const std::string comp_bw = has_metrics ? FormatHumanThroughput(row.compress_gbps) : "-";
    const std::string decomp_bw = has_metrics ? FormatHumanThroughput(row.decompress_gbps) : "-";

    (*ofs) << row.layer << ","
           << EscapeCSV(row.name) << ","
           << EscapeCSV(row.shape) << ","
           << EscapeCSV(row.dtype) << ","
           << EscapeCSV(FormatHumanBytes(row.orig_bytes)) << ","
           << EscapeCSV(comp_size) << ","
           << EscapeCSV(ratio_text) << ","
           << EscapeCSV(comp_time) << ","
           << EscapeCSV(h2d_time) << ","
           << EscapeCSV(decomp_time) << ","
           << EscapeCSV(h2d_decomp_time) << ","
           << EscapeCSV(comp_bw) << ","
           << EscapeCSV(decomp_bw) << "\n";
}

BenchRow ProcessTensor(const ManifestTensor& tensor, const ProgramOptions& options) {
    BenchRow row;
    row.layer = tensor.layer;
    row.name = tensor.name;
    row.shape = ShapeToString(tensor.shape);
    row.dtype = tensor.dtype;
    row.orig_bytes = tensor.orig_bytes;
    row.status = "ok";
    if (tensor.dtype != "bf16") {
        row.status = "failed";
        row.note = "manifest_dtype_must_be_bf16";
        return row;
    }

    if (tensor.shape.size() != 2) {
        row.status = "skipped";
        row.note = "non_2d_shape";
        return row;
    }
    const int rows = tensor.shape[0];
    const int cols = tensor.shape[1];
    if (rows <= 0 || cols <= 0) {
        row.status = "skipped";
        row.note = "invalid_shape_dimension";
        return row;
    }
    if (rows % 64 != 0 || cols % 64 != 0) {
        row.status = "skipped";
        row.note = "shape_not_multiple_of_64";
        return row;
    }

    const size_t file_nbytes = FileSize(tensor.path);
    if (file_nbytes == 0) {
        row.status = "skipped";
        row.note = "file_missing_or_empty";
        return row;
    }
    if (file_nbytes != tensor.orig_bytes) {
        std::ostringstream note;
        note << "manifest_nbytes_mismatch(file=" << file_nbytes
             << ",manifest=" << tensor.orig_bytes << ")";
        row.note = note.str();
        row.orig_bytes = file_nbytes;
    }

    std::vector<uint8_t> raw_data;
    std::string io_error;
    if (!ReadBinaryFile(tensor.path, &raw_data, &io_error)) {
        row.status = "skipped";
        row.note = io_error;
        return row;
    }

    std::vector<__nv_bfloat16> host_bf16;
    std::string convert_error;
    if (!ConvertRawToBF16(tensor, raw_data, &host_bf16, &convert_error)) {
        row.status = "skipped";
        row.note = convert_error;
        return row;
    }

    int top_exponents[7] = {0};
    analyzeExponentDistribution_BF16(host_bf16.data(), rows, cols, top_exponents);

    for (int i = 0; i < options.warmup; ++i) {
        CompressedBuffers warmup_buffers;
        std::string compression_error;
        if (!RunCompressionOnce(host_bf16.data(), rows, cols, top_exponents, &warmup_buffers, &compression_error)) {
            row.status = "failed";
            row.note = compression_error;
            return row;
        }
        FreeCompressedBuffers(&warmup_buffers);
    }

    CompressedBuffers final_buffers;
    double total_compress_ms = 0.0;
    for (int i = 0; i < options.iters; ++i) {
        CompressedBuffers iteration_buffers;
        std::string compression_error;
        std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
        bool ok = RunCompressionOnce(host_bf16.data(), rows, cols, top_exponents, &iteration_buffers, &compression_error);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        if (!ok) {
            row.status = "failed";
            row.note = compression_error;
            FreeCompressedBuffers(&iteration_buffers);
            FreeCompressedBuffers(&final_buffers);
            return row;
        }

        std::chrono::duration<double, std::milli> elapsed_ms = end - begin;
        total_compress_ms += elapsed_ms.count();

        if (i == options.iters - 1) {
            final_buffers = iteration_buffers;
        } else {
            FreeCompressedBuffers(&iteration_buffers);
        }
    }

    row.comp_bytes = final_buffers.comp_bytes;
    row.compress_ms = total_compress_ms / static_cast<double>(options.iters);
    row.compress_gbps = BytesPerSecondToGBps(row.orig_bytes, row.compress_ms);

    std::string decompress_error;
    double h2d_ms = 0.0;
    double decompress_ms = 0.0;
    double h2d_decompress_ms = 0.0;
    if (!BenchmarkDecompress(final_buffers, rows, cols, options, &h2d_ms, &decompress_ms, &h2d_decompress_ms, &decompress_error)) {
        row.status = "failed";
        row.note = decompress_error;
        FreeCompressedBuffers(&final_buffers);
        return row;
    }
    row.h2d_ms = h2d_ms;
    row.decompress_ms = decompress_ms;
    row.h2d_decompress_ms = h2d_decompress_ms;
    row.decompress_gbps = BytesPerSecondToGBps(row.orig_bytes, row.decompress_ms);

    FreeCompressedBuffers(&final_buffers);
    return row;
}

int main(int argc, char** argv) {
    ProgramOptions options;
    if (!ParseOptions(argc, argv, &options)) {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            return 0;
        }
        return 1;
    }

    std::set<int> allowed_layers;
    std::string layer_error;
    if (!ParseLayerSpec(options.layers_spec, &allowed_layers, &layer_error)) {
        std::cerr << "Layer parse error: " << layer_error << "\n";
        return 1;
    }
    const std::vector<std::string> name_filters = bench_common::SplitByComma(options.filter_spec);

    cudaError_t cuda_error = cudaSetDevice(options.device);
    if (cuda_error != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cuda_error) << "\n";
        return 1;
    }

    std::vector<ManifestTensor> all_tensors;
    std::string manifest_error;
    if (!LoadManifest(options.manifest_path, &all_tensors, &manifest_error)) {
        std::cerr << manifest_error << "\n";
        return 1;
    }

    std::vector<ManifestTensor> selected_tensors;
    selected_tensors.reserve(all_tensors.size());
    for (size_t i = 0; i < all_tensors.size(); ++i) {
        const ManifestTensor& t = all_tensors[i];
        if (!allowed_layers.empty() && allowed_layers.find(t.layer) == allowed_layers.end()) {
            continue;
        }
        if (!TensorPassesNameFilter(t.name, name_filters)) {
            continue;
        }
        selected_tensors.push_back(t);
    }

    std::sort(
        selected_tensors.begin(),
        selected_tensors.end(),
        [](const ManifestTensor& a, const ManifestTensor& b) {
            if (a.layer != b.layer) {
                return a.layer < b.layer;
            }
            return a.name < b.name;
        });

    if (selected_tensors.empty()) {
        std::cerr << "No tensors selected after applying --layers/--filter.\n";
        return 1;
    }

    std::ofstream ofs(options.out_csv.c_str(), std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output CSV: " << options.out_csv << "\n";
        return 1;
    }
    WriteCSVHeader(&ofs);

    std::cout << "===== Llama 3.1 70B Codec Benchmark =====\n";
    std::cout << "manifest=" << options.manifest_path << "\n";
    std::cout << "selected_tensors=" << selected_tensors.size() << "\n";
    std::cout << "warmup=" << options.warmup << ", iters=" << options.iters << "\n";
    std::cout << "device=" << options.device << ", flush_l2=" << (options.flush_l2 ? "true" : "false") << "\n";
    std::cout << "csv=" << options.out_csv << "\n\n";
    std::cout << "Units: size=KiB/MiB/GiB, time=auto(ns/us/ms/s), bandwidth=auto(B/s..TB/s)\n";
    PrintTableHeader();

    size_t ok_count = 0;
    size_t skipped_count = 0;
    size_t failed_count = 0;

    for (size_t idx = 0; idx < selected_tensors.size(); ++idx) {
        const ManifestTensor& tensor = selected_tensors[idx];
        BenchRow row = ProcessTensor(tensor, options);
        WriteCSVRow(&ofs, row);
        PrintTableRow(row);

        if (row.status == "ok") {
            ++ok_count;
        } else if (row.status == "skipped") {
            ++skipped_count;
        } else {
            ++failed_count;
        }
    }

    ofs.close();
    PrintTableRule();

    std::cout << "\n===== Summary =====\n";
    std::cout << "ok=" << ok_count << ", skipped=" << skipped_count << ", failed=" << failed_count << "\n";
    std::cout << "CSV written to: " << options.out_csv << "\n";
    return 0;
}
