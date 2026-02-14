#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
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
    int layer = 0;
    std::string ops_spec = "qkv,o,gateup,down";
    std::string tokens_spec = "1,2,4,8,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024";
    int warmup = 10;
    int iters = 100;
    std::string out_csv = "results_zipserv_gemm.csv";
    int device = 0;
    int seed = 1234;
    bool emit_aggregate = true;
    bool weight_offload = false;
};

struct ManifestTensor {
    int layer = -1;
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    std::string path;
    size_t nbytes = 0;
};

struct WeightSpec {
    std::string group;       // qkv/o/gateup/down
    std::string short_name;  // q_proj/k_proj/...
    ManifestTensor tensor;
    int rows = 0; // N_out
    int cols = 0; // K_in
    std::vector<__nv_bfloat16> host_weight;
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
};

struct GemmCsvRow {
    std::string kind; // tensor/aggregate
    std::string group;
    std::string name; // short tensor name used in terminal table
    int layer = 0;
    int tokens = 0;
    int M = 0;
    int N = 0;
    int K = 0;
    std::string weight_name;
    std::string weight_shape;
    std::string dtype;
    size_t orig_bytes = 0;
    size_t comp_bytes = 0;
    double baseline_ms = 0.0;
    double zipserv_ms = 0.0;
    double zip_over_base = 0.0;
    double baseline_gbps = 0.0;
    double zipserv_gbps = 0.0;
    std::string x_shape;
    std::string w_shape;
    std::string y_shape;
    std::string baseline_cfg;
    std::string zipserv_cfg;
};

bool LoadManifest(const std::string& path, std::vector<ManifestTensor>* tensors, std::string* error) {
    std::vector<bench_common::ManifestTensorInfo> parsed;
    if (!bench_common::LoadManifestJsonl(path, &parsed, error)) {
        return false;
    }
    tensors->clear();
    tensors->reserve(parsed.size());
    for (size_t i = 0; i < parsed.size(); ++i) {
        ManifestTensor t;
        t.layer = parsed[i].layer;
        t.name = parsed[i].name;
        t.shape = parsed[i].shape;
        t.dtype = parsed[i].dtype;
        t.path = parsed[i].path;
        t.nbytes = parsed[i].nbytes;
        tensors->push_back(t);
    }
    return true;
}

bool ParseIntList(const std::string& spec, std::vector<int>* out, std::string* error) {
    return bench_common::ParsePositiveIntList(spec, out, error);
}

void PrintUsage() {
    std::cout
        << "Usage: ./bench_zipserv_gemm_sweep --manifest <path> [options]\n"
        << "Options:\n"
        << "  --manifest <path>     Manifest JSONL path (required)\n"
        << "  --layer <int>         Layer index (default: 0)\n"
        << "  --ops <spec>          qkv,o,gateup,down (default: qkv,o,gateup,down)\n"
        << "  --tokens <list>       Tokens sweep list (default preset)\n"
        << "  --warmup <int>        Warmup iterations (default: 10)\n"
        << "  --iters <int>         Benchmark iterations (default: 100)\n"
        << "  --out <path>          Output CSV path (default: results_zipserv_gemm.csv)\n"
        << "  --device <int>        CUDA device index (default: 0)\n"
        << "  --seed <int>          Random seed (default: 1234)\n"
        << "  --emit_aggregate <0|1> Emit qkv_sum and gateup_sum rows (default: 1)\n"
        << "  --weight_offload <0|1> Include weight H2D transfer in timing (default: 0)\n"
        << "  --help                Show this help\n";
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
        } else if (arg == "--layer" && i + 1 < argc) {
            opts->layer = std::atoi(argv[++i]);
        } else if (arg == "--ops" && i + 1 < argc) {
            opts->ops_spec = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            opts->tokens_spec = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts->warmup = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            opts->iters = std::atoi(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            opts->out_csv = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            opts->device = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            opts->seed = std::atoi(argv[++i]);
        } else if (arg == "--emit_aggregate" && i + 1 < argc) {
            opts->emit_aggregate = (std::atoi(argv[++i]) != 0);
        } else if (arg == "--weight_offload" && i + 1 < argc) {
            opts->weight_offload = (std::atoi(argv[++i]) != 0);
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            PrintUsage();
            return false;
        }
    }

    if (opts->manifest_path.empty()) {
        std::cerr << "Error: --manifest is required.\n";
        return false;
    }
    if (opts->layer < 0) {
        std::cerr << "Error: --layer must be >= 0.\n";
        return false;
    }
    if (opts->warmup < 0 || opts->iters <= 0) {
        std::cerr << "Error: warmup must be >=0 and iters >0.\n";
        return false;
    }
    return true;
}

std::string ExtractShortName(const std::string& full_name) {
    const char* suffixes[] = {
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight"
    };
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i) {
        std::string suffix = std::string(".") + suffixes[i];
        if (bench_common::EndsWith(full_name, suffix)) {
            return std::string(suffixes[i]).substr(0, std::string(suffixes[i]).find(".weight"));
        }
    }
    return full_name;
}

bool ConvertRawToBF16(const ManifestTensor& tensor, std::vector<__nv_bfloat16>* out, std::string* error) {
    if (tensor.dtype != "bf16") {
        *error = "dtype must be bf16: " + tensor.name;
        return false;
    }
    if (tensor.shape.size() != 2) {
        *error = "shape must be 2D: " + tensor.name;
        return false;
    }
    int rows = tensor.shape[0];
    int cols = tensor.shape[1];
    if (rows <= 0 || cols <= 0) {
        *error = "invalid shape: " + tensor.name;
        return false;
    }
    std::vector<uint8_t> raw;
    if (!bench_common::ReadBinaryFile(tensor.path, &raw, error)) {
        return false;
    }

    size_t expected = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(uint16_t);
    if (raw.size() != expected) {
        std::ostringstream oss;
        oss << "file bytes mismatch for " << tensor.name << ": expected " << expected << ", got " << raw.size();
        *error = oss.str();
        return false;
    }

    out->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    std::memcpy(out->data(), raw.data(), expected);
    return true;
}

void FreeCompressedBuffers(CompressedBuffers* b) {
    free(b->sign_mantissa);
    free(b->compressed_full);
    free(b->bitmap1);
    free(b->bitmap2);
    free(b->bitmap3);
    free(b->tile_offsets);
    free(b->tile_offsets_median);
    free(b->tile_offsets_global);
    b->sign_mantissa = nullptr;
    b->compressed_full = nullptr;
    b->bitmap1 = nullptr;
    b->bitmap2 = nullptr;
    b->bitmap3 = nullptr;
    b->tile_offsets = nullptr;
    b->tile_offsets_median = nullptr;
    b->tile_offsets_global = nullptr;
    b->num_global_tiles = 0;
    b->max_high_freq_count = 0;
    b->max_full_count = 0;
    b->high_freq_count = 0;
    b->full_count = 0;
    b->start_exp = 0;
    b->comp_bytes = 0;
}

bool CompressWeightOnce(
    __nv_bfloat16* matrix,
    int rows,
    int cols,
    CompressedBuffers* out,
    std::string* error) {
    FreeCompressedBuffers(out);

    if (rows % 64 != 0 || cols % 64 != 0) {
        std::ostringstream oss;
        oss << "shape not multiple of 64: " << rows << "x" << cols;
        *error = oss.str();
        return false;
    }

    int top_exponents[7] = {0};
    analyzeExponentDistribution_BF16(matrix, rows, cols, top_exponents);

    int max_high_freq_count = 0;
    int max_full_count = 0;
    int num_global_tiles = InitBF16MatrixTripleBitmap(
        matrix,
        rows,
        cols,
        8, 16, 64,
        8, 64, 64,
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
        return false;
    }

    out->num_global_tiles = num_global_tiles;
    out->max_high_freq_count = max_high_freq_count;
    out->max_full_count = max_full_count;
    out->start_exp = static_cast<uint8_t>(top_exponents[0] - 1);
    out->high_freq_count = out->tile_offsets_global[num_global_tiles * 2];
    out->full_count = out->tile_offsets_global[num_global_tiles * 2 + 1];

    int num_tiles = (rows / 8) * (cols / 8);
    int num_median_tiles = (rows / 16) * (cols / 64);
    out->comp_bytes =
        static_cast<size_t>(out->high_freq_count) * sizeof(uint8_t) +
        static_cast<size_t>(out->full_count) * sizeof(__nv_bfloat16) +
        static_cast<size_t>(num_tiles) * sizeof(uint64_t) * 3 +
        static_cast<size_t>(num_median_tiles) * 2 * sizeof(int) +
        static_cast<size_t>(num_global_tiles + 1) * 2 * sizeof(int);
    return true;
}

void FreeDeviceCompressedBuffers(DeviceCompressedBuffers* b) {
    cudaFree(b->sign_mantissa);
    cudaFree(b->compressed_full);
    cudaFree(b->bitmap1);
    cudaFree(b->bitmap2);
    cudaFree(b->bitmap3);
    cudaFree(b->tile_offsets_median);
    cudaFree(b->tile_offsets_global);
    b->sign_mantissa = nullptr;
    b->compressed_full = nullptr;
    b->bitmap1 = nullptr;
    b->bitmap2 = nullptr;
    b->bitmap3 = nullptr;
    b->tile_offsets_median = nullptr;
    b->tile_offsets_global = nullptr;
}

bool PrepareDeviceCompressed(const CompressedBuffers& host, int rows, int cols, DeviceCompressedBuffers* device, std::string* error) {
    FreeDeviceCompressedBuffers(device);
    int num_tiles = (rows / 8) * (cols / 8);
    int num_median_tiles = (rows / 16) * (cols / 64);

    cudaError_t e = cudaSuccess;
    e = cudaMalloc(reinterpret_cast<void**>(&device->sign_mantissa), host.high_freq_count * sizeof(uint8_t));
    if (e != cudaSuccess) { *error = "cudaMalloc sign_mantissa failed"; return false; }
    e = cudaMalloc(reinterpret_cast<void**>(&device->compressed_full), host.full_count * sizeof(__nv_bfloat16));
    if (e != cudaSuccess) { *error = "cudaMalloc compressed_full failed"; FreeDeviceCompressedBuffers(device); return false; }
    e = cudaMalloc(reinterpret_cast<void**>(&device->bitmap1), num_tiles * sizeof(uint64_t));
    if (e != cudaSuccess) { *error = "cudaMalloc bitmap1 failed"; FreeDeviceCompressedBuffers(device); return false; }
    e = cudaMalloc(reinterpret_cast<void**>(&device->bitmap2), num_tiles * sizeof(uint64_t));
    if (e != cudaSuccess) { *error = "cudaMalloc bitmap2 failed"; FreeDeviceCompressedBuffers(device); return false; }
    e = cudaMalloc(reinterpret_cast<void**>(&device->bitmap3), num_tiles * sizeof(uint64_t));
    if (e != cudaSuccess) { *error = "cudaMalloc bitmap3 failed"; FreeDeviceCompressedBuffers(device); return false; }
    e = cudaMalloc(reinterpret_cast<void**>(&device->tile_offsets_median), num_median_tiles * 2 * sizeof(int));
    if (e != cudaSuccess) { *error = "cudaMalloc tile_offsets_median failed"; FreeDeviceCompressedBuffers(device); return false; }
    e = cudaMalloc(reinterpret_cast<void**>(&device->tile_offsets_global), (host.num_global_tiles + 1) * 2 * sizeof(int));
    if (e != cudaSuccess) { *error = "cudaMalloc tile_offsets_global failed"; FreeDeviceCompressedBuffers(device); return false; }

    cudaMemcpy(device->sign_mantissa, host.sign_mantissa, host.high_freq_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device->compressed_full, host.compressed_full, host.full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(device->bitmap1, host.bitmap1, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device->bitmap2, host.bitmap2, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device->bitmap3, host.bitmap3, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device->tile_offsets_median, host.tile_offsets_median, num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device->tile_offsets_global, host.tile_offsets_global, (host.num_global_tiles + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
    return true;
}

void BuildRandomInputB(int K, int tokens, int seed, std::vector<__nv_bfloat16>* out) {
    out->resize(static_cast<size_t>(K) * static_cast<size_t>(tokens));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int j = 0; j < tokens; ++j) {
        for (int i = 0; i < K; ++i) {
            (*out)[i + j * K] = __float2bfloat16(dis(gen));
        }
    }
}

bool RunBaselineGemm(
    cublasHandle_t handle,
    __nv_bfloat16* d_weight,
    const __nv_bfloat16* h_weight,
    size_t weight_bytes,
    const __nv_bfloat16* d_b,
    __nv_bfloat16* d_out,
    cudaStream_t stream,
    int N_out,
    int tokens,
    int K_in,
    int warmup,
    int iters,
    bool weight_offload,
    double* lat_ms,
    std::string* error) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < warmup; ++i) {
        if (weight_offload) {
            cudaError_t ce = cudaMemcpyAsync(
                d_weight, h_weight, weight_bytes, cudaMemcpyHostToDevice, stream);
            if (ce != cudaSuccess) {
                *error = std::string("baseline weight offload warmup failed: ") + cudaGetErrorString(ce);
                return false;
            }
        }

        cublasStatus_t st = cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            N_out,
            tokens,
            K_in,
            &alpha,
            d_weight,
            CUDA_R_16BF,
            K_in,
            d_b,
            CUDA_R_16BF,
            K_in,
            &beta,
            d_out,
            CUDA_R_16BF,
            N_out,
            CUDA_R_32F,
            static_cast<cublasGemmAlgo_t>(0));
        if (st != CUBLAS_STATUS_SUCCESS) {
            *error = "cublasGemmEx warmup failed";
            return false;
        }
    }

    cudaError_t ce = cudaStreamSynchronize(stream);
    if (ce != cudaSuccess) {
        *error = std::string("baseline warmup synchronize failed: ") + cudaGetErrorString(ce);
        return false;
    }

    bench_common::CudaEventTimer timer;
    if (!timer.RecordStart(stream, error)) {
        return false;
    }
    for (int i = 0; i < iters; ++i) {
        if (weight_offload) {
            cudaError_t ce = cudaMemcpyAsync(
                d_weight, h_weight, weight_bytes, cudaMemcpyHostToDevice, stream);
            if (ce != cudaSuccess) {
                *error = std::string("baseline weight offload benchmark failed: ") + cudaGetErrorString(ce);
                return false;
            }
        }

        cublasStatus_t st = cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            N_out,
            tokens,
            K_in,
            &alpha,
            d_weight,
            CUDA_R_16BF,
            K_in,
            d_b,
            CUDA_R_16BF,
            K_in,
            &beta,
            d_out,
            CUDA_R_16BF,
            N_out,
            CUDA_R_32F,
            static_cast<cublasGemmAlgo_t>(0));
        if (st != CUBLAS_STATUS_SUCCESS) {
            *error = "cublasGemmEx benchmark failed";
            return false;
        }
    }
    if (!timer.RecordStop(stream, error)) {
        return false;
    }
    if (!timer.SyncStop(error)) {
        return false;
    }

    float total_ms = 0.0f;
    if (!timer.ElapsedMs(&total_ms, error)) {
        return false;
    }
    *lat_ms = static_cast<double>(total_ms) / static_cast<double>(iters);
    return true;
}

bool CopyCompressedToDeviceAsync(
    const CompressedBuffers& host,
    int rows,
    int cols,
    const DeviceCompressedBuffers& device,
    cudaStream_t stream,
    std::string* error) {
    int num_tiles = (rows / 8) * (cols / 8);
    int num_median_tiles = (rows / 16) * (cols / 64);

    cudaError_t ce = cudaMemcpyAsync(
        device.sign_mantissa, host.sign_mantissa, host.high_freq_count * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) {
        *error = std::string("zipserv sign_mantissa offload failed: ") + cudaGetErrorString(ce);
        return false;
    }
    ce = cudaMemcpyAsync(
        device.compressed_full, host.compressed_full, host.full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) {
        *error = std::string("zipserv compressed_full offload failed: ") + cudaGetErrorString(ce);
        return false;
    }
    ce = cudaMemcpyAsync(
        device.bitmap1, host.bitmap1, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) {
        *error = std::string("zipserv bitmap1 offload failed: ") + cudaGetErrorString(ce);
        return false;
    }
    ce = cudaMemcpyAsync(
        device.bitmap2, host.bitmap2, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) {
        *error = std::string("zipserv bitmap2 offload failed: ") + cudaGetErrorString(ce);
        return false;
    }
    ce = cudaMemcpyAsync(
        device.bitmap3, host.bitmap3, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) {
        *error = std::string("zipserv bitmap3 offload failed: ") + cudaGetErrorString(ce);
        return false;
    }
    ce = cudaMemcpyAsync(
        device.tile_offsets_median, host.tile_offsets_median, num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) {
        *error = std::string("zipserv tile_offsets_median offload failed: ") + cudaGetErrorString(ce);
        return false;
    }
    ce = cudaMemcpyAsync(
        device.tile_offsets_global, host.tile_offsets_global, (host.num_global_tiles + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) {
        *error = std::string("zipserv tile_offsets_global offload failed: ") + cudaGetErrorString(ce);
        return false;
    }
    return true;
}

bool RunZipservGemm(
    const DeviceCompressedBuffers& comp,
    const CompressedBuffers& host_meta,
    const __nv_bfloat16* d_b,
    __nv_bfloat16* d_out,
    cudaStream_t stream,
    int N_out,
    int tokens,
    int K_in,
    int rows,
    int cols,
    int warmup,
    int iters,
    bool weight_offload,
    double* lat_ms,
    std::string* error) {
    for (int i = 0; i < warmup; ++i) {
        if (weight_offload) {
            if (!CopyCompressedToDeviceAsync(host_meta, rows, cols, comp, stream, error)) {
                return false;
            }
        }

        cudaError_t ce = BF16TripleBitmap_MM_API(
            stream,
            comp.sign_mantissa,
            comp.compressed_full,
            comp.bitmap1,
            comp.bitmap2,
            comp.bitmap3,
            comp.tile_offsets_median,
            comp.tile_offsets_global,
            host_meta.max_high_freq_count,
            host_meta.max_full_count,
            host_meta.start_exp,
            d_b,
            d_out,
            N_out,
            tokens,
            K_in,
            nullptr,
            1);
        if (ce != cudaSuccess) {
            *error = std::string("BF16TripleBitmap_MM_API warmup failed: ") + cudaGetErrorString(ce);
            return false;
        }
    }

    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        *error = std::string("zipserv warmup synchronize failed: ") + cudaGetErrorString(sync_err);
        return false;
    }

    bench_common::CudaEventTimer timer;
    if (!timer.RecordStart(stream, error)) {
        return false;
    }
    for (int i = 0; i < iters; ++i) {
        if (weight_offload) {
            if (!CopyCompressedToDeviceAsync(host_meta, rows, cols, comp, stream, error)) {
                return false;
            }
        }

        cudaError_t ce = BF16TripleBitmap_MM_API(
            stream,
            comp.sign_mantissa,
            comp.compressed_full,
            comp.bitmap1,
            comp.bitmap2,
            comp.bitmap3,
            comp.tile_offsets_median,
            comp.tile_offsets_global,
            host_meta.max_high_freq_count,
            host_meta.max_full_count,
            host_meta.start_exp,
            d_b,
            d_out,
            N_out,
            tokens,
            K_in,
            nullptr,
            1);
        if (ce != cudaSuccess) {
            *error = std::string("BF16TripleBitmap_MM_API benchmark failed: ") + cudaGetErrorString(ce);
            return false;
        }
    }
    if (!timer.RecordStop(stream, error)) {
        return false;
    }
    if (!timer.SyncStop(error)) {
        return false;
    }

    float total_ms = 0.0f;
    if (!timer.ElapsedMs(&total_ms, error)) {
        return false;
    }
    *lat_ms = static_cast<double>(total_ms) / static_cast<double>(iters);
    return true;
}

double GBpsFromBytesAndMs(size_t bytes, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / (ms / 1000.0) / 1e9;
}

std::string CublasMathModeToString(cublasMath_t mode) {
    if (mode == CUBLAS_DEFAULT_MATH) {
        return "CUBLAS_DEFAULT_MATH";
    }
    if (mode == CUBLAS_TENSOR_OP_MATH) {
        return "CUBLAS_TENSOR_OP_MATH";
    }
    if (mode == CUBLAS_TF32_TENSOR_OP_MATH) {
        return "CUBLAS_TF32_TENSOR_OP_MATH";
    }
    if (mode == CUBLAS_PEDANTIC_MATH) {
        return "CUBLAS_PEDANTIC_MATH";
    }
    if (mode == CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION) {
        return "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION";
    }

    std::ostringstream oss;
    oss << "UNKNOWN(" << static_cast<int>(mode) << ")";
    return oss.str();
}

void PrintResultTableHeader() {
    std::cout << "\n===== Per-Tensor Results =====\n";
    std::cout << std::left
              << std::setw(10) << "group"
              << std::setw(12) << "name"
              << std::right
              << std::setw(8) << "tok"
              << std::setw(8) << "M"
              << std::setw(8) << "N"
              << std::setw(8) << "K"
              << std::setw(14) << "base_ms"
              << std::setw(14) << "zip_ms"
              << std::setw(12) << "zip/base"
              << "\n";
    std::cout << std::string(94, '-') << "\n";
}

void PrintResultTableRow(
    const std::string& group,
    const std::string& name,
    int tokens,
    int M,
    int N,
    int K,
    double baseline_ms,
    double zipserv_ms) {
    double ratio = (baseline_ms > 0.0) ? (zipserv_ms / baseline_ms) : 0.0;

    std::cout << std::left
              << std::setw(10) << group
              << std::setw(12) << name
              << std::right
              << std::setw(8) << tokens
              << std::setw(8) << M
              << std::setw(8) << N
              << std::setw(8) << K
              << std::setw(14) << std::fixed << std::setprecision(6) << baseline_ms
              << std::setw(14) << std::fixed << std::setprecision(6) << zipserv_ms
              << std::setw(12) << std::fixed << std::setprecision(3) << ratio
              << "\n";
}

void WriteCsvHeader(std::ofstream* ofs) {
    (*ofs) << "kind,group,name,tok,M,N,K,base_ms,zip_ms,zip/base\n";
}

void WriteCsvRow(std::ofstream* ofs, const GemmCsvRow& row) {
    std::ostringstream base_ms, zip_ms, zip_over_base;
    base_ms << std::fixed << std::setprecision(6) << row.baseline_ms;
    zip_ms << std::fixed << std::setprecision(6) << row.zipserv_ms;
    zip_over_base << std::fixed << std::setprecision(3) << row.zip_over_base;

    (*ofs) << bench_common::EscapeCSV(row.kind) << ","
           << bench_common::EscapeCSV(row.group) << ","
           << bench_common::EscapeCSV(row.name) << ","
           << row.tokens << ","
           << row.M << ","
           << row.N << ","
           << row.K << ","
           << base_ms.str() << ","
           << zip_ms.str() << ","
           << zip_over_base.str()
           << "\n";
}

bool ShouldUseGroup(const std::set<std::string>& wanted, const std::string& group) {
    return wanted.find(group) != wanted.end();
}

bool ParseOps(const std::string& spec, std::set<std::string>* ops, std::string* error) {
    ops->clear();
    std::vector<std::string> tokens = bench_common::SplitByComma(spec);
    if (tokens.empty()) {
        *error = "ops spec is empty";
        return false;
    }

    const std::set<std::string> valid = {"qkv", "o", "gateup", "down"};
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string t = tokens[i];
        std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (t == "all") {
            ops->insert("qkv");
            ops->insert("o");
            ops->insert("gateup");
            ops->insert("down");
            continue;
        }
        if (valid.find(t) == valid.end()) {
            *error = "invalid op group: " + t;
            return false;
        }
        ops->insert(t);
    }
    return true;
}

int main(int argc, char** argv) {
    ProgramOptions opts;
    if (!ParseOptions(argc, argv, &opts)) {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            return 0;
        }
        return 1;
    }

    std::set<std::string> wanted_ops;
    std::string ops_error;
    if (!ParseOps(opts.ops_spec, &wanted_ops, &ops_error)) {
        std::cerr << "ops parse error: " << ops_error << "\n";
        return 1;
    }

    std::vector<int> tokens;
    std::string tokens_error;
    if (!ParseIntList(opts.tokens_spec, &tokens, &tokens_error)) {
        std::cerr << "tokens parse error: " << tokens_error << "\n";
        return 1;
    }

    cudaError_t ce = cudaSetDevice(opts.device);
    if (ce != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(ce) << "\n";
        return 1;
    }

    std::vector<ManifestTensor> manifest;
    std::string manifest_error;
    if (!LoadManifest(opts.manifest_path, &manifest, &manifest_error)) {
        std::cerr << manifest_error << "\n";
        return 1;
    }

    std::map<std::string, ManifestTensor> by_suffix;
    for (size_t i = 0; i < manifest.size(); ++i) {
        const ManifestTensor& t = manifest[i];
        if (t.layer != opts.layer) {
            continue;
        }
        if (bench_common::EndsWith(t.name, ".q_proj.weight")) by_suffix["q_proj"] = t;
        if (bench_common::EndsWith(t.name, ".k_proj.weight")) by_suffix["k_proj"] = t;
        if (bench_common::EndsWith(t.name, ".v_proj.weight")) by_suffix["v_proj"] = t;
        if (bench_common::EndsWith(t.name, ".o_proj.weight")) by_suffix["o_proj"] = t;
        if (bench_common::EndsWith(t.name, ".gate_proj.weight")) by_suffix["gate_proj"] = t;
        if (bench_common::EndsWith(t.name, ".up_proj.weight")) by_suffix["up_proj"] = t;
        if (bench_common::EndsWith(t.name, ".down_proj.weight")) by_suffix["down_proj"] = t;
    }

    struct OpGroupSpec {
        const char* group;
        const char* tensors[3];
        int tensor_count;
    };
    const OpGroupSpec kOpSpecs[] = {
        {"qkv", {"q_proj", "k_proj", "v_proj"}, 3},
        {"o", {"o_proj", nullptr, nullptr}, 1},
        {"gateup", {"gate_proj", "up_proj", nullptr}, 2},
        {"down", {"down_proj", nullptr, nullptr}, 1},
    };

    std::vector<std::pair<std::string, std::string>> request_order;
    for (size_t spec_i = 0; spec_i < sizeof(kOpSpecs) / sizeof(kOpSpecs[0]); ++spec_i) {
        const OpGroupSpec& spec = kOpSpecs[spec_i];
        if (!ShouldUseGroup(wanted_ops, spec.group)) {
            continue;
        }
        for (int ti = 0; ti < spec.tensor_count; ++ti) {
            request_order.push_back({spec.group, spec.tensors[ti]});
        }
    }

    std::vector<WeightSpec> weights;
    for (size_t i = 0; i < request_order.size(); ++i) {
        const std::string& group = request_order[i].first;
        const std::string& short_name = request_order[i].second;
        if (by_suffix.find(short_name) == by_suffix.end()) {
            std::cerr << "Missing tensor in manifest (layer " << opts.layer << "): " << short_name << "\n";
            return 1;
        }
        WeightSpec w;
        w.group = group;
        w.short_name = short_name;
        w.tensor = by_suffix[short_name];
        if (w.tensor.shape.size() != 2) {
            std::cerr << "Tensor shape is not 2D: " << w.tensor.name << "\n";
            return 1;
        }
        w.rows = w.tensor.shape[0];
        w.cols = w.tensor.shape[1];

        std::string load_error;
        if (!ConvertRawToBF16(w.tensor, &w.host_weight, &load_error)) {
            std::cerr << load_error << "\n";
            return 1;
        }
        weights.push_back(w);
    }

    std::ofstream ofs(opts.out_csv.c_str(), std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open CSV: " << opts.out_csv << "\n";
        return 1;
    }
    WriteCsvHeader(&ofs);

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasCreate failed\n";
        return 1;
    }
    cudaStream_t benchmark_stream = 0;
    if (cublasSetStream(handle, benchmark_stream) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSetStream failed\n";
        cublasDestroy(handle);
        return 1;
    }
    if (cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSetMathMode failed\n";
        cublasDestroy(handle);
        return 1;
    }
    cublasMath_t current_math_mode = CUBLAS_DEFAULT_MATH;
    if (cublasGetMathMode(handle, &current_math_mode) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasGetMathMode failed\n";
        cublasDestroy(handle);
        return 1;
    }

    const std::string baseline_cfg =
        std::string("api=cublasGemmEx,algo=0,opA=T,opB=N,a_type=CUDA_R_16BF,b_type=CUDA_R_16BF,c_type=CUDA_R_16BF,")
        + "compute=CUDA_R_32F,math=" + CublasMathModeToString(current_math_mode)
        + ",lda=K,ldb=K,ldc=N,weight_offload=" + (opts.weight_offload ? "1" : "0")
        + ",timing=" + (opts.weight_offload ? "H2D(weight)+GEMM" : "GEMM_only");
    const std::string zipserv_cfg =
        std::string("api=BF16TripleBitmap_MM_API,split_k=1,stream=0,weight_offload=")
        + (opts.weight_offload ? "1" : "0")
        + ",timing="
        + (opts.weight_offload ? "H2D(compressed_weight)+ZipServGEMM,compress_excluded=1"
                               : "ZipServGEMM_only");

    std::cout << "===== ZipServ GEMM Sweep =====\n";
    std::cout << "manifest=" << opts.manifest_path << "\n";
    std::cout << "layer=" << opts.layer << "\n";
    std::cout << "ops=" << opts.ops_spec << "\n";
    std::cout << "tokens=" << opts.tokens_spec << "\n";
    std::cout << "warmup=" << opts.warmup << ", iters=" << opts.iters << "\n";
    std::cout << "device=" << opts.device << "\n";
    std::cout << "weight_offload=" << (opts.weight_offload ? 1 : 0) << "\n";
    std::cout << "baseline_cfg=" << baseline_cfg << "\n";
    std::cout << "zipserv_cfg=" << zipserv_cfg << "\n";
    PrintResultTableHeader();

    std::vector<GemmCsvRow> all_rows;

    for (size_t wi = 0; wi < weights.size(); ++wi) {
        WeightSpec& w = weights[wi];

        CompressedBuffers comp_host;
        std::string comp_error;
        if (!CompressWeightOnce(w.host_weight.data(), w.rows, w.cols, &comp_host, &comp_error)) {
            std::cerr << "Compress failed for " << w.tensor.name << ": " << comp_error << "\n";
            cublasDestroy(handle);
            return 1;
        }

        DeviceCompressedBuffers comp_dev;
        if (!PrepareDeviceCompressed(comp_host, w.rows, w.cols, &comp_dev, &comp_error)) {
            std::cerr << "Prepare compressed device buffers failed for " << w.tensor.name << ": " << comp_error << "\n";
            FreeCompressedBuffers(&comp_host);
            cublasDestroy(handle);
            return 1;
        }

        __nv_bfloat16* d_weight = nullptr;
        size_t weight_bytes = static_cast<size_t>(w.rows) * static_cast<size_t>(w.cols) * sizeof(__nv_bfloat16);
        cudaMalloc(reinterpret_cast<void**>(&d_weight), weight_bytes);
        cudaMemcpy(d_weight, w.host_weight.data(), weight_bytes, cudaMemcpyHostToDevice);
        checkLastCudaError(__LINE__);

        for (size_t ti = 0; ti < tokens.size(); ++ti) {
            int tok = tokens[ti];
            int logical_M = tok;
            int logical_N = w.rows;
            int logical_K = w.cols;

            std::vector<__nv_bfloat16> b_host;
            BuildRandomInputB(w.cols, tok, opts.seed + static_cast<int>(wi * 97 + tok), &b_host);

            __nv_bfloat16* d_b = nullptr;
            __nv_bfloat16* d_out_baseline = nullptr;
            __nv_bfloat16* d_out_zipserv = nullptr;
            size_t b_bytes = static_cast<size_t>(w.cols) * static_cast<size_t>(tok) * sizeof(__nv_bfloat16);
            size_t out_bytes = static_cast<size_t>(w.rows) * static_cast<size_t>(tok) * sizeof(__nv_bfloat16);
            cudaMalloc(reinterpret_cast<void**>(&d_b), b_bytes);
            cudaMalloc(reinterpret_cast<void**>(&d_out_baseline), out_bytes);
            cudaMalloc(reinterpret_cast<void**>(&d_out_zipserv), out_bytes);
            cudaMemcpy(d_b, b_host.data(), b_bytes, cudaMemcpyHostToDevice);
            cudaMemset(d_out_baseline, 0, out_bytes);
            cudaMemset(d_out_zipserv, 0, out_bytes);
            checkLastCudaError(__LINE__);

            double baseline_ms = 0.0;
            double zipserv_ms = 0.0;
            std::string run_error;
            if (!RunBaselineGemm(
                    handle,
                    d_weight,
                    w.host_weight.data(),
                    weight_bytes,
                    d_b,
                    d_out_baseline,
                    benchmark_stream,
                    w.rows,
                    tok,
                    w.cols,
                    opts.warmup,
                    opts.iters,
                    opts.weight_offload,
                    &baseline_ms,
                    &run_error)) {
                std::cerr << "Baseline GEMM failed: " << run_error << " tensor=" << w.tensor.name << " tokens=" << tok << "\n";
                cudaFree(d_b);
                cudaFree(d_out_baseline);
                cudaFree(d_out_zipserv);
                cudaFree(d_weight);
                FreeDeviceCompressedBuffers(&comp_dev);
                FreeCompressedBuffers(&comp_host);
                cublasDestroy(handle);
                return 1;
            }

            if (!RunZipservGemm(
                    comp_dev,
                    comp_host,
                    d_b,
                    d_out_zipserv,
                    benchmark_stream,
                    w.rows,
                    tok,
                    w.cols,
                    w.rows,
                    w.cols,
                    opts.warmup,
                    opts.iters,
                    opts.weight_offload,
                    &zipserv_ms,
                    &run_error)) {
                std::cerr << "ZipServ GEMM failed: " << run_error << " tensor=" << w.tensor.name << " tokens=" << tok << "\n";
                cudaFree(d_b);
                cudaFree(d_out_baseline);
                cudaFree(d_out_zipserv);
                cudaFree(d_weight);
                FreeDeviceCompressedBuffers(&comp_dev);
                FreeCompressedBuffers(&comp_host);
                cublasDestroy(handle);
                return 1;
            }

            GemmCsvRow row;
            row.kind = "tensor";
            row.group = w.group;
            row.name = w.short_name;
            row.layer = opts.layer;
            row.tokens = tok;
            row.M = logical_M;
            row.N = logical_N;
            row.K = logical_K;
            row.weight_name = w.tensor.name;
            row.weight_shape = bench_common::ShapeToString(w.tensor.shape);
            row.dtype = w.tensor.dtype;
            row.orig_bytes = w.tensor.nbytes;
            row.comp_bytes = comp_host.comp_bytes;
            row.baseline_ms = baseline_ms;
            row.zipserv_ms = zipserv_ms;
            row.zip_over_base = (baseline_ms > 0.0) ? (zipserv_ms / baseline_ms) : 0.0;
            row.baseline_gbps = GBpsFromBytesAndMs(w.tensor.nbytes, baseline_ms);
            row.zipserv_gbps = GBpsFromBytesAndMs(w.tensor.nbytes, zipserv_ms);
            row.x_shape = "[" + std::to_string(tok) + "," + std::to_string(w.cols) + "]";
            row.w_shape = "[" + std::to_string(w.rows) + "," + std::to_string(w.cols) + "]";
            row.y_shape = "[" + std::to_string(tok) + "," + std::to_string(w.rows) + "]";
            row.baseline_cfg = baseline_cfg;
            row.zipserv_cfg = zipserv_cfg;

            all_rows.push_back(row);

            PrintResultTableRow(
                w.group,
                w.short_name,
                tok,
                logical_M,
                logical_N,
                logical_K,
                baseline_ms,
                zipserv_ms);

            cudaFree(d_b);
            cudaFree(d_out_baseline);
            cudaFree(d_out_zipserv);
        }

        cudaFree(d_weight);
        FreeDeviceCompressedBuffers(&comp_dev);
        FreeCompressedBuffers(&comp_host);
    }

    if (opts.emit_aggregate) {
        struct Agg {
            size_t orig_bytes = 0;
            size_t comp_bytes = 0;
            double baseline_ms = 0.0;
            double zipserv_ms = 0.0;
        };
        std::map<std::string, Agg> aggs;

        for (size_t i = 0; i < all_rows.size(); ++i) {
            const GemmCsvRow& r = all_rows[i];
            if (r.kind != "tensor") {
                continue;
            }
            if (!(r.group == "qkv" || r.group == "gateup")) {
                continue;
            }
            std::string key = r.group + "|" + std::to_string(r.tokens);
            Agg& a = aggs[key];
            a.orig_bytes += r.orig_bytes;
            a.comp_bytes += r.comp_bytes;
            a.baseline_ms += r.baseline_ms;
            a.zipserv_ms += r.zipserv_ms;
        }

        for (std::map<std::string, Agg>::const_iterator it = aggs.begin(); it != aggs.end(); ++it) {
            std::string key = it->first;
            const Agg& a = it->second;
            std::vector<std::string> parts;
            std::stringstream ss(key);
            std::string part;
            while (std::getline(ss, part, '|')) {
                parts.push_back(part);
            }
            if (parts.size() != 2) {
                continue;
            }

            GemmCsvRow row;
            row.kind = "aggregate";
            row.group = parts[0] + "_sum";
            row.name = row.group;
            row.layer = opts.layer;
            row.tokens = std::atoi(parts[1].c_str());
            row.M = row.tokens;
            row.N = 0;
            row.K = 0;
            row.weight_name = row.group;
            row.weight_shape = "mixed";
            row.dtype = "bf16";
            row.orig_bytes = a.orig_bytes;
            row.comp_bytes = a.comp_bytes;
            row.baseline_ms = a.baseline_ms;
            row.zipserv_ms = a.zipserv_ms;
            row.zip_over_base = (a.baseline_ms > 0.0) ? (a.zipserv_ms / a.baseline_ms) : 0.0;
            row.baseline_gbps = GBpsFromBytesAndMs(a.orig_bytes, a.baseline_ms);
            row.zipserv_gbps = GBpsFromBytesAndMs(a.orig_bytes, a.zipserv_ms);
            row.x_shape = "mixed";
            row.w_shape = "mixed";
            row.y_shape = "mixed";
            row.baseline_cfg = baseline_cfg;
            row.zipserv_cfg = zipserv_cfg;
            all_rows.push_back(row);
            PrintResultTableRow(
                row.group,
                row.name,
                row.tokens,
                row.M,
                row.N,
                row.K,
                row.baseline_ms,
                row.zipserv_ms);
        }
    }

    for (size_t i = 0; i < all_rows.size(); ++i) {
        WriteCsvRow(&ofs, all_rows[i]);
    }
    ofs.close();

    cublasDestroy(handle);

    std::cout << "===== GEMM Sweep Summary =====\n";
    std::cout << "rows_written=" << all_rows.size() << "\n";
    std::cout << "csv=" << opts.out_csv << "\n";
    return 0;
}
