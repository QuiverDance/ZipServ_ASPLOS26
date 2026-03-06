#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "L_API.cuh"
#include "bench_manifest_utils.h"
#include "csv_writer.h"
#include "cuda_timer.h"
#include "./utils.h"

#if defined(ZIPSERV_ENABLE_NVCOMP) && __has_include(<nvcomp/native/bitcomp.h>)
#include <nvcomp/native/bitcomp.h>
#define ZIPSERV_NVCOMP_BITCOMP_ENABLED 1
#else
#define ZIPSERV_NVCOMP_BITCOMP_ENABLED 0
#endif

struct ProgramOptions {
    std::string model_path;
    std::string manifest_path_override;
    int warmup = 10;
    int iters = 100;
    int seed = 1234;
    int device = 0;
    int stream_sync = 1;
    std::string csv_path = "llama31_70b_zipserv_vs_bitcomp_compress.csv";
    bool verbose = false;

    int layer_idx = -1;
    std::string weight_filter;
    int max_tensors = -1;
    int aggregate_only = 0;
    int use_bitcomp_batch = 0;
    int bitcomp_include_size_query = 1;
};

struct ManifestTensor {
    int layer = -1;
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    std::string path;
    size_t orig_bytes = 0;
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

struct MetricStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    bool valid = false;
};

struct BackendTensorResult {
    std::string backend;
    std::string tensor_name;
    int layer_idx = -1;
    size_t input_bytes = 0;
    size_t compressed_bytes = 0;

    std::vector<double> compress_write_ms;
    std::vector<double> size_query_ms;
    std::vector<double> total_ms;

    std::string status = "ok";
    std::string note;
};

void PrintUsage() {
    std::cout
        << "Usage: ./bench_llama31_70b_zipserv_vs_bitcomp_compress --model_path <path> [options]\n"
        << "Options:\n"
        << "  --model_path <path>                Model root path (expects manifest/weights_manifest.jsonl)\n"
        << "  --manifest <path>                  Manifest JSONL path override\n"
        << "  --iters <int>                      Benchmark iterations per tensor (default: 100)\n"
        << "  --warmup <int>                     Warmup iterations per tensor (default: 10)\n"
        << "  --seed <int>                       RNG seed (default: 1234)\n"
        << "  --device <int>                     CUDA device index (default: 0)\n"
        << "  --stream_sync <0|1>                Synchronize stream after timed region (default: 1)\n"
        << "  --csv <path>                       Output CSV path\n"
        << "  --verbose                          Verbose logs\n"
        << "  --layer_idx <int>                  Specific layer only (default: -1 means all)\n"
        << "  --weight_filter <regex|string>     Tensor filter (substring default, regex when regex-like)\n"
        << "  --max_tensors <int>                Limit number of selected tensors\n"
        << "  --aggregate_only <0|1>             Print only aggregate rows (default: 0)\n"
        << "  --use_bitcomp_batch <0|1>          Use bitcomp batched API path (default: 0)\n"
        << "  --bitcomp_include_size_query <0|1> Measure bitcomp size query separately (default: 1)\n"
        << "  --help                             Show help\n";
}

bool ParseBool01(const std::string& arg_name, const std::string& text, int* out) {
    if (text == "0") {
        *out = 0;
        return true;
    }
    if (text == "1") {
        *out = 1;
        return true;
    }
    std::cerr << "Error: " << arg_name << " must be 0 or 1, got: " << text << "\n";
    return false;
}

bool ParseOptions(int argc, char** argv, ProgramOptions* opts) {
    if (argc <= 1) {
        PrintUsage();
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            PrintUsage();
            return false;
        } else if (arg == "--model_path" && i + 1 < argc) {
            opts->model_path = argv[++i];
        } else if (arg == "--manifest" && i + 1 < argc) {
            opts->manifest_path_override = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            opts->iters = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts->warmup = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            opts->seed = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            opts->device = std::atoi(argv[++i]);
        } else if (arg == "--stream_sync" && i + 1 < argc) {
            int value = 0;
            if (!ParseBool01("--stream_sync", argv[++i], &value)) {
                return false;
            }
            opts->stream_sync = value;
        } else if (arg == "--csv" && i + 1 < argc) {
            opts->csv_path = argv[++i];
        } else if (arg == "--verbose") {
            opts->verbose = true;
        } else if (arg == "--layer_idx" && i + 1 < argc) {
            opts->layer_idx = std::atoi(argv[++i]);
        } else if (arg == "--weight_filter" && i + 1 < argc) {
            opts->weight_filter = argv[++i];
        } else if (arg == "--max_tensors" && i + 1 < argc) {
            opts->max_tensors = std::atoi(argv[++i]);
        } else if (arg == "--aggregate_only" && i + 1 < argc) {
            int value = 0;
            if (!ParseBool01("--aggregate_only", argv[++i], &value)) {
                return false;
            }
            opts->aggregate_only = value;
        } else if (arg == "--use_bitcomp_batch" && i + 1 < argc) {
            int value = 0;
            if (!ParseBool01("--use_bitcomp_batch", argv[++i], &value)) {
                return false;
            }
            opts->use_bitcomp_batch = value;
        } else if (arg == "--bitcomp_include_size_query" && i + 1 < argc) {
            int value = 0;
            if (!ParseBool01("--bitcomp_include_size_query", argv[++i], &value)) {
                return false;
            }
            opts->bitcomp_include_size_query = value;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            PrintUsage();
            return false;
        }
    }

    if (opts->manifest_path_override.empty() && opts->model_path.empty()) {
        std::cerr << "Error: --model_path is required when --manifest is not provided.\n";
        return false;
    }
    if (opts->warmup < 0 || opts->iters <= 0) {
        std::cerr << "Error: warmup must be >= 0 and iters must be > 0.\n";
        return false;
    }
    if (opts->device < 0) {
        std::cerr << "Error: --device must be >= 0.\n";
        return false;
    }
    if (opts->layer_idx < -1) {
        std::cerr << "Error: --layer_idx must be >= -1.\n";
        return false;
    }
    if (opts->max_tensors == 0 || opts->max_tensors < -1) {
        std::cerr << "Error: --max_tensors must be -1 or > 0.\n";
        return false;
    }
    return true;
}

std::string ResolveManifestPath(const ProgramOptions& opts) {
    if (!opts.manifest_path_override.empty()) {
        return opts.manifest_path_override;
    }
    std::string base = opts.model_path;
    while (!base.empty() && base.back() == '/') {
        base.pop_back();
    }
    return base + "/manifest/weights_manifest.jsonl";
}

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
        t.orig_bytes = parsed[i].nbytes;
        tensors->push_back(t);
    }
    return true;
}

bool ContainsRegexMeta(const std::string& pattern) {
    const std::string metas = R"(.^$|()[]{}*+?\\)";
    for (size_t i = 0; i < pattern.size(); ++i) {
        if (metas.find(pattern[i]) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool BuildWeightFilterRegex(
    const std::string& weight_filter,
    bool* use_regex,
    std::regex* compiled,
    std::string* note) {
    *use_regex = false;
    if (weight_filter.empty()) {
        return true;
    }
    if (!ContainsRegexMeta(weight_filter)) {
        return true;
    }

    try {
        *compiled = std::regex(weight_filter);
        *use_regex = true;
        return true;
    } catch (const std::exception& e) {
        if (note != nullptr) {
            *note = std::string("weight_filter regex parse failed; fallback to substring: ") + e.what();
        }
        *use_regex = false;
        return true;
    }
}

bool TensorPassesFilter(const ManifestTensor& t, const std::string& weight_filter, bool use_regex, const std::regex& compiled) {
    if (weight_filter.empty()) {
        return true;
    }
    if (use_regex) {
        return std::regex_search(t.name, compiled);
    }
    return t.name.find(weight_filter) != std::string::npos;
}

std::vector<ManifestTensor> SelectTensors(const std::vector<ManifestTensor>& all, const ProgramOptions& opts, bool use_regex, const std::regex& compiled) {
    std::vector<ManifestTensor> selected;
    selected.reserve(all.size());

    for (size_t i = 0; i < all.size(); ++i) {
        const ManifestTensor& t = all[i];
        if (opts.layer_idx >= 0 && t.layer != opts.layer_idx) {
            continue;
        }
        if (!TensorPassesFilter(t, opts.weight_filter, use_regex, compiled)) {
            continue;
        }
        selected.push_back(t);
    }

    std::sort(
        selected.begin(),
        selected.end(),
        [](const ManifestTensor& a, const ManifestTensor& b) {
            if (a.layer != b.layer) {
                return a.layer < b.layer;
            }
            return a.name < b.name;
        });

    if (opts.max_tensors > 0 && static_cast<int>(selected.size()) > opts.max_tensors) {
        std::mt19937 rng(static_cast<unsigned>(opts.seed));
        std::shuffle(selected.begin(), selected.end(), rng);
        selected.resize(static_cast<size_t>(opts.max_tensors));
    }

    return selected;
}

bool ValidateTensor(const ManifestTensor& tensor, std::string* error) {
    if (tensor.dtype != "bf16") {
        *error = "manifest_dtype_must_be_bf16";
        return false;
    }
    if (tensor.shape.size() != 2) {
        *error = "non_2d_shape";
        return false;
    }
    const int rows = tensor.shape[0];
    const int cols = tensor.shape[1];
    if (rows <= 0 || cols <= 0) {
        *error = "invalid_shape_dimension";
        return false;
    }
    if ((rows % 64) != 0 || (cols % 64) != 0) {
        *error = "shape_not_multiple_of_64";
        return false;
    }
    return true;
}

bool ReadTensorAsBF16(const ManifestTensor& tensor, std::vector<__nv_bfloat16>* out, std::string* error) {
    std::vector<uint8_t> raw;
    if (!bench_common::ReadBinaryFile(tensor.path, &raw, error)) {
        return false;
    }

    const int rows = tensor.shape[0];
    const int cols = tensor.shape[1];
    const size_t expected_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(uint16_t);
    if (raw.size() != expected_bytes) {
        std::ostringstream oss;
        oss << "file_bytes_mismatch(expected=" << expected_bytes << ",got=" << raw.size() << ")";
        *error = oss.str();
        return false;
    }

    out->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    std::memcpy(out->data(), raw.data(), expected_bytes);
    return true;
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

size_t MeanBytesRounded(const std::vector<size_t>& values) {
    if (values.empty()) {
        return 0;
    }
    long double sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        sum += static_cast<long double>(values[i]);
    }
    return static_cast<size_t>(std::llround(sum / static_cast<long double>(values.size())));
}

MetricStats ComputeStats(const std::vector<double>& values) {
    MetricStats s;
    if (values.empty()) {
        return s;
    }

    s.valid = true;
    s.min = values[0];
    s.max = values[0];
    double sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double v = values[i];
        sum += v;
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
    }
    s.mean = sum / static_cast<double>(values.size());

    double var = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double d = values[i] - s.mean;
        var += d * d;
    }
    var /= static_cast<double>(values.size());
    s.stddev = std::sqrt(var);
    return s;
}

double CompressionRatio(size_t input_bytes, size_t compressed_bytes) {
    if (input_bytes == 0 || compressed_bytes == 0) {
        return 0.0;
    }
    return static_cast<double>(input_bytes) / static_cast<double>(compressed_bytes);
}

double ThroughputGBps(size_t bytes, double milliseconds) {
    if (bytes == 0 || milliseconds <= 0.0) {
        return 0.0;
    }
    const double sec = milliseconds / 1000.0;
    return static_cast<double>(bytes) / sec / 1e9;
}

std::string FormatDouble(double v, int precision = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

std::string EscapeCSV(const std::string& s) {
    return bench_common::EscapeCSV(s);
}

void WriteCSVHeader(std::ofstream* ofs) {
    (*ofs)
        << "backend,tensor_name,layer_idx,input_bytes,compressed_bytes,compression_ratio,"
        << "compress_write_ms_mean,compress_write_ms_std,compress_write_ms_min,compress_write_ms_max,"
        << "size_query_ms_mean,size_query_ms_std,size_query_ms_min,size_query_ms_max,"
        << "total_ms_mean,total_ms_std,total_ms_min,total_ms_max,"
        << "throughput_input_GBps,throughput_output_GBps,iters,warmup,status,note\n";
}

void WriteCSVRow(std::ofstream* ofs, const BackendTensorResult& row, const ProgramOptions& opts) {
    const MetricStats comp = ComputeStats(row.compress_write_ms);
    const MetricStats size_q = ComputeStats(row.size_query_ms);
    const MetricStats total = ComputeStats(row.total_ms);

    const double ratio = CompressionRatio(row.input_bytes, row.compressed_bytes);
    const double input_gbps = ThroughputGBps(row.input_bytes, total.mean);
    const double output_gbps = ThroughputGBps(row.compressed_bytes, total.mean);

    (*ofs)
        << EscapeCSV(row.backend) << ","
        << EscapeCSV(row.tensor_name) << ","
        << row.layer_idx << ","
        << row.input_bytes << ","
        << row.compressed_bytes << ","
        << FormatDouble(ratio, 6) << ","
        << FormatDouble(comp.mean) << ","
        << FormatDouble(comp.stddev) << ","
        << FormatDouble(comp.min) << ","
        << FormatDouble(comp.max) << ","
        << FormatDouble(size_q.mean) << ","
        << FormatDouble(size_q.stddev) << ","
        << FormatDouble(size_q.min) << ","
        << FormatDouble(size_q.max) << ","
        << FormatDouble(total.mean) << ","
        << FormatDouble(total.stddev) << ","
        << FormatDouble(total.min) << ","
        << FormatDouble(total.max) << ","
        << FormatDouble(input_gbps, 6) << ","
        << FormatDouble(output_gbps, 6) << ","
        << opts.iters << ","
        << opts.warmup << ","
        << EscapeCSV(row.status) << ","
        << EscapeCSV(row.note)
        << "\n";
}

void PrintHeader() {
    std::cout << std::left
              << std::setw(10) << "backend"
              << std::setw(8) << "status"
              << std::setw(8) << "layer"
              << std::setw(38) << "tensor"
              << std::setw(12) << "in_bytes"
              << std::setw(12) << "cmp_bytes"
              << std::setw(10) << "ratio"
              << std::setw(15) << "comp_ms(mean)"
              << std::setw(15) << "size_ms(mean)"
              << std::setw(15) << "total_ms(mean)"
              << std::setw(14) << "in_GBps"
              << std::setw(14) << "out_GBps"
              << "note\n";
    std::cout << std::string(190, '-') << "\n";
}

std::string Truncate(const std::string& s, size_t n) {
    if (s.size() <= n) {
        return s;
    }
    if (n <= 3) {
        return s.substr(0, n);
    }
    return s.substr(0, n - 3) + "...";
}

void PrintRow(const BackendTensorResult& row) {
    const MetricStats comp = ComputeStats(row.compress_write_ms);
    const MetricStats size_q = ComputeStats(row.size_query_ms);
    const MetricStats total = ComputeStats(row.total_ms);

    const double ratio = CompressionRatio(row.input_bytes, row.compressed_bytes);
    const double input_gbps = ThroughputGBps(row.input_bytes, total.mean);
    const double output_gbps = ThroughputGBps(row.compressed_bytes, total.mean);

    std::cout << std::left
              << std::setw(10) << row.backend
              << std::setw(8) << row.status
              << std::setw(8) << row.layer_idx
              << std::setw(38) << Truncate(row.tensor_name, 37)
              << std::setw(12) << row.input_bytes
              << std::setw(12) << row.compressed_bytes
              << std::setw(10) << FormatDouble(ratio, 3)
              << std::setw(15) << FormatDouble(comp.mean, 4)
              << std::setw(15) << FormatDouble(size_q.mean, 4)
              << std::setw(15) << FormatDouble(total.mean, 4)
              << std::setw(14) << FormatDouble(input_gbps, 4)
              << std::setw(14) << FormatDouble(output_gbps, 4)
              << Truncate(row.note, 32)
              << "\n";

    if (row.status == "ok") {
        std::cout << "  stats(" << row.backend << ", " << row.tensor_name << "): "
                  << "comp_ms(mean/std/min/max)=" << FormatDouble(comp.mean, 4) << "/" << FormatDouble(comp.stddev, 4)
                  << "/" << FormatDouble(comp.min, 4) << "/" << FormatDouble(comp.max, 4)
                  << ", size_ms(mean/std/min/max)=" << FormatDouble(size_q.mean, 4) << "/" << FormatDouble(size_q.stddev, 4)
                  << "/" << FormatDouble(size_q.min, 4) << "/" << FormatDouble(size_q.max, 4)
                  << ", total_ms(mean/std/min/max)=" << FormatDouble(total.mean, 4) << "/" << FormatDouble(total.stddev, 4)
                  << "/" << FormatDouble(total.min, 4) << "/" << FormatDouble(total.max, 4)
                  << "\n";
    }
}

BackendTensorResult MakeFailedResult(
    const std::string& backend,
    const ManifestTensor& tensor,
    const std::string& note,
    size_t input_bytes = 0) {
    BackendTensorResult row;
    row.backend = backend;
    row.tensor_name = tensor.name;
    row.layer_idx = tensor.layer;
    row.input_bytes = input_bytes;
    row.status = "failed";
    row.note = note;
    return row;
}

BackendTensorResult BenchmarkZipServ(
    const ManifestTensor& tensor,
    const std::vector<__nv_bfloat16>& host_bf16,
    const ProgramOptions& opts) {
    BackendTensorResult row;
    row.backend = "zipserv";
    row.tensor_name = tensor.name;
    row.layer_idx = tensor.layer;
    row.input_bytes = tensor.orig_bytes;

    const int rows = tensor.shape[0];
    const int cols = tensor.shape[1];

    int top_exponents[7] = {0};
    analyzeExponentDistribution_BF16(const_cast<__nv_bfloat16*>(host_bf16.data()), rows, cols, top_exponents);

    for (int i = 0; i < opts.warmup; ++i) {
        CompressedBuffers warmup;
        std::string error;
        if (!RunCompressionOnce(const_cast<__nv_bfloat16*>(host_bf16.data()), rows, cols, top_exponents, &warmup, &error)) {
            return MakeFailedResult("zipserv", tensor, error, tensor.orig_bytes);
        }
        FreeCompressedBuffers(&warmup);
    }

    std::vector<size_t> comp_bytes_samples;
    comp_bytes_samples.reserve(static_cast<size_t>(opts.iters));
    row.compress_write_ms.reserve(static_cast<size_t>(opts.iters));
    row.size_query_ms.reserve(static_cast<size_t>(opts.iters));
    row.total_ms.reserve(static_cast<size_t>(opts.iters));

    for (int i = 0; i < opts.iters; ++i) {
        CompressedBuffers iteration;
        std::string error;

        const std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
        const bool ok = RunCompressionOnce(const_cast<__nv_bfloat16*>(host_bf16.data()), rows, cols, top_exponents, &iteration, &error);
        const std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        if (!ok) {
            FreeCompressedBuffers(&iteration);
            return MakeFailedResult("zipserv", tensor, error, tensor.orig_bytes);
        }

        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        row.compress_write_ms.push_back(elapsed_ms);
        row.size_query_ms.push_back(0.0);
        row.total_ms.push_back(elapsed_ms);
        comp_bytes_samples.push_back(iteration.comp_bytes);

        FreeCompressedBuffers(&iteration);
    }

    row.compressed_bytes = MeanBytesRounded(comp_bytes_samples);
    if (row.compressed_bytes == 0) {
        return MakeFailedResult("zipserv", tensor, "zipserv_compressed_bytes_zero", tensor.orig_bytes);
    }
    return row;
}

#if ZIPSERV_NVCOMP_BITCOMP_ENABLED

const char* BitcompResultToString(bitcompResult_t result) {
    switch (result) {
        case BITCOMP_SUCCESS:
            return "BITCOMP_SUCCESS";
        case BITCOMP_INVALID_PARAMETER:
            return "BITCOMP_INVALID_PARAMETER";
        case BITCOMP_INVALID_COMPRESSED_DATA:
            return "BITCOMP_INVALID_COMPRESSED_DATA";
        case BITCOMP_INVALID_ALIGNMENT:
            return "BITCOMP_INVALID_ALIGNMENT";
        case BITCOMP_INVALID_INPUT_LENGTH:
            return "BITCOMP_INVALID_INPUT_LENGTH";
        case BITCOMP_CUDA_KERNEL_LAUNCH_ERROR:
            return "BITCOMP_CUDA_KERNEL_LAUNCH_ERROR";
        case BITCOMP_CUDA_API_ERROR:
            return "BITCOMP_CUDA_API_ERROR";
        case BITCOMP_UNKNOWN_ERROR:
            return "BITCOMP_UNKNOWN_ERROR";
        default:
            return "BITCOMP_UNRECOGNIZED_ERROR";
    }
}

bool BitcompStatusOk(bitcompResult_t status, const char* what, std::string* error) {
    if (status == BITCOMP_SUCCESS) {
        return true;
    }
    std::ostringstream oss;
    oss << what << " failed: " << BitcompResultToString(status) << " (" << static_cast<int>(status) << ")";
    *error = oss.str();
    return false;
}

struct BitcompContext {
    void* d_input = nullptr;
    void* d_output = nullptr;
    size_t* d_compressed_bytes = nullptr;

    size_t input_bytes = 0;
    size_t max_output_bytes = 0;

    bitcompHandle_t handle = nullptr;
    cudaStream_t stream = 0;
};

void FreeBitcompContext(BitcompContext* ctx) {
    if (ctx->handle != nullptr) {
        bitcompDestroyPlan(ctx->handle);
    }
    cudaFree(ctx->d_input);
    cudaFree(ctx->d_output);
    cudaFree(ctx->d_compressed_bytes);
    ctx->d_input = nullptr;
    ctx->d_output = nullptr;
    ctx->d_compressed_bytes = nullptr;
    ctx->input_bytes = 0;
    ctx->max_output_bytes = 0;
    ctx->handle = nullptr;
}

bool PrepareBitcompContext(const std::vector<__nv_bfloat16>& host_bf16, BitcompContext* ctx, std::string* error) {
    FreeBitcompContext(ctx);

    ctx->input_bytes = host_bf16.size() * sizeof(__nv_bfloat16);
    ctx->max_output_bytes = bitcompMaxBuflen(ctx->input_bytes);
    if (ctx->max_output_bytes == 0) {
        *error = "bitcompMaxBuflen returned zero";
        return false;
    }

    cudaError_t ce = cudaSuccess;
    ce = cudaMalloc(&ctx->d_input, ctx->input_bytes);
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(d_input) failed: ") + cudaGetErrorString(ce);
        FreeBitcompContext(ctx);
        return false;
    }

    ce = cudaMalloc(&ctx->d_output, ctx->max_output_bytes);
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(d_output) failed: ") + cudaGetErrorString(ce);
        FreeBitcompContext(ctx);
        return false;
    }

    ce = cudaMalloc(reinterpret_cast<void**>(&ctx->d_compressed_bytes), sizeof(size_t));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(d_compressed_bytes) failed: ") + cudaGetErrorString(ce);
        FreeBitcompContext(ctx);
        return false;
    }

    ce = cudaMemcpy(ctx->d_input, host_bf16.data(), ctx->input_bytes, cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
        *error = std::string("cudaMemcpy(H2D input) failed: ") + cudaGetErrorString(ce);
        FreeBitcompContext(ctx);
        return false;
    }

    bitcompResult_t status = bitcompCreatePlan(
        &ctx->handle,
        ctx->input_bytes,
        BITCOMP_UNSIGNED_16BIT,
        BITCOMP_LOSSLESS,
        BITCOMP_DEFAULT_ALGO);
    if (!BitcompStatusOk(status, "bitcompCreatePlan", error)) {
        FreeBitcompContext(ctx);
        return false;
    }

    status = bitcompSetStream(ctx->handle, ctx->stream);
    if (!BitcompStatusOk(status, "bitcompSetStream", error)) {
        FreeBitcompContext(ctx);
        return false;
    }

    return true;
}

bool LaunchBitcompCompress(BitcompContext* ctx, std::string* error) {
    // For fairness with ZipServ path, use single-buffer bitcomp compression per tensor.
    bitcompResult_t status = bitcompCompressLossless(ctx->handle, ctx->d_input, ctx->d_output);
    if (!BitcompStatusOk(status, "bitcompCompressLossless", error)) {
        return false;
    }
    return true;
}

bool QueryCompressedSizeAsync(BitcompContext* ctx, size_t* host_comp_bytes, std::string* error) {
    bitcompResult_t status = bitcompGetCompressedSizeAsync(ctx->d_output, ctx->d_compressed_bytes, ctx->stream);
    if (!BitcompStatusOk(status, "bitcompGetCompressedSizeAsync", error)) {
        return false;
    }

    cudaError_t ce = cudaMemcpyAsync(
        host_comp_bytes,
        ctx->d_compressed_bytes,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        ctx->stream);
    if (ce != cudaSuccess) {
        *error = std::string("cudaMemcpyAsync(d_compressed_bytes) failed: ") + cudaGetErrorString(ce);
        return false;
    }
    return true;
}

BackendTensorResult BenchmarkBitcomp(
    const ManifestTensor& tensor,
    const std::vector<__nv_bfloat16>& host_bf16,
    const ProgramOptions& opts) {
    BackendTensorResult row;
    row.backend = "bitcomp";
    row.tensor_name = tensor.name;
    row.layer_idx = tensor.layer;
    row.input_bytes = tensor.orig_bytes;

    BitcompContext ctx;
    std::string prep_error;
    if (!PrepareBitcompContext(host_bf16, &ctx, &prep_error)) {
        return MakeFailedResult("bitcomp", tensor, prep_error, tensor.orig_bytes);
    }

    if (opts.use_bitcomp_batch != 0 && opts.verbose) {
        std::cout << "[bitcomp] --use_bitcomp_batch=1 selected; this implementation keeps single-buffer API for fairness with ZipServ.\n";
    }

    for (int i = 0; i < opts.warmup; ++i) {
        std::string launch_error;
        if (!LaunchBitcompCompress(&ctx, &launch_error)) {
            FreeBitcompContext(&ctx);
            return MakeFailedResult("bitcomp", tensor, launch_error, tensor.orig_bytes);
        }
        cudaError_t ce = cudaStreamSynchronize(ctx.stream);
        if (ce != cudaSuccess) {
            FreeBitcompContext(&ctx);
            return MakeFailedResult("bitcomp", tensor, std::string("cudaStreamSynchronize warmup failed: ") + cudaGetErrorString(ce), tensor.orig_bytes);
        }
    }

    bench_common::CudaEventTimer timer;
    row.compress_write_ms.reserve(static_cast<size_t>(opts.iters));
    row.size_query_ms.reserve(static_cast<size_t>(opts.iters));
    row.total_ms.reserve(static_cast<size_t>(opts.iters));
    std::vector<size_t> compressed_bytes_samples;
    compressed_bytes_samples.reserve(static_cast<size_t>(opts.iters));

    for (int i = 0; i < opts.iters; ++i) {
        std::string timer_error;
        if (!timer.RecordStart(ctx.stream, &timer_error)) {
            FreeBitcompContext(&ctx);
            return MakeFailedResult("bitcomp", tensor, timer_error, tensor.orig_bytes);
        }

        std::string launch_error;
        if (!LaunchBitcompCompress(&ctx, &launch_error)) {
            FreeBitcompContext(&ctx);
            return MakeFailedResult("bitcomp", tensor, launch_error, tensor.orig_bytes);
        }

        if (!timer.RecordStop(ctx.stream, &timer_error)) {
            FreeBitcompContext(&ctx);
            return MakeFailedResult("bitcomp", tensor, timer_error, tensor.orig_bytes);
        }
        if (!timer.SyncStop(&timer_error)) {
            FreeBitcompContext(&ctx);
            return MakeFailedResult("bitcomp", tensor, timer_error, tensor.orig_bytes);
        }

        float compress_ms_f = 0.0f;
        if (!timer.ElapsedMs(&compress_ms_f, &timer_error)) {
            FreeBitcompContext(&ctx);
            return MakeFailedResult("bitcomp", tensor, timer_error, tensor.orig_bytes);
        }
        if (opts.stream_sync != 0) {
            cudaError_t ce = cudaStreamSynchronize(ctx.stream);
            if (ce != cudaSuccess) {
                FreeBitcompContext(&ctx);
                return MakeFailedResult("bitcomp", tensor, std::string("cudaStreamSynchronize after compress failed: ") + cudaGetErrorString(ce), tensor.orig_bytes);
            }
        }

        double size_query_ms = 0.0;
        size_t host_comp_bytes = 0;
        if (opts.bitcomp_include_size_query != 0) {
            if (!timer.RecordStart(ctx.stream, &timer_error)) {
                FreeBitcompContext(&ctx);
                return MakeFailedResult("bitcomp", tensor, timer_error, tensor.orig_bytes);
            }
            if (!QueryCompressedSizeAsync(&ctx, &host_comp_bytes, &timer_error)) {
                FreeBitcompContext(&ctx);
                return MakeFailedResult("bitcomp", tensor, timer_error, tensor.orig_bytes);
            }
            if (!timer.RecordStop(ctx.stream, &timer_error)) {
                FreeBitcompContext(&ctx);
                return MakeFailedResult("bitcomp", tensor, timer_error, tensor.orig_bytes);
            }
            if (!timer.SyncStop(&timer_error)) {
                FreeBitcompContext(&ctx);
                return MakeFailedResult("bitcomp", tensor, timer_error, tensor.orig_bytes);
            }

            float size_ms_f = 0.0f;
            if (!timer.ElapsedMs(&size_ms_f, &timer_error)) {
                FreeBitcompContext(&ctx);
                return MakeFailedResult("bitcomp", tensor, timer_error, tensor.orig_bytes);
            }
            size_query_ms = static_cast<double>(size_ms_f);
            if (opts.stream_sync != 0) {
                cudaError_t ce = cudaStreamSynchronize(ctx.stream);
                if (ce != cudaSuccess) {
                    FreeBitcompContext(&ctx);
                    return MakeFailedResult("bitcomp", tensor, std::string("cudaStreamSynchronize after size query failed: ") + cudaGetErrorString(ce), tensor.orig_bytes);
                }
            }
        } else {
            if (!QueryCompressedSizeAsync(&ctx, &host_comp_bytes, &timer_error)) {
                FreeBitcompContext(&ctx);
                return MakeFailedResult("bitcomp", tensor, timer_error, tensor.orig_bytes);
            }
            cudaError_t ce = cudaStreamSynchronize(ctx.stream);
            if (ce != cudaSuccess) {
                FreeBitcompContext(&ctx);
                return MakeFailedResult("bitcomp", tensor, std::string("cudaStreamSynchronize for size readback failed: ") + cudaGetErrorString(ce), tensor.orig_bytes);
            }
        }

        if (host_comp_bytes == 0) {
            FreeBitcompContext(&ctx);
            return MakeFailedResult("bitcomp", tensor, "bitcomp_compressed_bytes_zero", tensor.orig_bytes);
        }

        const double compress_ms = static_cast<double>(compress_ms_f);
        row.compress_write_ms.push_back(compress_ms);
        row.size_query_ms.push_back(size_query_ms);
        row.total_ms.push_back(compress_ms + size_query_ms);
        compressed_bytes_samples.push_back(host_comp_bytes);
    }

    row.compressed_bytes = MeanBytesRounded(compressed_bytes_samples);
    FreeBitcompContext(&ctx);
    return row;
}

#else

BackendTensorResult BenchmarkBitcomp(
    const ManifestTensor& tensor,
    const std::vector<__nv_bfloat16>&,
    const ProgramOptions&) {
    return MakeFailedResult(
        "bitcomp",
        tensor,
        "nvCOMP bitcomp is unavailable. Rebuild with NVCOMP_ROOT and nvcomp headers/libs.",
        tensor.orig_bytes);
}

#endif

BackendTensorResult BuildAggregateResult(const std::vector<BackendTensorResult>& rows, const std::string& backend) {
    BackendTensorResult agg;
    agg.backend = backend;
    agg.tensor_name = "AGGREGATE";
    agg.layer_idx = -1;

    std::vector<const BackendTensorResult*> ok_rows;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].status == "ok" && rows[i].backend == backend) {
            ok_rows.push_back(&rows[i]);
        }
    }
    if (ok_rows.empty()) {
        agg.status = "failed";
        agg.note = "no_success_rows";
        return agg;
    }

    const size_t iters = ok_rows[0]->total_ms.size();
    if (iters == 0) {
        agg.status = "failed";
        agg.note = "empty_metric_samples";
        return agg;
    }

    agg.compress_write_ms.assign(iters, 0.0);
    agg.size_query_ms.assign(iters, 0.0);
    agg.total_ms.assign(iters, 0.0);

    for (size_t r = 0; r < ok_rows.size(); ++r) {
        const BackendTensorResult* row = ok_rows[r];
        if (row->total_ms.size() != iters || row->compress_write_ms.size() != iters || row->size_query_ms.size() != iters) {
            agg.status = "failed";
            agg.note = "inconsistent_iteration_count";
            return agg;
        }

        agg.input_bytes += row->input_bytes;
        agg.compressed_bytes += row->compressed_bytes;

        for (size_t i = 0; i < iters; ++i) {
            agg.compress_write_ms[i] += row->compress_write_ms[i];
            agg.size_query_ms[i] += row->size_query_ms[i];
            agg.total_ms[i] += row->total_ms[i];
        }
    }

    agg.status = "ok";
    std::ostringstream note;
    note << "aggregate_over_tensors=" << ok_rows.size();
    agg.note = note.str();
    return agg;
}

int main(int argc, char** argv) {
    ProgramOptions opts;
    if (!ParseOptions(argc, argv, &opts)) {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            return 0;
        }
        return 1;
    }

#if !ZIPSERV_NVCOMP_BITCOMP_ENABLED
    std::cerr << "bitcomp backend unavailable: nvCOMP headers/libs not found at build time.\n";
    std::cerr << "Hint: set NVCOMP_ROOT and rebuild, e.g. make NVCOMP_ROOT=/path/to/nvcomp bench_llama31_70b_zipserv_vs_bitcomp_compress\n";
    return 1;
#endif

    const std::string manifest_path = ResolveManifestPath(opts);

    cudaError_t ce = cudaSetDevice(opts.device);
    if (ce != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(ce) << "\n";
        return 1;
    }

    std::vector<ManifestTensor> all_tensors;
    std::string manifest_error;
    if (!LoadManifest(manifest_path, &all_tensors, &manifest_error)) {
        std::cerr << manifest_error << "\n";
        return 1;
    }

    bool use_regex_filter = false;
    std::regex compiled_filter;
    std::string filter_note;
    if (!BuildWeightFilterRegex(opts.weight_filter, &use_regex_filter, &compiled_filter, &filter_note)) {
        std::cerr << "Failed to parse weight filter.\n";
        return 1;
    }
    if (!filter_note.empty()) {
        std::cerr << "[warning] " << filter_note << "\n";
    }

    std::vector<ManifestTensor> selected = SelectTensors(all_tensors, opts, use_regex_filter, compiled_filter);
    if (selected.empty()) {
        std::cerr << "No tensors selected. Check --layer_idx / --weight_filter / --max_tensors.\n";
        return 1;
    }

    std::ofstream ofs(opts.csv_path.c_str(), std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open CSV output: " << opts.csv_path << "\n";
        return 1;
    }
    WriteCSVHeader(&ofs);

    std::cout << "===== Llama 3.1 70B BF16 ZipServ vs bitcomp Compression Benchmark =====\n";
    std::cout << "model_path=" << opts.model_path << "\n";
    std::cout << "manifest=" << manifest_path << "\n";
    std::cout << "selected_tensors=" << selected.size() << "\n";
    std::cout << "warmup=" << opts.warmup << ", iters=" << opts.iters << "\n";
    std::cout << "device=" << opts.device << ", stream_sync=" << opts.stream_sync << "\n";
    std::cout << "bitcomp_include_size_query=" << opts.bitcomp_include_size_query
              << ", use_bitcomp_batch=" << opts.use_bitcomp_batch << "\n";
    std::cout << "csv=" << opts.csv_path << "\n\n";

    if (!opts.aggregate_only) {
        PrintHeader();
    }

    std::vector<BackendTensorResult> all_rows;
    all_rows.reserve(selected.size() * 2);

    size_t ok_count = 0;
    size_t failed_count = 0;

    for (size_t i = 0; i < selected.size(); ++i) {
        const ManifestTensor& tensor = selected[i];

        std::string validation_error;
        if (!ValidateTensor(tensor, &validation_error)) {
            BackendTensorResult z_fail = MakeFailedResult("zipserv", tensor, validation_error, tensor.orig_bytes);
            BackendTensorResult b_fail = MakeFailedResult("bitcomp", tensor, validation_error, tensor.orig_bytes);
            all_rows.push_back(z_fail);
            all_rows.push_back(b_fail);
            if (!opts.aggregate_only) {
                WriteCSVRow(&ofs, z_fail, opts);
                WriteCSVRow(&ofs, b_fail, opts);
            }
            if (!opts.aggregate_only) {
                PrintRow(z_fail);
                PrintRow(b_fail);
            }
            failed_count += 2;
            continue;
        }

        std::vector<__nv_bfloat16> host_bf16;
        std::string read_error;
        if (!ReadTensorAsBF16(tensor, &host_bf16, &read_error)) {
            BackendTensorResult z_fail = MakeFailedResult("zipserv", tensor, read_error, tensor.orig_bytes);
            BackendTensorResult b_fail = MakeFailedResult("bitcomp", tensor, read_error, tensor.orig_bytes);
            all_rows.push_back(z_fail);
            all_rows.push_back(b_fail);
            if (!opts.aggregate_only) {
                WriteCSVRow(&ofs, z_fail, opts);
                WriteCSVRow(&ofs, b_fail, opts);
            }
            if (!opts.aggregate_only) {
                PrintRow(z_fail);
                PrintRow(b_fail);
            }
            failed_count += 2;
            continue;
        }

        BackendTensorResult zip_row = BenchmarkZipServ(tensor, host_bf16, opts);
        BackendTensorResult bitcomp_row = BenchmarkBitcomp(tensor, host_bf16, opts);

        all_rows.push_back(zip_row);
        all_rows.push_back(bitcomp_row);
        if (!opts.aggregate_only) {
            WriteCSVRow(&ofs, zip_row, opts);
            WriteCSVRow(&ofs, bitcomp_row, opts);
        }

        if (!opts.aggregate_only) {
            PrintRow(zip_row);
            PrintRow(bitcomp_row);
        }

        if (zip_row.status == "ok") {
            ++ok_count;
        } else {
            ++failed_count;
        }
        if (bitcomp_row.status == "ok") {
            ++ok_count;
        } else {
            ++failed_count;
        }

        if (opts.verbose) {
            std::cout << "[progress] " << (i + 1) << "/" << selected.size() << " tensors processed\n";
        }
    }

    BackendTensorResult zip_agg = BuildAggregateResult(all_rows, "zipserv");
    BackendTensorResult bitcomp_agg = BuildAggregateResult(all_rows, "bitcomp");

    std::cout << "\n===== Aggregate =====\n";
    PrintHeader();
    PrintRow(zip_agg);
    PrintRow(bitcomp_agg);

    WriteCSVRow(&ofs, zip_agg, opts);
    WriteCSVRow(&ofs, bitcomp_agg, opts);

    ofs.close();

    std::cout << "\n===== Summary =====\n";
    std::cout << "backend_rows_ok=" << ok_count << ", backend_rows_failed=" << failed_count << "\n";
    std::cout << "CSV written to: " << opts.csv_path << "\n";
    return (failed_count == 0) ? 0 : 1;
}
