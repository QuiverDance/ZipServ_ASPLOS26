#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <omp.h>
#include <immintrin.h>

#include "L_API.cuh"
#include "bench_manifest_utils.h"
#include "csv_writer.h"
#include "cuda_timer.h"
#include "./utils.h"

struct ProgramOptions {
    std::string input_dir = "~/saved_kv_cache";
    int warmup = 10;
    int iters = 100;
    int max_files = -1;
    int recursive = 0;
    int verify = 1;
    std::string out_csv = "kv_cache_zipserv_results.csv";
    std::string ext_filter;
    int device = 0;
    int print_skipped = 1;
};

struct NpyInfo {
    int major = 0;
    int minor = 0;
    std::string descr;
    bool fortran_order = false;
    std::vector<int64_t> shape;
    size_t data_offset = 0;
    size_t data_bytes = 0;
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

// Pre-allocated workspace for buffer-reuse compression (zero malloc/free per call)
struct CompressReusableBuffers {
    // Output buffers (allocated at max possible size)
    uint8_t* sign_mantissa = nullptr;
    __nv_bfloat16* compressed_full = nullptr;
    uint64_t* bitmap1 = nullptr;
    uint64_t* bitmap2 = nullptr;
    uint64_t* bitmap3 = nullptr;
    int* tile_offsets = nullptr;
    int* tile_offsets_median = nullptr;
    int* tile_offsets_global = nullptr;

    // Temp workspace
    uint8_t* temp_sm = nullptr;
    __nv_bfloat16* temp_full = nullptr;
    int* gt_hf_count = nullptr;
    int* gt_full_count = nullptr;
    int* hf_offsets = nullptr;
    int* full_offsets = nullptr;

    // Dimensions used for allocation
    int alloc_rows = 0;
    int alloc_cols = 0;
    int num_global_tiles = 0;
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

struct MetricStats {
    bool valid = false;
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct BenchRow {
    std::string file_path;
    std::string file_name;
    std::string tensor_label;
    int tensor_id = -1;

    std::string shape3d;
    std::string mapped_shape;
    std::string padded_shape;

    size_t logical_bytes = 0;
    size_t input_bytes = 0;
    size_t compressed_bytes = 0;

    size_t logical_numel = 0;
    size_t input_numel = 0;
    size_t pad_elems = 0;
    double pad_ratio = 0.0;

    double convert_ms = 0.0;
    double compress_ms = 0.0;
    double decompress_ms = 0.0;

    double ratio_logical = 0.0;
    double ratio_input = 0.0;
    double compress_gbps_logical = 0.0;
    double decompress_gbps_logical = 0.0;
    double top7_raw_cover = 0.0;
    double top7_selected_cover = 0.0;
    std::string top7_raw;
    std::string top7_selected;

    size_t verify_mismatch = 0;
    std::string status = "ok";
    std::string note;
};

struct AggregateSummary {
    size_t total = 0;
    size_t ok = 0;
    size_t skipped = 0;
    size_t failed = 0;

    size_t total_logical_bytes = 0;
    size_t total_input_bytes = 0;
    size_t total_compressed_bytes = 0;

    double total_ratio_logical = 0.0;
    double total_ratio_input = 0.0;

    MetricStats ratio_logical;
    MetricStats ratio_input;
    MetricStats pad_ratio;
    MetricStats convert_ms;
    MetricStats compress_ms;
    MetricStats decompress_ms;
    MetricStats compress_gbps_logical;
    MetricStats decompress_gbps_logical;
};

void PrintUsage() {
    std::cout
        << "Usage: ./bench_kv_cache_zipserv_compress [options]\n"
        << "Options:\n"
        << "  --input_dir <path>      Input directory or single file (default: ~/saved_kv_cache)\n"
        << "  --warmup <int>          Warmup iterations (default: 10)\n"
        << "  --iters <int>           Benchmark iterations (default: 100)\n"
        << "  --max_files <int>       Max files after sorting (-1: all, default: -1)\n"
        << "  --recursive <0|1>       Recursive directory traversal (default: 0)\n"
        << "  --verify <0|1>          Bit-exact verify after decompress (default: 1)\n"
        << "  --out_csv <path>        Output CSV path (default: kv_cache_zipserv_results.csv)\n"
        << "  --ext_filter <csv>      Extension allow-list without dot, e.g. npy,noext\n"
        << "  --device <int>          CUDA device index (default: 0)\n"
        << "  --print_skipped <0|1>   Print skipped rows to console (default: 1)\n"
        << "  --help                  Show help\n";
}

bool ParseBool01(const std::string& name, const std::string& value, int* out) {
    if (value == "0") {
        *out = 0;
        return true;
    }
    if (value == "1") {
        *out = 1;
        return true;
    }
    std::cerr << "Error: " << name << " must be 0 or 1, got: " << value << "\n";
    return false;
}

bool ParseOptions(int argc, char** argv, ProgramOptions* opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            PrintUsage();
            return false;
        } else if (arg == "--input_dir" && i + 1 < argc) {
            opts->input_dir = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts->warmup = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            opts->iters = std::atoi(argv[++i]);
        } else if (arg == "--max_files" && i + 1 < argc) {
            opts->max_files = std::atoi(argv[++i]);
        } else if (arg == "--recursive" && i + 1 < argc) {
            int value = 0;
            if (!ParseBool01("--recursive", argv[++i], &value)) {
                return false;
            }
            opts->recursive = value;
        } else if (arg == "--verify" && i + 1 < argc) {
            int value = 0;
            if (!ParseBool01("--verify", argv[++i], &value)) {
                return false;
            }
            opts->verify = value;
        } else if (arg == "--out_csv" && i + 1 < argc) {
            opts->out_csv = argv[++i];
        } else if (arg == "--ext_filter" && i + 1 < argc) {
            opts->ext_filter = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            opts->device = std::atoi(argv[++i]);
        } else if (arg == "--print_skipped" && i + 1 < argc) {
            int value = 0;
            if (!ParseBool01("--print_skipped", argv[++i], &value)) {
                return false;
            }
            opts->print_skipped = value;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            PrintUsage();
            return false;
        }
    }

    if (opts->warmup < 0 || opts->iters <= 0) {
        std::cerr << "Error: warmup must be >= 0 and iters must be > 0.\n";
        return false;
    }
    if (opts->max_files == 0 || opts->max_files < -1) {
        std::cerr << "Error: --max_files must be -1 or > 0.\n";
        return false;
    }
    if (opts->device < 0) {
        std::cerr << "Error: --device must be >= 0.\n";
        return false;
    }
    return true;
}

std::string ExpandUserPath(const std::string& input) {
    if (input.empty() || input[0] != '~') {
        return input;
    }
    const char* home = std::getenv("HOME");
    if (home == nullptr) {
        return input;
    }
    if (input.size() == 1) {
        return std::string(home);
    }
    if (input[1] == '/') {
        return std::string(home) + input.substr(1);
    }
    return input;
}

std::string ToLower(const std::string& s) {
    std::string out = s;
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(out[i])));
    }
    return out;
}

std::string Basename(const std::string& path) {
    const size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

std::string GetExtensionNoDotLower(const std::string& path) {
    const std::string base = Basename(path);
    const size_t dot = base.find_last_of('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 >= base.size()) {
        return "";
    }
    return ToLower(base.substr(dot + 1));
}

std::set<std::string> ParseExtensionFilter(const std::string& spec) {
    std::set<std::string> exts;
    const std::vector<std::string> tokens = bench_common::SplitByComma(spec);
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string t = ToLower(bench_common::Trim(tokens[i]));
        if (t.empty()) {
            continue;
        }
        if (!t.empty() && t[0] == '.') {
            t = t.substr(1);
        }
        if (t == "noext" || t == "(noext)") {
            exts.insert("");
        } else {
            exts.insert(t);
        }
    }
    return exts;
}

bool FilePassesExtensionFilter(const std::string& path, const std::set<std::string>& ext_filter) {
    if (ext_filter.empty()) {
        return true;
    }
    const std::string ext = GetExtensionNoDotLower(path);
    return ext_filter.find(ext) != ext_filter.end();
}

bool IsDirectory(const std::string& path, std::string* error) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        if (error != nullptr) {
            std::ostringstream oss;
            oss << "stat failed for " << path << ": " << std::strerror(errno);
            *error = oss.str();
        }
        return false;
    }
    return S_ISDIR(st.st_mode);
}

bool CollectInputFiles(const ProgramOptions& opts,
                       const std::set<std::string>& ext_filter,
                       std::vector<std::string>* files,
                       std::string* error) {
    files->clear();
    const std::string root = ExpandUserPath(opts.input_dir);

    struct stat root_st;
    if (stat(root.c_str(), &root_st) != 0) {
        std::ostringstream oss;
        oss << "Failed to access input_dir: " << root << " (" << std::strerror(errno) << ")";
        *error = oss.str();
        return false;
    }

    if (S_ISREG(root_st.st_mode)) {
        if (FilePassesExtensionFilter(root, ext_filter)) {
            files->push_back(root);
        }
    } else if (S_ISDIR(root_st.st_mode)) {
        std::vector<std::string> dir_stack;
        dir_stack.push_back(root);

        while (!dir_stack.empty()) {
            const std::string dir = dir_stack.back();
            dir_stack.pop_back();

            DIR* dp = opendir(dir.c_str());
            if (dp == nullptr) {
                std::ostringstream oss;
                oss << "opendir failed for " << dir << " (" << std::strerror(errno) << ")";
                *error = oss.str();
                return false;
            }

            struct dirent* ent = nullptr;
            while ((ent = readdir(dp)) != nullptr) {
                const std::string name = ent->d_name;
                if (name == "." || name == "..") {
                    continue;
                }
                const std::string full = dir + "/" + name;

                struct stat st;
                if (stat(full.c_str(), &st) != 0) {
                    continue;
                }
                if (S_ISREG(st.st_mode)) {
                    if (FilePassesExtensionFilter(full, ext_filter)) {
                        files->push_back(full);
                    }
                } else if (S_ISDIR(st.st_mode) && opts.recursive != 0) {
                    dir_stack.push_back(full);
                }
            }
            closedir(dp);
        }
    } else {
        *error = "input_dir is neither a regular file nor a directory";
        return false;
    }

    std::sort(files->begin(), files->end());
    if (opts.max_files > 0 && static_cast<int>(files->size()) > opts.max_files) {
        files->resize(static_cast<size_t>(opts.max_files));
    }
    return true;
}

bool SafeMulSize(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

bool ReadLittleU16(const std::vector<uint8_t>& raw, size_t off, uint16_t* out) {
    if (off + 2 > raw.size()) {
        return false;
    }
    *out = static_cast<uint16_t>(raw[off]) |
           static_cast<uint16_t>(raw[off + 1] << 8);
    return true;
}

bool ReadLittleU32(const std::vector<uint8_t>& raw, size_t off, uint32_t* out) {
    if (off + 4 > raw.size()) {
        return false;
    }
    *out = static_cast<uint32_t>(raw[off]) |
           (static_cast<uint32_t>(raw[off + 1]) << 8) |
           (static_cast<uint32_t>(raw[off + 2]) << 16) |
           (static_cast<uint32_t>(raw[off + 3]) << 24);
    return true;
}

bool LocateNpyValue(const std::string& header, const std::string& key, size_t* value_pos) {
    const std::string key1 = "'" + key + "'";
    const std::string key2 = "\"" + key + "\"";

    size_t key_pos = header.find(key1);
    if (key_pos == std::string::npos) {
        key_pos = header.find(key2);
    }
    if (key_pos == std::string::npos) {
        return false;
    }

    const size_t colon = header.find(':', key_pos);
    if (colon == std::string::npos) {
        return false;
    }

    size_t pos = colon + 1;
    while (pos < header.size() && std::isspace(static_cast<unsigned char>(header[pos]))) {
        ++pos;
    }
    if (pos >= header.size()) {
        return false;
    }
    *value_pos = pos;
    return true;
}

bool ParseQuotedStringValue(const std::string& header, const std::string& key, std::string* out) {
    size_t pos = 0;
    if (!LocateNpyValue(header, key, &pos)) {
        return false;
    }
    const char quote = header[pos];
    if (quote != '\'' && quote != '"') {
        return false;
    }
    ++pos;

    std::string value;
    while (pos < header.size()) {
        const char c = header[pos++];
        if (c == '\\') {
            if (pos >= header.size()) {
                return false;
            }
            value.push_back(header[pos++]);
            continue;
        }
        if (c == quote) {
            *out = value;
            return true;
        }
        value.push_back(c);
    }
    return false;
}

bool ParseBoolValue(const std::string& header, const std::string& key, bool* out) {
    size_t pos = 0;
    if (!LocateNpyValue(header, key, &pos)) {
        return false;
    }
    if (header.compare(pos, 4, "True") == 0) {
        *out = true;
        return true;
    }
    if (header.compare(pos, 5, "False") == 0) {
        *out = false;
        return true;
    }
    return false;
}

bool ParseShapeValue(const std::string& header, const std::string& key, std::vector<int64_t>* shape) {
    size_t pos = 0;
    if (!LocateNpyValue(header, key, &pos)) {
        return false;
    }
    if (header[pos] != '(') {
        return false;
    }

    const size_t end = header.find(')', pos + 1);
    if (end == std::string::npos) {
        return false;
    }

    const std::string tuple_content = header.substr(pos + 1, end - (pos + 1));
    std::stringstream ss(tuple_content);
    std::string token;
    shape->clear();

    while (std::getline(ss, token, ',')) {
        token = bench_common::Trim(token);
        if (token.empty()) {
            continue;
        }
        int64_t dim = 0;
        try {
            dim = std::stoll(token);
        } catch (...) {
            return false;
        }
        shape->push_back(dim);
    }

    return !shape->empty();
}

bool ParseNpyInfo(const std::vector<uint8_t>& raw, NpyInfo* info, std::string* error) {
    const uint8_t kMagic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (raw.size() < 10) {
        *error = "npy file too small";
        return false;
    }
    for (int i = 0; i < 6; ++i) {
        if (raw[static_cast<size_t>(i)] != kMagic[i]) {
            *error = "invalid npy magic";
            return false;
        }
    }

    info->major = raw[6];
    info->minor = raw[7];

    uint32_t header_len = 0;
    size_t header_offset = 0;
    if (info->major == 1) {
        uint16_t header_len16 = 0;
        if (!ReadLittleU16(raw, 8, &header_len16)) {
            *error = "failed to read npy v1 header length";
            return false;
        }
        header_len = header_len16;
        header_offset = 10;
    } else if (info->major == 2) {
        if (!ReadLittleU32(raw, 8, &header_len)) {
            *error = "failed to read npy v2 header length";
            return false;
        }
        header_offset = 12;
    } else {
        std::ostringstream oss;
        oss << "unsupported npy major version: " << info->major;
        *error = oss.str();
        return false;
    }

    size_t header_end = 0;
    if (!SafeMulSize(1, static_cast<size_t>(header_len), &header_end)) {
        *error = "npy header length overflow";
        return false;
    }
    header_end += header_offset;

    if (header_end > raw.size()) {
        *error = "npy header exceeds file size";
        return false;
    }

    const std::string header(reinterpret_cast<const char*>(raw.data() + header_offset),
                             static_cast<size_t>(header_len));

    if (!ParseQuotedStringValue(header, "descr", &info->descr)) {
        *error = "npy header missing descr";
        return false;
    }
    if (!ParseBoolValue(header, "fortran_order", &info->fortran_order)) {
        *error = "npy header missing fortran_order";
        return false;
    }
    if (!ParseShapeValue(header, "shape", &info->shape)) {
        *error = "npy header missing/invalid shape";
        return false;
    }

    info->data_offset = header_end;
    info->data_bytes = raw.size() - header_end;
    return true;
}

bool IsSupportedFloat16Descr(const std::string& descr) {
    // ZipServ benchmark path expects little-endian fp16 source.
    return descr == "<f2" || descr == "=f2" || descr == "f2" || descr == "|f2";
}

int RoundUpTo64(int x) {
    return ((x + 63) / 64) * 64;
}

std::string ShapeToString64(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::string FormatExponentList(const std::array<int, 7>& exponents) {
    std::ostringstream oss;
    for (size_t i = 0; i < exponents.size(); ++i) {
        if (i > 0) {
            oss << "/";
        }
        if (exponents[i] >= 0) {
            oss << exponents[i];
        } else {
            oss << "-";
        }
    }
    return oss.str();
}

// Lightweight coverage using pre-built histogram (avoids re-scanning data)
void CollectExponentCoverageFromHistogram(
    const int* exponent_counts,
    size_t total_elements,
    const std::array<int, 7>& selected_exponents,
    std::array<int, 7>* raw_top_exponents,
    double* raw_cover,
    double* selected_cover) {
    std::vector<std::pair<int, int>> sorted;
    sorted.reserve(256);
    for (int e = 0; e < 256; ++e) {
        if (exponent_counts[e] > 0) {
            sorted.push_back(std::make_pair(e, exponent_counts[e]));
        }
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;
              });

    raw_top_exponents->fill(-1);
    for (size_t i = 0; i < raw_top_exponents->size() && i < sorted.size(); ++i) {
        (*raw_top_exponents)[i] = sorted[i].first;
    }

    bool raw_mask[256] = {false};
    bool selected_mask[256] = {false};
    for (size_t i = 0; i < 7; ++i) {
        if ((*raw_top_exponents)[i] >= 0) raw_mask[(*raw_top_exponents)[i]] = true;
        if (selected_exponents[i] >= 0 && selected_exponents[i] < 256)
            selected_mask[selected_exponents[i]] = true;
    }

    size_t raw_count = 0, sel_count = 0;
    for (int e = 0; e < 256; ++e) {
        if (raw_mask[e]) raw_count += static_cast<size_t>(exponent_counts[e]);
        if (selected_mask[e]) sel_count += static_cast<size_t>(exponent_counts[e]);
    }

    *raw_cover = (total_elements > 0)
                     ? static_cast<double>(raw_count) / static_cast<double>(total_elements)
                     : 0.0;
    *selected_cover = (total_elements > 0)
                          ? static_cast<double>(sel_count) / static_cast<double>(total_elements)
                          : 0.0;
}

// Fused FP16->BF16 conversion + exponent analysis in a single SIMD-accelerated pass.
// Returns the exponent histogram for CollectExponentCoverageFromHistogram.
bool ConvertFp16ToBf16WithAnalysis(const NpyInfo& info,
                                   const std::vector<uint8_t>& raw,
                                   int64_t seq_begin,
                                   int64_t seq_count,
                                   std::vector<__nv_bfloat16>* out,
                                   int* mapped_rows_out,
                                   int* mapped_cols_out,
                                   int* padded_rows_out,
                                   int* padded_cols_out,
                                   size_t* logical_numel,
                                   size_t* input_numel,
                                   double* convert_ms,
                                   int* top_exponents,
                                   int* exponent_counts_out,
                                   std::string* error) {
    if (info.shape.size() != 3) {
        *error = "non_3d_shape";
        return false;
    }

    const int64_t t = info.shape[0];
    const int64_t h = info.shape[1];
    const int64_t d = info.shape[2];
    if (t <= 0 || h <= 0 || d <= 0) {
        *error = "shape_has_non_positive_dimension";
        return false;
    }
    if (h > std::numeric_limits<int>::max() || d > std::numeric_limits<int>::max()) {
        *error = "shape_dimension_too_large";
        return false;
    }
    if (seq_begin < 0 || seq_count <= 0 || seq_begin + seq_count > t) {
        *error = "invalid_seq_chunk_range";
        return false;
    }

    const int64_t rows64 = seq_count * h;
    if (rows64 > std::numeric_limits<int>::max()) {
        *error = "mapped_rows_overflow";
        return false;
    }

    const int mapped_rows = static_cast<int>(rows64);
    const int mapped_cols = static_cast<int>(d);
    *mapped_rows_out = mapped_rows;
    *mapped_cols_out = mapped_cols;
    *padded_rows_out = RoundUpTo64(mapped_rows);
    *padded_cols_out = RoundUpTo64(mapped_cols);
    const int padded_rows = *padded_rows_out;
    const int padded_cols = *padded_cols_out;

    size_t logical_numel_local = 0;
    size_t input_numel_local = 0;
    if (!SafeMulSize(static_cast<size_t>(mapped_rows), static_cast<size_t>(mapped_cols), &logical_numel_local) ||
        !SafeMulSize(static_cast<size_t>(padded_rows), static_cast<size_t>(padded_cols), &input_numel_local)) {
        *error = "numel_overflow";
        return false;
    }

    size_t full_numel = 0;
    size_t full_numel_tmp = 0;
    if (!SafeMulSize(static_cast<size_t>(t), static_cast<size_t>(h), &full_numel_tmp) ||
        !SafeMulSize(full_numel_tmp, static_cast<size_t>(d), &full_numel)) {
        *error = "full_numel_overflow";
        return false;
    }

    const size_t expected_fp16_bytes = full_numel * sizeof(uint16_t);
    if (info.data_bytes != expected_fp16_bytes) {
        std::ostringstream oss;
        oss << "npy_data_bytes_mismatch(expected=" << expected_fp16_bytes
            << ",got=" << info.data_bytes << ")";
        *error = oss.str();
        return false;
    }

    out->assign(input_numel_local, __float2bfloat16(0.0f));

    const uint16_t* fp16_bits = reinterpret_cast<const uint16_t*>(raw.data() + info.data_offset);
    const size_t seq_row_base = static_cast<size_t>(seq_begin) * static_cast<size_t>(h);
    uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());

    // Zero histogram
    memset(exponent_counts_out, 0, 256 * sizeof(int));

    const std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();

    // SIMD constants
    const __m256i v_round_bias = _mm256_set1_epi32(0x7FFF);
    const __m256i v_one_32 = _mm256_set1_epi32(1);
    const __m128i v_0xFF = _mm_set1_epi16(0xFF);
    const int simd_width = 8;
    const int aligned_cols = mapped_cols & ~(simd_width - 1);

    for (int r = 0; r < mapped_rows; ++r) {
        const size_t global_row = seq_row_base + static_cast<size_t>(r);
        const uint16_t* src_row = fp16_bits + global_row * static_cast<size_t>(mapped_cols);
        uint16_t* dst_row = out_data + static_cast<size_t>(r) * static_cast<size_t>(padded_cols);

        int c = 0;
        // SIMD path: 8 FP16 -> 8 BF16 per iteration using F16C
        for (; c < aligned_cols; c += simd_width) {
            // 1) Load 8 x FP16
            __m128i fp16_8 = _mm_loadu_si128((const __m128i*)(src_row + c));
            // 2) FP16 -> FP32 via F16C
            __m256 fp32_8 = _mm256_cvtph_ps(fp16_8);
            __m256i fp32_bits = _mm256_castps_si256(fp32_8);
            // 3) FP32 -> BF16 with round-to-nearest-even
            __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(fp32_bits, 16), v_one_32);
            __m256i rounded = _mm256_add_epi32(fp32_bits, _mm256_add_epi32(v_round_bias, lsb));
            __m256i bf16_32 = _mm256_srli_epi32(rounded, 16);
            // 4) Pack 8 x 32-bit -> 8 x 16-bit
            __m256i packed = _mm256_packus_epi32(bf16_32, _mm256_setzero_si256());
            __m128i lo = _mm256_castsi256_si128(packed);
            __m128i hi = _mm256_extracti128_si256(packed, 1);
            __m128i bf16_8 = _mm_unpacklo_epi64(lo, hi);
            // 5) Store 8 BF16
            _mm_storeu_si128((__m128i*)(dst_row + c), bf16_8);
            // 6) Extract exponents and update histogram
            __m128i exp_8 = _mm_and_si128(_mm_srli_epi16(bf16_8, 7), v_0xFF);
            uint16_t exp_arr[8];
            _mm_storeu_si128((__m128i*)exp_arr, exp_8);
            exponent_counts_out[exp_arr[0]]++;
            exponent_counts_out[exp_arr[1]]++;
            exponent_counts_out[exp_arr[2]]++;
            exponent_counts_out[exp_arr[3]]++;
            exponent_counts_out[exp_arr[4]]++;
            exponent_counts_out[exp_arr[5]]++;
            exponent_counts_out[exp_arr[6]]++;
            exponent_counts_out[exp_arr[7]]++;
        }
        // Scalar tail
        for (; c < mapped_cols; ++c) {
            __half_raw hr;
            hr.x = src_row[c];
            const __half hval = hr;
            const __nv_bfloat16 bf = __float2bfloat16(__half2float(hval));
            dst_row[c] = __bfloat16_as_ushort(bf);
            exponent_counts_out[(dst_row[c] >> 7) & 0xFF]++;
        }
    }

    // Account for padding zeros in histogram (BF16 0.0 has exponent 0)
    const int padding_count = padded_rows * padded_cols - mapped_rows * mapped_cols;
    exponent_counts_out[0] += padding_count;

    const std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    *convert_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    *logical_numel = logical_numel_local;
    *input_numel = input_numel_local;

    // Analyze exponent distribution: find top-7 contiguous exponent range
    const int top_n = 7;
    int original_top[7];
    bool used[256] = {false};
    int n_found = 0;
    for (int ti = 0; ti < top_n; ++ti) {
        int best_exp = -1;
        int best_count = 0;
        for (int e = 0; e < 256; ++e) {
            if (!used[e] && exponent_counts_out[e] > best_count) {
                best_count = exponent_counts_out[e];
                best_exp = e;
            }
        }
        if (best_exp >= 0) {
            original_top[n_found++] = best_exp;
            used[best_exp] = true;
        }
    }
    while (n_found < top_n) {
        original_top[n_found] = 127 - n_found;
        n_found++;
    }
    std::sort(original_top, original_top + top_n);

    bool found_continuous = false;
    for (int ci = 0; ci < top_n; ++ci) {
        const int start = original_top[ci];
        if (start < 0 || start + top_n - 1 > 255) continue;
        bool all_exist = true;
        for (int i = 0; i < top_n; ++i) {
            if (exponent_counts_out[start + i] <= 0) { all_exist = false; break; }
        }
        if (!all_exist) continue;
        for (int i = 0; i < top_n; ++i) top_exponents[i] = start + i;
        found_continuous = true;
        break;
    }
    if (!found_continuous) {
        int best_start = original_top[0];
        int max_length = 1;
        int current_length = 1;
        for (int i = 1; i < top_n; i++) {
            if (original_top[i] == original_top[i-1] + 1) {
                current_length++;
                if (current_length > max_length) {
                    max_length = current_length;
                    best_start = original_top[i] - current_length + 1;
                }
            } else {
                current_length = 1;
            }
        }
        for (int i = 0; i < top_n; i++) top_exponents[i] = best_start + i;
    }

    return true;
}

bool AllocReusableBuffers(int padded_rows, int padded_cols, CompressReusableBuffers* rb) {
    const int tile_m = 8, tile_m_median = 16, tile_m_global = 64;
    const int tile_k = 8, tile_k_median = 64, tile_k_global = 64;

    const int num_tiles = (padded_rows / tile_m) * (padded_cols / tile_k);
    const int num_median_tiles = (padded_rows / tile_m_median) * (padded_cols / tile_k_median);
    const int num_global_tiles = (padded_rows / tile_m_global) * (padded_cols / tile_k_global);
    const size_t max_elems = (size_t)padded_rows * padded_cols;
    const int max_elem_per_gtile = tile_m_global * tile_k_global;
    const int max_sm_per_gtile = max_elem_per_gtile + 15;
    const int max_full_per_gtile = max_elem_per_gtile + 7;

    // Output buffers (max possible size)
    rb->sign_mantissa = (uint8_t*)malloc(max_elems > 0 ? max_elems : 1);
    rb->compressed_full = (__nv_bfloat16*)malloc((max_elems > 0 ? max_elems : 1) * sizeof(__nv_bfloat16));
    rb->bitmap1 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    rb->bitmap2 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    rb->bitmap3 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    rb->tile_offsets = (int*)malloc(num_tiles * 2 * sizeof(int));
    rb->tile_offsets_median = (int*)malloc(num_median_tiles * 2 * sizeof(int));
    rb->tile_offsets_global = (int*)malloc((num_global_tiles + 1) * 2 * sizeof(int));

    // Temp workspace
    rb->temp_sm = (uint8_t*)malloc((size_t)num_global_tiles * max_sm_per_gtile);
    rb->temp_full = (__nv_bfloat16*)malloc((size_t)num_global_tiles * max_full_per_gtile * sizeof(__nv_bfloat16));
    rb->gt_hf_count = (int*)malloc(num_global_tiles * sizeof(int));
    rb->gt_full_count = (int*)malloc(num_global_tiles * sizeof(int));
    rb->hf_offsets = (int*)malloc((num_global_tiles + 1) * sizeof(int));
    rb->full_offsets = (int*)malloc((num_global_tiles + 1) * sizeof(int));

    rb->alloc_rows = padded_rows;
    rb->alloc_cols = padded_cols;
    rb->num_global_tiles = num_global_tiles;

    return rb->sign_mantissa && rb->compressed_full &&
           rb->bitmap1 && rb->bitmap2 && rb->bitmap3 &&
           rb->tile_offsets && rb->tile_offsets_median && rb->tile_offsets_global &&
           rb->temp_sm && rb->temp_full &&
           rb->gt_hf_count && rb->gt_full_count &&
           rb->hf_offsets && rb->full_offsets;
}

void FreeReusableBuffers(CompressReusableBuffers* rb) {
    free(rb->sign_mantissa);
    free(rb->compressed_full);
    free(rb->bitmap1);
    free(rb->bitmap2);
    free(rb->bitmap3);
    free(rb->tile_offsets);
    free(rb->tile_offsets_median);
    free(rb->tile_offsets_global);
    free(rb->temp_sm);
    free(rb->temp_full);
    free(rb->gt_hf_count);
    free(rb->gt_full_count);
    free(rb->hf_offsets);
    free(rb->full_offsets);
    *rb = CompressReusableBuffers();
}

bool RunCompressionReuse(__nv_bfloat16* matrix,
                         int rows,
                         int cols,
                         const int* top_exponents,
                         CompressReusableBuffers* rb,
                         CompressedBuffers* out,
                         std::string* error) {
    const int tile_m = 8, tile_m_median = 16, tile_m_global = 64;
    const int tile_k = 8, tile_k_median = 64, tile_k_global = 64;

    int max_high_freq_count = 0;
    int max_full_count = 0;
    int total_hf = 0;
    int total_full = 0;

    const int num_global_tiles = InitBF16MatrixTripleBitmap_Reuse_SIMD(
        matrix, rows, cols,
        tile_m, tile_m_median, tile_m_global,
        tile_k, tile_k_median, tile_k_global,
        top_exponents,
        rb->sign_mantissa, rb->compressed_full,
        rb->bitmap1, rb->bitmap2, rb->bitmap3,
        rb->tile_offsets, rb->tile_offsets_median, rb->tile_offsets_global,
        rb->temp_sm, rb->temp_full,
        rb->gt_hf_count, rb->gt_full_count,
        rb->hf_offsets, rb->full_offsets,
        max_high_freq_count, max_full_count,
        total_hf, total_full);

    if (num_global_tiles <= 0) {
        *error = "InitBF16MatrixTripleBitmap_Reuse failed";
        return false;
    }

    out->sign_mantissa = rb->sign_mantissa;
    out->compressed_full = rb->compressed_full;
    out->bitmap1 = rb->bitmap1;
    out->bitmap2 = rb->bitmap2;
    out->bitmap3 = rb->bitmap3;
    out->tile_offsets = rb->tile_offsets;
    out->tile_offsets_median = rb->tile_offsets_median;
    out->tile_offsets_global = rb->tile_offsets_global;

    out->num_global_tiles = num_global_tiles;
    out->max_high_freq_count = max_high_freq_count;
    out->max_full_count = max_full_count;
    out->start_exp = static_cast<uint8_t>(top_exponents[0] - 1);
    out->high_freq_count = total_hf;
    out->full_count = total_full;

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

bool PrepareDeviceCompressedBuffers(const CompressedBuffers& host,
                                    int rows,
                                    int cols,
                                    DeviceCompressedBuffers* device,
                                    std::string* error) {
    FreeDeviceCompressedBuffers(device);

    const int num_tiles = (rows / 8) * (cols / 8);
    const int num_median_tiles = (rows / 16) * (cols / 64);
    const size_t out_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(__nv_bfloat16);

    const size_t sign_bytes = static_cast<size_t>(host.high_freq_count) * sizeof(uint8_t);
    const size_t full_bytes = static_cast<size_t>(host.full_count) * sizeof(__nv_bfloat16);
    const size_t bitmap_bytes = static_cast<size_t>(num_tiles) * sizeof(uint64_t);
    const size_t median_bytes = static_cast<size_t>(num_median_tiles) * 2 * sizeof(int);
    const size_t global_bytes = static_cast<size_t>(host.num_global_tiles + 1) * 2 * sizeof(int);

    cudaError_t ce = cudaSuccess;

    ce = cudaMalloc(reinterpret_cast<void**>(&device->sign_mantissa), std::max<size_t>(sign_bytes, 1));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(sign_mantissa) failed: ") + cudaGetErrorString(ce);
        return false;
    }
    ce = cudaMalloc(reinterpret_cast<void**>(&device->compressed_full), std::max<size_t>(full_bytes, 1));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(compressed_full) failed: ") + cudaGetErrorString(ce);
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    ce = cudaMalloc(reinterpret_cast<void**>(&device->bitmap1), std::max<size_t>(bitmap_bytes, 1));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(bitmap1) failed: ") + cudaGetErrorString(ce);
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    ce = cudaMalloc(reinterpret_cast<void**>(&device->bitmap2), std::max<size_t>(bitmap_bytes, 1));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(bitmap2) failed: ") + cudaGetErrorString(ce);
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    ce = cudaMalloc(reinterpret_cast<void**>(&device->bitmap3), std::max<size_t>(bitmap_bytes, 1));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(bitmap3) failed: ") + cudaGetErrorString(ce);
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    ce = cudaMalloc(reinterpret_cast<void**>(&device->tile_offsets_median), std::max<size_t>(median_bytes, 1));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(tile_offsets_median) failed: ") + cudaGetErrorString(ce);
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    ce = cudaMalloc(reinterpret_cast<void**>(&device->tile_offsets_global), std::max<size_t>(global_bytes, 1));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(tile_offsets_global) failed: ") + cudaGetErrorString(ce);
        FreeDeviceCompressedBuffers(device);
        return false;
    }
    ce = cudaMalloc(reinterpret_cast<void**>(&device->output), std::max<size_t>(out_bytes, 1));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(output) failed: ") + cudaGetErrorString(ce);
        FreeDeviceCompressedBuffers(device);
        return false;
    }

    if (sign_bytes > 0) {
        ce = cudaMemcpy(device->sign_mantissa, host.sign_mantissa, sign_bytes, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            *error = std::string("cudaMemcpy(sign_mantissa) failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(device);
            return false;
        }
    }
    if (full_bytes > 0) {
        ce = cudaMemcpy(device->compressed_full, host.compressed_full, full_bytes, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            *error = std::string("cudaMemcpy(compressed_full) failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(device);
            return false;
        }
    }
    if (bitmap_bytes > 0) {
        ce = cudaMemcpy(device->bitmap1, host.bitmap1, bitmap_bytes, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            *error = std::string("cudaMemcpy(bitmap1) failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(device);
            return false;
        }
        ce = cudaMemcpy(device->bitmap2, host.bitmap2, bitmap_bytes, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            *error = std::string("cudaMemcpy(bitmap2) failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(device);
            return false;
        }
        ce = cudaMemcpy(device->bitmap3, host.bitmap3, bitmap_bytes, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            *error = std::string("cudaMemcpy(bitmap3) failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(device);
            return false;
        }
    }
    if (median_bytes > 0) {
        ce = cudaMemcpy(device->tile_offsets_median, host.tile_offsets_median, median_bytes, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            *error = std::string("cudaMemcpy(tile_offsets_median) failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(device);
            return false;
        }
    }
    if (global_bytes > 0) {
        ce = cudaMemcpy(device->tile_offsets_global, host.tile_offsets_global, global_bytes, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            *error = std::string("cudaMemcpy(tile_offsets_global) failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(device);
            return false;
        }
    }

    ce = cudaMemset(device->output, 0, out_bytes);
    if (ce != cudaSuccess) {
        *error = std::string("cudaMemset(output) failed: ") + cudaGetErrorString(ce);
        FreeDeviceCompressedBuffers(device);
        return false;
    }

    return true;
}

bool BenchmarkDecompress(const CompressedBuffers& host,
                         int rows,
                         int cols,
                         const ProgramOptions& opts,
                         double* decompress_ms,
                         std::vector<__nv_bfloat16>* verify_output,
                         std::string* error) {
    DeviceCompressedBuffers device;
    if (!PrepareDeviceCompressedBuffers(host, rows, cols, &device, error)) {
        return false;
    }

    // Warmup for decompression-only benchmark.
    for (int i = 0; i < opts.warmup; ++i) {
        const cudaError_t ce = BF16TripleBitmap_Decompress_API(
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
        if (ce != cudaSuccess) {
            *error = std::string("BF16TripleBitmap_Decompress_API warmup failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
    }

    cudaError_t ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) {
        *error = std::string("cudaDeviceSynchronize after warmup failed: ") + cudaGetErrorString(ce);
        FreeDeviceCompressedBuffers(&device);
        return false;
    }

    bench_common::CudaEventTimer timer;
    float total_decomp_ms = 0.0f;

    for (int i = 0; i < opts.iters; ++i) {
        if (!timer.RecordStart(0, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }

        ce = BF16TripleBitmap_Decompress_API(
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
        if (ce != cudaSuccess) {
            *error = std::string("BF16TripleBitmap_Decompress_API benchmark failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(&device);
            return false;
        }

        if (!timer.RecordStop(0, error) || !timer.SyncStop(error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }

        float iter_ms = 0.0f;
        if (!timer.ElapsedMs(&iter_ms, error)) {
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
        total_decomp_ms += iter_ms;
    }

    *decompress_ms = static_cast<double>(total_decomp_ms) / static_cast<double>(opts.iters);

    if (verify_output != nullptr) {
        ce = BF16TripleBitmap_Decompress_API(
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
        if (ce != cudaSuccess) {
            *error = std::string("BF16TripleBitmap_Decompress_API verify run failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(&device);
            return false;
        }

        ce = cudaDeviceSynchronize();
        if (ce != cudaSuccess) {
            *error = std::string("cudaDeviceSynchronize verify failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(&device);
            return false;
        }

        const size_t out_numel = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        const size_t out_bytes = out_numel * sizeof(__nv_bfloat16);
        verify_output->resize(out_numel);
        ce = cudaMemcpy(verify_output->data(), device.output, out_bytes, cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) {
            *error = std::string("cudaMemcpy verify output failed: ") + cudaGetErrorString(ce);
            FreeDeviceCompressedBuffers(&device);
            return false;
        }
    }

    FreeDeviceCompressedBuffers(&device);
    return true;
}

size_t CountBF16Mismatches(const std::vector<__nv_bfloat16>& expected,
                           const std::vector<__nv_bfloat16>& actual) {
    if (expected.size() != actual.size()) {
        return std::numeric_limits<size_t>::max();
    }
    size_t mismatch = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (__bfloat16_as_ushort(expected[i]) != __bfloat16_as_ushort(actual[i])) {
            ++mismatch;
        }
    }
    return mismatch;
}

double BytesPerSecondToGBps(size_t bytes, double milliseconds) {
    if (milliseconds <= 0.0) {
        return 0.0;
    }
    const double seconds = milliseconds / 1000.0;
    return static_cast<double>(bytes) / seconds / 1e9;
}

int ParseTensorIdFromName(const std::string& name) {
    if (name.size() <= 2 || name[0] != 't' || name[1] != '_') {
        return -1;
    }
    for (size_t i = 2; i < name.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
            return -1;
        }
    }
    try {
        return std::stoi(name.substr(2));
    } catch (...) {
        return -1;
    }
}

std::string BuildTensorBaseLabel(const std::string& file_name) {
    const int parsed = ParseTensorIdFromName(file_name);
    if (parsed >= 0) {
        return std::to_string(parsed);
    }
    return file_name;
}

std::vector<BenchRow> ProcessFile(const std::string& file_path,
                                  const ProgramOptions& opts,
                                  double* compress_wall_ms_out,
                                  double* service_wall_ms_out) {
    std::vector<BenchRow> rows;
    double file_compress_wall_ms = 0.0;
    double file_analyze_wall_ms = 0.0;
    if (compress_wall_ms_out != nullptr) {
        *compress_wall_ms_out = 0.0;
    }
    if (service_wall_ms_out != nullptr) {
        *service_wall_ms_out = 0.0;
    }
    const std::string file_name = Basename(file_path);
    const int parsed_tensor_id = ParseTensorIdFromName(file_name);
    const std::string tensor_base_label = BuildTensorBaseLabel(file_name);

    auto MakeBaseRow = [&](int sub_id) {
        BenchRow row;
        row.file_path = file_path;
        row.file_name = file_name;
        row.tensor_id = parsed_tensor_id;
        row.tensor_label = tensor_base_label + "-" + std::to_string(sub_id);
        return row;
    };

    BenchRow file_row = MakeBaseRow(0);
    std::vector<uint8_t> raw;
    std::string error;
    if (!bench_common::ReadBinaryFile(file_path, &raw, &error)) {
        file_row.status = "failed";
        file_row.note = error;
        rows.push_back(file_row);
        return rows;
    }

    NpyInfo info;
    if (!ParseNpyInfo(raw, &info, &error)) {
        file_row.status = "failed";
        file_row.note = error;
        rows.push_back(file_row);
        return rows;
    }

    file_row.shape3d = ShapeToString64(info.shape);

    if (info.fortran_order) {
        file_row.status = "skipped";
        file_row.note = "fortran_order_true_not_supported";
        rows.push_back(file_row);
        return rows;
    }

    if (!IsSupportedFloat16Descr(info.descr)) {
        file_row.status = "skipped";
        file_row.note = "dtype_not_float16(descr=" + info.descr + ")";
        rows.push_back(file_row);
        return rows;
    }

    if (info.shape.size() != 3) {
        file_row.status = "skipped";
        file_row.note = "non_3d_shape";
        rows.push_back(file_row);
        return rows;
    }

    const int64_t t = info.shape[0];
    const int64_t h = info.shape[1];
    const int64_t d = info.shape[2];
    if (t <= 0 || h <= 0 || d <= 0) {
        file_row.status = "failed";
        file_row.note = "shape_has_non_positive_dimension";
        rows.push_back(file_row);
        return rows;
    }

    const int64_t chunk_len = 16;
    const int64_t num_chunks = (t + chunk_len - 1) / chunk_len;
    struct ChunkWork {
        BenchRow row;
        std::vector<__nv_bfloat16> host_bf16;
        std::array<int, 7> top_exponents = {0, 0, 0, 0, 0, 0, 0};
        int padded_rows = 0;
        int padded_cols = 0;
        bool preprocess_ok = false;
        bool compress_ok = false;
    };

    std::vector<ChunkWork> work_items(static_cast<size_t>(num_chunks));

    // 1) Lightweight serial init: fill metadata from shape info only (no heavy work)
    int max_padded_rows = 0, max_padded_cols = 0;
    for (int64_t sub_id = 0; sub_id < num_chunks; ++sub_id) {
        ChunkWork& work = work_items[static_cast<size_t>(sub_id)];
        BenchRow& row = work.row;
        row = MakeBaseRow(static_cast<int>(sub_id));

        const int64_t seq_begin = sub_id * chunk_len;
        const int64_t seq_count = (sub_id == num_chunks - 1) ? (t - seq_begin) : chunk_len;
        row.shape3d = "[" + std::to_string(seq_count) + "," + std::to_string(h) + "," + std::to_string(d) + "]";

        // Pre-compute padded dimensions from shape alone
        const int mapped_rows = static_cast<int>(seq_count * h);
        const int mapped_cols = static_cast<int>(d);
        work.padded_rows = RoundUpTo64(mapped_rows);
        work.padded_cols = RoundUpTo64(mapped_cols);
        max_padded_rows = std::max(max_padded_rows, work.padded_rows);
        max_padded_cols = std::max(max_padded_cols, work.padded_cols);
    }

    // 2) Parallel preprocessing + compression benchmark
    //    Pre-allocate per-thread reusable buffers, then run conversion,
    //    analysis, and compression in a single parallel region.

    // Allocate one CompressReusableBuffers per OMP thread
    const int num_threads = omp_get_max_threads();
    std::vector<CompressReusableBuffers> thread_bufs(static_cast<size_t>(num_threads));
    bool alloc_ok = true;
    if (max_padded_rows > 0 && max_padded_cols > 0) {
        for (int t = 0; t < num_threads; ++t) {
            if (!AllocReusableBuffers(max_padded_rows, max_padded_cols, &thread_bufs[static_cast<size_t>(t)])) {
                alloc_ok = false;
                break;
            }
        }
    }
    if (!alloc_ok) {
        for (int t = 0; t < num_threads; ++t) {
            FreeReusableBuffers(&thread_bufs[static_cast<size_t>(t)]);
        }
        for (int64_t idx = 0; idx < num_chunks; ++idx) {
            work_items[static_cast<size_t>(idx)].row.status = "failed";
            work_items[static_cast<size_t>(idx)].row.note = "AllocReusableBuffers failed (per-thread)";
        }
    }

    // Per-chunk accumulators for per-thread timing (for CSV per-chunk compress_ms)
    std::vector<double> chunk_total_compress_ms(static_cast<size_t>(num_chunks), 0.0);
    std::vector<size_t> chunk_last_comp_bytes(static_cast<size_t>(num_chunks), 0);
    std::vector<bool> chunk_failed(static_cast<size_t>(num_chunks), false);

    if (alloc_ok && max_padded_rows > 0) {
        // 2a) Parallel preprocessing: warmup + timed iterations
        //     First opts.warmup runs warm up caches; remaining opts.iters runs are timed.
        double total_preprocess_wall_ms = 0.0;
        const int preprocess_total_runs = opts.warmup + opts.iters;
        for (int preprocess_run = 0; preprocess_run < preprocess_total_runs; ++preprocess_run) {
            const bool is_timed = (preprocess_run >= opts.warmup);
            std::chrono::high_resolution_clock::time_point preprocess_wall_begin;
            if (is_timed) {
                preprocess_wall_begin = std::chrono::high_resolution_clock::now();
            }

            #pragma omp parallel for schedule(dynamic, 1)
            for (int64_t idx = 0; idx < num_chunks; ++idx) {
                ChunkWork& work = work_items[static_cast<size_t>(idx)];
                BenchRow& row = work.row;

                const int64_t seq_begin = idx * chunk_len;
                const int64_t seq_count = (idx == num_chunks - 1) ? (t - seq_begin) : chunk_len;

                int mapped_rows = 0;
                int mapped_cols = 0;
                int padded_rows = 0;
                int padded_cols = 0;
                int exponent_counts[256];
                std::string local_error;

                if (!ConvertFp16ToBf16WithAnalysis(info,
                                                   raw,
                                                   seq_begin,
                                                   seq_count,
                                                   &work.host_bf16,
                                                   &mapped_rows,
                                                   &mapped_cols,
                                                   &padded_rows,
                                                   &padded_cols,
                                                   &row.logical_numel,
                                                   &row.input_numel,
                                                   &row.convert_ms,
                                                   work.top_exponents.data(),
                                                   exponent_counts,
                                                   &local_error)) {
                    row.status = (local_error == "non_3d_shape") ? "skipped" : "failed";
                    row.note = local_error;
                    continue;
                }

                row.mapped_shape = "[" + std::to_string(mapped_rows) + "," + std::to_string(mapped_cols) + "]";
                row.padded_shape = "[" + std::to_string(padded_rows) + "," + std::to_string(padded_cols) + "]";

                row.logical_bytes = row.logical_numel * sizeof(__nv_bfloat16);
                row.input_bytes = row.input_numel * sizeof(__nv_bfloat16);
                row.pad_elems = row.input_numel - row.logical_numel;
                row.pad_ratio = (row.logical_numel > 0)
                                    ? static_cast<double>(row.pad_elems) / static_cast<double>(row.logical_numel)
                                    : 0.0;

                if (padded_rows % 64 != 0 || padded_cols % 64 != 0) {
                    row.status = "failed";
                    row.note = "internal_error_padded_shape_not_multiple_of_64";
                    continue;
                }

                const std::array<int, 7> selected_exponents = {
                    work.top_exponents[0], work.top_exponents[1], work.top_exponents[2], work.top_exponents[3],
                    work.top_exponents[4], work.top_exponents[5], work.top_exponents[6]
                };
                std::array<int, 7> raw_top_exponents;
                CollectExponentCoverageFromHistogram(exponent_counts,
                                                     static_cast<size_t>(padded_rows) * static_cast<size_t>(padded_cols),
                                                     selected_exponents,
                                                     &raw_top_exponents,
                                                     &row.top7_raw_cover,
                                                     &row.top7_selected_cover);

                row.top7_raw = FormatExponentList(raw_top_exponents);
                row.top7_selected = FormatExponentList(selected_exponents);

                work.padded_rows = padded_rows;
                work.padded_cols = padded_cols;
                work.preprocess_ok = true;
            }

            if (is_timed) {
                const auto preprocess_wall_end = std::chrono::high_resolution_clock::now();
                total_preprocess_wall_ms += std::chrono::duration<double, std::milli>(preprocess_wall_end - preprocess_wall_begin).count();
            }
        }
        file_analyze_wall_ms = total_preprocess_wall_ms / static_cast<double>(opts.iters);

        // 2b) Warmup: parallel over chunks
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t idx = 0; idx < num_chunks; ++idx) {
            ChunkWork& work = work_items[static_cast<size_t>(idx)];
            if (!work.preprocess_ok) continue;
            const int tid = omp_get_thread_num();
            CompressReusableBuffers& rb = thread_bufs[static_cast<size_t>(tid)];
            std::string local_error;
            for (int i = 0; i < opts.warmup; ++i) {
                CompressedBuffers warmup;
                if (!RunCompressionReuse(work.host_bf16.data(),
                                         work.padded_rows, work.padded_cols,
                                         work.top_exponents.data(),
                                         &rb, &warmup, &local_error)) {
                    work.row.status = "failed";
                    work.row.note = local_error;
                    chunk_failed[static_cast<size_t>(idx)] = true;
                    break;
                }
            }
        }

        // 2c) Timed iterations: iter loop OUTSIDE, parallel-for INSIDE
        //     Wall clock wraps the entire parallel compression per iteration.
        double total_compress_wall_ms = 0.0;
        for (int iter = 0; iter < opts.iters; ++iter) {
            const auto wall_begin = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for schedule(dynamic, 1)
            for (int64_t idx = 0; idx < num_chunks; ++idx) {
                if (chunk_failed[static_cast<size_t>(idx)]) continue;
                ChunkWork& work = work_items[static_cast<size_t>(idx)];
                if (!work.preprocess_ok) continue;

                const int tid = omp_get_thread_num();
                CompressReusableBuffers& rb = thread_bufs[static_cast<size_t>(tid)];
                std::string local_error;
                CompressedBuffers iteration;

                const auto t_begin = std::chrono::high_resolution_clock::now();
                const bool ok = RunCompressionReuse(work.host_bf16.data(),
                                                    work.padded_rows, work.padded_cols,
                                                    work.top_exponents.data(),
                                                    &rb, &iteration, &local_error);
                const auto t_end = std::chrono::high_resolution_clock::now();

                if (!ok) {
                    work.row.status = "failed";
                    work.row.note = local_error;
                    chunk_failed[static_cast<size_t>(idx)] = true;
                } else {
                    chunk_total_compress_ms[static_cast<size_t>(idx)] +=
                        std::chrono::duration<double, std::milli>(t_end - t_begin).count();
                    chunk_last_comp_bytes[static_cast<size_t>(idx)] = iteration.comp_bytes;
                }
            }

            const auto wall_end = std::chrono::high_resolution_clock::now();
            total_compress_wall_ms += std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
        }

        file_compress_wall_ms = total_compress_wall_ms / static_cast<double>(opts.iters);
    }

    // Free per-thread buffers
    for (int t = 0; t < num_threads; ++t) {
        FreeReusableBuffers(&thread_bufs[static_cast<size_t>(t)]);
    }

    // Fill per-chunk metrics from accumulators
    for (int64_t idx = 0; idx < num_chunks; ++idx) {
        ChunkWork& work = work_items[static_cast<size_t>(idx)];
        if (!work.preprocess_ok || chunk_failed[static_cast<size_t>(idx)]) continue;

        BenchRow& row = work.row;
        row.compress_ms = chunk_total_compress_ms[static_cast<size_t>(idx)] / static_cast<double>(opts.iters);
        row.compressed_bytes = chunk_last_comp_bytes[static_cast<size_t>(idx)];
        if (row.compressed_bytes == 0) {
            row.status = "failed";
            row.note = "compressed_bytes_zero";
            continue;
        }

        row.ratio_logical = static_cast<double>(row.compressed_bytes) / static_cast<double>(row.logical_bytes);
        row.ratio_input = static_cast<double>(row.compressed_bytes) / static_cast<double>(row.input_bytes);
        row.compress_gbps_logical = BytesPerSecondToGBps(row.logical_bytes, row.compress_ms);
        work.compress_ok = true;
    }

    // 3) Serial decompress/verify to avoid GPU timing interference
    for (int64_t idx = 0; idx < num_chunks; ++idx) {
        ChunkWork& work = work_items[static_cast<size_t>(idx)];
        if (!work.compress_ok) {
            continue;
        }

        BenchRow& row = work.row;
        std::string local_error;

        CompressReusableBuffers reuse_buffers;
        if (!AllocReusableBuffers(work.padded_rows, work.padded_cols, &reuse_buffers)) {
            row.status = "failed";
            row.note = "AllocReusableBuffers failed (decompress stage)";
            FreeReusableBuffers(&reuse_buffers);
            continue;
        }

        CompressedBuffers final_buffers;
        if (!RunCompressionReuse(work.host_bf16.data(),
                                 work.padded_rows,
                                 work.padded_cols,
                                 work.top_exponents.data(),
                                 &reuse_buffers,
                                 &final_buffers,
                                 &local_error)) {
            row.status = "failed";
            row.note = local_error;
            FreeReusableBuffers(&reuse_buffers);
            continue;
        }

        std::vector<__nv_bfloat16> verify_output;
        if (!BenchmarkDecompress(final_buffers,
                                 work.padded_rows,
                                 work.padded_cols,
                                 opts,
                                 &row.decompress_ms,
                                 (opts.verify != 0) ? &verify_output : nullptr,
                                 &local_error)) {
            row.status = "failed";
            row.note = local_error;
            FreeReusableBuffers(&reuse_buffers);
            continue;
        }

        row.decompress_gbps_logical = BytesPerSecondToGBps(row.logical_bytes, row.decompress_ms);
        if (opts.verify != 0) {
            row.verify_mismatch = CountBF16Mismatches(work.host_bf16, verify_output);
            if (row.verify_mismatch == std::numeric_limits<size_t>::max()) {
                row.status = "failed";
                row.note = "verify_size_mismatch";
                FreeReusableBuffers(&reuse_buffers);
                continue;
            }
            if (row.verify_mismatch > 0) {
                std::ostringstream oss;
                oss << "verify_mismatch=" << row.verify_mismatch;
                row.status = "failed";
                row.note = oss.str();
                FreeReusableBuffers(&reuse_buffers);
                continue;
            }
        }

        row.status = "ok";
        FreeReusableBuffers(&reuse_buffers);
    }

    rows.reserve(static_cast<size_t>(num_chunks));
    for (int64_t sub_id = 0; sub_id < num_chunks; ++sub_id) {
        rows.push_back(work_items[static_cast<size_t>(sub_id)].row);
    }
    if (compress_wall_ms_out != nullptr) {
        *compress_wall_ms_out = file_compress_wall_ms;
    }
    if (service_wall_ms_out != nullptr) {
        *service_wall_ms_out = file_analyze_wall_ms + file_compress_wall_ms;
    }
    return rows;
}

MetricStats ComputeStats(const std::vector<double>& values) {
    MetricStats stats;
    if (values.empty()) {
        return stats;
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    stats.valid = true;
    stats.mean = sum / static_cast<double>(sorted.size());
    stats.min = sorted.front();
    stats.max = sorted.back();

    const size_t n = sorted.size();
    if (n % 2 == 1) {
        stats.median = sorted[n / 2];
    } else {
        stats.median = 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    }
    return stats;
}

AggregateSummary BuildAggregate(const std::vector<BenchRow>& rows) {
    AggregateSummary agg;
    agg.total = rows.size();

    std::vector<double> ratio_logical;
    std::vector<double> ratio_input;
    std::vector<double> pad_ratio;
    std::vector<double> convert_ms;
    std::vector<double> compress_ms;
    std::vector<double> decompress_ms;
    std::vector<double> compress_gbps_logical;
    std::vector<double> decompress_gbps_logical;

    for (size_t i = 0; i < rows.size(); ++i) {
        const BenchRow& row = rows[i];
        if (row.status == "ok") {
            ++agg.ok;
            agg.total_logical_bytes += row.logical_bytes;
            agg.total_input_bytes += row.input_bytes;
            agg.total_compressed_bytes += row.compressed_bytes;

            ratio_logical.push_back(row.ratio_logical);
            ratio_input.push_back(row.ratio_input);
            pad_ratio.push_back(row.pad_ratio);
            convert_ms.push_back(row.convert_ms);
            compress_ms.push_back(row.compress_ms);
            decompress_ms.push_back(row.decompress_ms);
            compress_gbps_logical.push_back(row.compress_gbps_logical);
            decompress_gbps_logical.push_back(row.decompress_gbps_logical);
        } else if (row.status == "skipped") {
            ++agg.skipped;
        } else {
            ++agg.failed;
        }
    }

    if (agg.total_logical_bytes > 0) {
        agg.total_ratio_logical = static_cast<double>(agg.total_compressed_bytes) /
                                  static_cast<double>(agg.total_logical_bytes);
    }
    if (agg.total_input_bytes > 0) {
        agg.total_ratio_input = static_cast<double>(agg.total_compressed_bytes) /
                                static_cast<double>(agg.total_input_bytes);
    }
    agg.ratio_logical = ComputeStats(ratio_logical);
    agg.ratio_input = ComputeStats(ratio_input);
    agg.pad_ratio = ComputeStats(pad_ratio);
    agg.convert_ms = ComputeStats(convert_ms);
    agg.compress_ms = ComputeStats(compress_ms);
    agg.decompress_ms = ComputeStats(decompress_ms);
    agg.compress_gbps_logical = ComputeStats(compress_gbps_logical);
    agg.decompress_gbps_logical = ComputeStats(decompress_gbps_logical);

    return agg;
}

std::string FormatDouble(double x, int precision = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << x;
    return oss.str();
}

std::string FormatTop7WithCover(const std::string& top7, double cover_ratio) {
    return top7 + " (" + FormatDouble(cover_ratio * 100.0, 1) + "%)";
}

void PrintRowHeader() {
    std::cout << std::string(214, '-') << "\n";
    std::cout << std::left
              << std::setw(14) << "TensorID"
              << std::setw(14) << "Shape3D"
              << std::setw(14) << "Mapped2D"
              << std::setw(14) << "Padded2D"
              << std::setw(13) << "OrigL(B)"
              << std::setw(13) << "Comp(B)"
              << std::setw(9) << "RatioL"
              << std::setw(9) << "RatioIn"
              << std::setw(38) << "Top7Raw"
              << std::setw(38) << "Top7Sel"
              << std::setw(10) << "PadRatio"
              << std::setw(10) << "Conv(ms)"
              << std::setw(10) << "Comp(ms)"
              << std::setw(10) << "Decomp"
              << "\n";
    std::cout << std::string(214, '-') << "\n";
}

void PrintRow(const BenchRow& row) {
    const bool ok = (row.status == "ok");
    const std::string top7_raw_with_cover = FormatTop7WithCover(row.top7_raw, row.top7_raw_cover);
    const std::string top7_selected_with_cover = FormatTop7WithCover(row.top7_selected, row.top7_selected_cover);
    std::cout << std::left
              << std::setw(14) << row.tensor_label
              << std::setw(14) << row.shape3d
              << std::setw(14) << row.mapped_shape
              << std::setw(14) << row.padded_shape
              << std::setw(13) << (ok ? std::to_string(row.logical_bytes) : "-")
              << std::setw(13) << (ok ? std::to_string(row.compressed_bytes) : "-")
              << std::setw(9) << (row.compressed_bytes > 0 ? FormatDouble(row.ratio_logical, 3) : "-")
              << std::setw(9) << (row.compressed_bytes > 0 ? FormatDouble(row.ratio_input, 3) : "-")
              << std::setw(38) << (ok ? top7_raw_with_cover : "-")
              << std::setw(38) << (ok ? top7_selected_with_cover : "-")
              << std::setw(10) << (row.input_numel > 0 ? FormatDouble(row.pad_ratio, 3) : "-")
              << std::setw(10) << (ok ? FormatDouble(row.convert_ms, 3) : "-")
              << std::setw(10) << (ok ? FormatDouble(row.compress_ms, 3) : "-")
              << std::setw(10) << (ok ? FormatDouble(row.decompress_ms, 3) : "-")
              << "\n";
}

void WriteCSVHeader(std::ofstream* ofs) {
    (*ofs)
        << "tensor_id,shape3d,mapped2d,padded2d,origl_b,comp_b,ratio_l,ratio_in,"
        << "top7_raw,top7_sel,pad_ratio,conv_ms,comp_ms,decomp_ms\n";
}

void WriteCSVRow(std::ofstream* ofs, const BenchRow& row) {
    const bool ok = (row.status == "ok");
    const std::string top7_raw_with_cover = FormatTop7WithCover(row.top7_raw, row.top7_raw_cover);
    const std::string top7_selected_with_cover = FormatTop7WithCover(row.top7_selected, row.top7_selected_cover);
    (*ofs)
        << bench_common::EscapeCSV(row.tensor_label) << ","
        << bench_common::EscapeCSV(row.shape3d) << ","
        << bench_common::EscapeCSV(row.mapped_shape) << ","
        << bench_common::EscapeCSV(row.padded_shape) << ","
        << (ok ? std::to_string(row.logical_bytes) : "-") << ","
        << (ok ? std::to_string(row.compressed_bytes) : "-") << ","
        << (row.compressed_bytes > 0 ? FormatDouble(row.ratio_logical, 3) : "-") << ","
        << (row.compressed_bytes > 0 ? FormatDouble(row.ratio_input, 3) : "-") << ","
        << bench_common::EscapeCSV(ok ? top7_raw_with_cover : "-") << ","
        << bench_common::EscapeCSV(ok ? top7_selected_with_cover : "-") << ","
        << (row.input_numel > 0 ? FormatDouble(row.pad_ratio, 3) : "-") << ","
        << (ok ? FormatDouble(row.convert_ms, 3) : "-") << ","
        << (ok ? FormatDouble(row.compress_ms, 3) : "-") << ","
        << (ok ? FormatDouble(row.decompress_ms, 3) : "-")
        << "\n";
}

void PrintSummary(const AggregateSummary& agg) {
    auto print_stats = [](const char* label, const MetricStats& s) {
        if (!s.valid) {
            std::cout << label << ": n/a\n";
            return;
        }
        std::cout << label
                  << ": mean=" << FormatDouble(s.mean, 4)
                  << ", median=" << FormatDouble(s.median, 4)
                  << ", min=" << FormatDouble(s.min, 4)
                  << ", max=" << FormatDouble(s.max, 4)
                  << "\n";
    };

    std::cout << "\n===== Summary =====\n";
    std::cout << "total=" << agg.total
              << ", ok=" << agg.ok
              << ", skipped=" << agg.skipped
              << ", failed=" << agg.failed << "\n";

    std::cout << "total_logical_bytes=" << agg.total_logical_bytes
              << ", total_input_bytes=" << agg.total_input_bytes
              << ", total_compressed_bytes=" << agg.total_compressed_bytes << "\n";

    if (agg.total_logical_bytes > 0 || agg.total_input_bytes > 0) {
        std::cout << "total_ratio_logical=" << FormatDouble(agg.total_ratio_logical, 4)
                  << ", total_ratio_input=" << FormatDouble(agg.total_ratio_input, 4) << "\n";
    }

    print_stats("ratio_logical", agg.ratio_logical);
    print_stats("ratio_input", agg.ratio_input);
    print_stats("pad_ratio", agg.pad_ratio);
    print_stats("convert_ms", agg.convert_ms);
    print_stats("compress_ms", agg.compress_ms);
    print_stats("decompress_ms", agg.decompress_ms);
    print_stats("compress_gbps_logical", agg.compress_gbps_logical);
    print_stats("decompress_gbps_logical", agg.decompress_gbps_logical);
}

int main(int argc, char** argv) {
#ifdef _OPENMP
    std::cout << "OpenMP enabled: max_threads=" << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP disabled: single-threaded mode\n";
#endif

    ProgramOptions opts;
    if (!ParseOptions(argc, argv, &opts)) {
        if (argc >= 2 && std::string(argv[1]) == "--help") {
            return 0;
        }
        return 1;
    }

    const std::set<std::string> ext_filter = ParseExtensionFilter(opts.ext_filter);

    std::vector<std::string> files;
    std::string error;
    if (!CollectInputFiles(opts, ext_filter, &files, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    if (files.empty()) {
        std::cerr << "No files found in input path after ext filter.\n";
        return 1;
    }

    const cudaError_t ce = cudaSetDevice(opts.device);
    if (ce != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(ce) << "\n";
        return 1;
    }

    std::ofstream csv(opts.out_csv.c_str(), std::ios::out | std::ios::trunc);
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV output: " << opts.out_csv << "\n";
        return 1;
    }
    WriteCSVHeader(&csv);

    std::cout << "===== KV Cache ZipServ Benchmark (BF16 fixed) =====\n";
    std::cout << "input_dir=" << opts.input_dir
              << ", files=" << files.size()
              << ", recursive=" << opts.recursive
              << ", ext_filter=" << (opts.ext_filter.empty() ? "<all>" : opts.ext_filter) << "\n";
    std::cout << "warmup=" << opts.warmup
              << ", iters=" << opts.iters
              << ", verify=" << opts.verify
              << ", device=" << opts.device << "\n";
    std::cout << "out_csv=" << opts.out_csv << "\n\n";

    PrintRowHeader();

    std::vector<BenchRow> rows;
    rows.reserve(files.size() * 8);
    double total_compress_wall_ms = 0.0;
    double total_service_wall_ms = 0.0;
    for (size_t i = 0; i < files.size(); ++i) {
        double file_compress_wall_ms = 0.0;
        double file_service_wall_ms = 0.0;
        const std::vector<BenchRow> file_rows = ProcessFile(files[i], opts, &file_compress_wall_ms, &file_service_wall_ms);
        total_compress_wall_ms += file_compress_wall_ms;
        total_service_wall_ms += file_service_wall_ms;
        for (size_t j = 0; j < file_rows.size(); ++j) {
            const BenchRow& row = file_rows[j];
            WriteCSVRow(&csv, row);
            if (row.status != "skipped" || opts.print_skipped != 0) {
                PrintRow(row);
            }
            rows.push_back(row);
        }
    }

    csv.close();
    std::cout << std::string(214, '-') << "\n";

    const AggregateSummary agg = BuildAggregate(rows);
    PrintSummary(agg);

    // Compression-only wall-clock metrics
    if (total_compress_wall_ms > 0.0) {
        std::cout << "\n--- Compress Wall-Clock ---\n";
        std::cout << "compress_wall_ms=" << FormatDouble(total_compress_wall_ms, 3);
        if (agg.ok > 0) {
            const double compress_wall_chunks_per_ms = static_cast<double>(agg.ok) / total_compress_wall_ms;
            std::cout << ", compress_wall_chunks_per_ms=" << FormatDouble(compress_wall_chunks_per_ms, 4);
        }
        if (agg.total_logical_bytes > 0) {
            const double compress_wall_gbps_logical = BytesPerSecondToGBps(agg.total_logical_bytes, total_compress_wall_ms);
            std::cout << ", compress_wall_gbps_logical=" << FormatDouble(compress_wall_gbps_logical, 4);
        }
        std::cout << ", omp_threads=" << omp_get_max_threads();
        std::cout << "\n";
    }

    // Service wall-clock metrics (analyze + compress; excludes convert/I/O/decompress)
    if (total_service_wall_ms > 0.0) {
        std::cout << "\n--- Service Wall-Clock (Analyze+Compress) ---\n";
        std::cout << "service_wall_ms=" << FormatDouble(total_service_wall_ms, 3);
        if (agg.ok > 0) {
            const double service_wall_chunks_per_ms = static_cast<double>(agg.ok) / total_service_wall_ms;
            std::cout << ", service_wall_chunks_per_ms=" << FormatDouble(service_wall_chunks_per_ms, 4);
        }
        if (agg.total_logical_bytes > 0) {
            const double service_wall_gbps_logical = BytesPerSecondToGBps(agg.total_logical_bytes, total_service_wall_ms);
            std::cout << ", service_wall_gbps_logical=" << FormatDouble(service_wall_gbps_logical, 4);
        }
        std::cout << ", omp_threads=" << omp_get_max_threads();
        std::cout << "\n";
    }

    std::cout << "CSV written to: " << opts.out_csv << "\n";

    return 0;
}
