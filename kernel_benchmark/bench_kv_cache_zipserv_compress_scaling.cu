/***************************************************************************
 * KV Cache ZipServ Compression Scaling Benchmark
 *
 * Measures total compression latency as a function of the number of
 * input KV-cache files (1, 2, 4, ..., 512) across three compression
 * methods: Single (1 thread), MT (OpenMP), MT-AVX (OpenMP + SIMD).
 *
 * Pipeline per file:
 *   Split (chunk_len=16) → parallel { Analyze → Compress } per chunk
 ***************************************************************************/
#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
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

#include "L_API.cuh"
#include "bench_manifest_utils.h"
#include "csv_writer.h"

// ===== Configuration =====
static const int kFileCounts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
static const int kNumFileCounts = sizeof(kFileCounts) / sizeof(kFileCounts[0]);
static const int64_t kChunkLen = 16;

enum CompressMethod {
    METHOD_SINGLE = 0,
    METHOD_MT     = 1,
    METHOD_MT_AVX = 2,
    METHOD_COUNT  = 3
};

static const char* kMethodNames[] = {"Single", "MT", "MT-AVX"};

// ===== Program Options =====
struct ProgramOptions {
    std::string input_dir = "~/saved_kv_cache";
    int warmup = 10;
    int iters = 100;
    int max_files = -1;
    int recursive = 0;
    std::string out_csv = "kv_cache_zipserv_scaling_results.csv";
    std::string ext_filter;
    int device = 0;
};

// ===== NPY Info =====
struct NpyInfo {
    int major = 0;
    int minor = 0;
    std::string descr;
    bool fortran_order = false;
    std::vector<int64_t> shape;
    size_t data_offset = 0;
    size_t data_bytes = 0;
};

// ===== Compression Buffers =====
struct CompressReusableBuffers {
    uint8_t* sign_mantissa = nullptr;
    __nv_bfloat16* compressed_full = nullptr;
    uint64_t* bitmap1 = nullptr;
    uint64_t* bitmap2 = nullptr;
    uint64_t* bitmap3 = nullptr;
    int* tile_offsets = nullptr;
    int* tile_offsets_median = nullptr;
    int* tile_offsets_global = nullptr;

    uint8_t* temp_sm = nullptr;
    __nv_bfloat16* temp_full = nullptr;
    int* gt_hf_count = nullptr;
    int* gt_full_count = nullptr;
    int* hf_offsets = nullptr;
    int* full_offsets = nullptr;

    int alloc_rows = 0;
    int alloc_cols = 0;
    int num_global_tiles = 0;
};

// ===== Preloaded File =====
struct PreloadedFile {
    std::vector<uint8_t> raw;
    NpyInfo info;
    int64_t num_chunks = 0;
    int max_padded_rows = 0;
    int max_padded_cols = 0;
};

// ===== Compression Stats (collected outside timed loop) =====
struct CompressionStats {
    size_t total_original_bytes = 0;
    size_t total_compressed_bytes = 0;
    double Ratio() const {
        return (total_compressed_bytes > 0)
            ? static_cast<double>(total_original_bytes) / static_cast<double>(total_compressed_bytes)
            : 0.0;
    }
};

// ===== Benchmark Result =====
struct ScalingResult {
    int file_count;
    CompressMethod method;
    double latency_ms;
    double comp_ratio;
};

// ===================================================================
// Utility functions (reused from bench_kv_cache_zipserv_compress.cu)
// ===================================================================

void PrintUsage() {
    std::cout
        << "Usage: ./bench_kv_cache_zipserv_compress_scaling [options]\n"
        << "Options:\n"
        << "  --input_dir <path>      Input directory (default: ~/saved_kv_cache)\n"
        << "  --warmup <int>          Warmup iterations (default: 10)\n"
        << "  --iters <int>           Benchmark iterations (default: 100)\n"
        << "  --max_files <int>       Max files after sorting (-1: all, default: -1)\n"
        << "  --recursive <0|1>       Recursive directory traversal (default: 0)\n"
        << "  --out_csv <path>        Output CSV path (default: kv_cache_zipserv_scaling_results.csv)\n"
        << "  --ext_filter <csv>      Extension allow-list, e.g. npy,noext\n"
        << "  --device <int>          CUDA device index (default: 0)\n"
        << "  --help                  Show help\n";
}

bool ParseBool01(const std::string& name, const std::string& value, int* out) {
    if (value == "0") { *out = 0; return true; }
    if (value == "1") { *out = 1; return true; }
    std::cerr << "Error: " << name << " must be 0 or 1, got: " << value << "\n";
    return false;
}

bool ParseOptions(int argc, char** argv, ProgramOptions* opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") { PrintUsage(); return false; }
        else if (arg == "--input_dir" && i + 1 < argc) { opts->input_dir = argv[++i]; }
        else if (arg == "--warmup" && i + 1 < argc) { opts->warmup = std::atoi(argv[++i]); }
        else if (arg == "--iters" && i + 1 < argc) { opts->iters = std::atoi(argv[++i]); }
        else if (arg == "--max_files" && i + 1 < argc) { opts->max_files = std::atoi(argv[++i]); }
        else if (arg == "--recursive" && i + 1 < argc) {
            int v = 0; if (!ParseBool01("--recursive", argv[++i], &v)) return false;
            opts->recursive = v;
        }
        else if (arg == "--out_csv" && i + 1 < argc) { opts->out_csv = argv[++i]; }
        else if (arg == "--ext_filter" && i + 1 < argc) { opts->ext_filter = argv[++i]; }
        else if (arg == "--device" && i + 1 < argc) { opts->device = std::atoi(argv[++i]); }
        else { std::cerr << "Unknown or incomplete argument: " << arg << "\n"; PrintUsage(); return false; }
    }
    if (opts->warmup < 0 || opts->iters <= 0) {
        std::cerr << "Error: warmup >= 0 and iters > 0 required.\n"; return false;
    }
    if (opts->max_files == 0 || opts->max_files < -1) {
        std::cerr << "Error: --max_files must be -1 or > 0.\n"; return false;
    }
    return true;
}

std::string ExpandUserPath(const std::string& input) {
    if (input.empty() || input[0] != '~') return input;
    const char* home = std::getenv("HOME");
    if (!home) return input;
    if (input.size() == 1) return std::string(home);
    if (input[1] == '/') return std::string(home) + input.substr(1);
    return input;
}

std::string ToLower(const std::string& s) {
    std::string out = s;
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(out[i])));
    return out;
}

std::string Basename(const std::string& path) {
    const size_t pos = path.find_last_of('/');
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

std::string GetExtensionNoDotLower(const std::string& path) {
    const std::string base = Basename(path);
    const size_t dot = base.find_last_of('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 >= base.size()) return "";
    return ToLower(base.substr(dot + 1));
}

std::set<std::string> ParseExtensionFilter(const std::string& spec) {
    std::set<std::string> exts;
    const std::vector<std::string> tokens = bench_common::SplitByComma(spec);
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string t = ToLower(bench_common::Trim(tokens[i]));
        if (t.empty()) continue;
        if (t[0] == '.') t = t.substr(1);
        if (t == "noext" || t == "(noext)") exts.insert("");
        else exts.insert(t);
    }
    return exts;
}

bool FilePassesExtensionFilter(const std::string& path, const std::set<std::string>& ext_filter) {
    if (ext_filter.empty()) return true;
    return ext_filter.find(GetExtensionNoDotLower(path)) != ext_filter.end();
}

bool CollectInputFiles(const ProgramOptions& opts,
                       const std::set<std::string>& ext_filter,
                       std::vector<std::string>* files,
                       std::string* error) {
    files->clear();
    const std::string root = ExpandUserPath(opts.input_dir);
    struct stat root_st;
    if (stat(root.c_str(), &root_st) != 0) {
        *error = "Failed to access input_dir: " + root + " (" + std::strerror(errno) + ")";
        return false;
    }
    if (S_ISREG(root_st.st_mode)) {
        if (FilePassesExtensionFilter(root, ext_filter)) files->push_back(root);
    } else if (S_ISDIR(root_st.st_mode)) {
        std::vector<std::string> dir_stack;
        dir_stack.push_back(root);
        while (!dir_stack.empty()) {
            const std::string dir = dir_stack.back(); dir_stack.pop_back();
            DIR* dp = opendir(dir.c_str());
            if (!dp) { *error = "opendir failed for " + dir; return false; }
            struct dirent* ent = nullptr;
            while ((ent = readdir(dp)) != nullptr) {
                const std::string name = ent->d_name;
                if (name == "." || name == "..") continue;
                const std::string full = dir + "/" + name;
                struct stat st;
                if (stat(full.c_str(), &st) != 0) continue;
                if (S_ISREG(st.st_mode)) {
                    if (FilePassesExtensionFilter(full, ext_filter)) files->push_back(full);
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
    if (opts.max_files > 0 && static_cast<int>(files->size()) > opts.max_files)
        files->resize(static_cast<size_t>(opts.max_files));
    return true;
}

// ===== NPY Parsing =====

bool SafeMulSize(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool ReadLittleU16(const std::vector<uint8_t>& raw, size_t off, uint16_t* out) {
    if (off + 2 > raw.size()) return false;
    *out = static_cast<uint16_t>(raw[off]) | static_cast<uint16_t>(raw[off + 1] << 8);
    return true;
}

bool ReadLittleU32(const std::vector<uint8_t>& raw, size_t off, uint32_t* out) {
    if (off + 4 > raw.size()) return false;
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
    if (key_pos == std::string::npos) key_pos = header.find(key2);
    if (key_pos == std::string::npos) return false;
    const size_t colon = header.find(':', key_pos);
    if (colon == std::string::npos) return false;
    size_t pos = colon + 1;
    while (pos < header.size() && std::isspace(static_cast<unsigned char>(header[pos]))) ++pos;
    if (pos >= header.size()) return false;
    *value_pos = pos;
    return true;
}

bool ParseQuotedStringValue(const std::string& header, const std::string& key, std::string* out) {
    size_t pos = 0;
    if (!LocateNpyValue(header, key, &pos)) return false;
    const char quote = header[pos];
    if (quote != '\'' && quote != '"') return false;
    ++pos;
    std::string value;
    while (pos < header.size()) {
        const char c = header[pos++];
        if (c == '\\') { if (pos >= header.size()) return false; value.push_back(header[pos++]); continue; }
        if (c == quote) { *out = value; return true; }
        value.push_back(c);
    }
    return false;
}

bool ParseBoolValue(const std::string& header, const std::string& key, bool* out) {
    size_t pos = 0;
    if (!LocateNpyValue(header, key, &pos)) return false;
    if (header.compare(pos, 4, "True") == 0) { *out = true; return true; }
    if (header.compare(pos, 5, "False") == 0) { *out = false; return true; }
    return false;
}

bool ParseShapeValue(const std::string& header, const std::string& key, std::vector<int64_t>* shape) {
    size_t pos = 0;
    if (!LocateNpyValue(header, key, &pos)) return false;
    if (header[pos] != '(') return false;
    const size_t end = header.find(')', pos + 1);
    if (end == std::string::npos) return false;
    const std::string tuple_content = header.substr(pos + 1, end - (pos + 1));
    std::stringstream ss(tuple_content);
    std::string token;
    shape->clear();
    while (std::getline(ss, token, ',')) {
        token = bench_common::Trim(token);
        if (token.empty()) continue;
        try { shape->push_back(std::stoll(token)); }
        catch (...) { return false; }
    }
    return !shape->empty();
}

bool ParseNpyInfo(const std::vector<uint8_t>& raw, NpyInfo* info, std::string* error) {
    const uint8_t kMagic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (raw.size() < 10) { *error = "npy file too small"; return false; }
    for (int i = 0; i < 6; ++i) {
        if (raw[static_cast<size_t>(i)] != kMagic[i]) { *error = "invalid npy magic"; return false; }
    }
    info->major = raw[6]; info->minor = raw[7];
    uint32_t header_len = 0; size_t header_offset = 0;
    if (info->major == 1) {
        uint16_t hl16 = 0;
        if (!ReadLittleU16(raw, 8, &hl16)) { *error = "failed to read npy v1 header length"; return false; }
        header_len = hl16; header_offset = 10;
    } else if (info->major == 2) {
        if (!ReadLittleU32(raw, 8, &header_len)) { *error = "failed to read npy v2 header length"; return false; }
        header_offset = 12;
    } else { *error = "unsupported npy major version"; return false; }
    const size_t header_end = header_offset + static_cast<size_t>(header_len);
    if (header_end > raw.size()) { *error = "npy header exceeds file size"; return false; }
    const std::string header(reinterpret_cast<const char*>(raw.data() + header_offset),
                             static_cast<size_t>(header_len));
    if (!ParseQuotedStringValue(header, "descr", &info->descr)) { *error = "npy header missing descr"; return false; }
    if (!ParseBoolValue(header, "fortran_order", &info->fortran_order)) { *error = "npy header missing fortran_order"; return false; }
    if (!ParseShapeValue(header, "shape", &info->shape)) { *error = "npy header missing/invalid shape"; return false; }
    info->data_offset = header_end;
    info->data_bytes = raw.size() - header_end;
    return true;
}

bool IsSupportedBf16Descr(const std::string& descr) {
    return descr == "|V2" || descr == "<V2" || descr == "V2" ||
           descr == "|u2" || descr == "<u2" || descr == "u2";
}

int RoundUpTo64(int x) { return ((x + 63) / 64) * 64; }

// ===== LoadBf16WithAnalysis (BF16 copy + exponent analysis) =====
bool LoadBf16WithAnalysis(const NpyInfo& info,
                          const std::vector<uint8_t>& raw,
                          int64_t seq_begin,
                          int64_t seq_count,
                          std::vector<__nv_bfloat16>* out,
                          int* padded_rows_out,
                          int* padded_cols_out,
                          int* top_exponents,
                          std::string* error) {
    if (info.shape.size() != 3) { *error = "non_3d_shape"; return false; }
    const int64_t t = info.shape[0];
    const int64_t h = info.shape[1];
    const int64_t d = info.shape[2];
    if (t <= 0 || h <= 0 || d <= 0) { *error = "shape_has_non_positive_dimension"; return false; }
    if (seq_begin < 0 || seq_count <= 0 || seq_begin + seq_count > t) {
        *error = "invalid_seq_chunk_range"; return false;
    }

    const int mapped_rows = static_cast<int>(seq_count * h);
    const int mapped_cols = static_cast<int>(d);
    const int padded_rows = RoundUpTo64(mapped_rows);
    const int padded_cols = RoundUpTo64(mapped_cols);
    *padded_rows_out = padded_rows;
    *padded_cols_out = padded_cols;

    const size_t input_numel = static_cast<size_t>(padded_rows) * static_cast<size_t>(padded_cols);
    out->assign(input_numel, __float2bfloat16(0.0f));

    const uint16_t* bf16_bits = reinterpret_cast<const uint16_t*>(raw.data() + info.data_offset);
    const size_t seq_row_base = static_cast<size_t>(seq_begin) * static_cast<size_t>(h);
    uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());

    // BF16 copy + exponent histogram
    int exponent_counts[256];
    memset(exponent_counts, 0, 256 * sizeof(int));

    for (int r = 0; r < mapped_rows; ++r) {
        const size_t global_row = seq_row_base + static_cast<size_t>(r);
        const uint16_t* src_row = bf16_bits + global_row * static_cast<size_t>(mapped_cols);
        uint16_t* dst_row = out_data + static_cast<size_t>(r) * static_cast<size_t>(padded_cols);
        std::memcpy(dst_row, src_row, static_cast<size_t>(mapped_cols) * sizeof(uint16_t));
        for (int c = 0; c < mapped_cols; ++c) {
            exponent_counts[(src_row[c] >> 7) & 0xFF]++;
        }
    }

    // Padding zeros (BF16 0.0 has exponent 0)
    const int padding_count = padded_rows * padded_cols - mapped_rows * mapped_cols;
    exponent_counts[0] += padding_count;

    // Find top-7 contiguous exponent range
    const int top_n = 7;
    int original_top[7];
    bool used[256] = {false};
    int n_found = 0;
    for (int ti = 0; ti < top_n; ++ti) {
        int best_exp = -1, best_count = 0;
        for (int e = 0; e < 256; ++e) {
            if (!used[e] && exponent_counts[e] > best_count) {
                best_count = exponent_counts[e]; best_exp = e;
            }
        }
        if (best_exp >= 0) { original_top[n_found++] = best_exp; used[best_exp] = true; }
    }
    while (n_found < top_n) { original_top[n_found] = 127 - n_found; n_found++; }
    std::sort(original_top, original_top + top_n);

    bool found_continuous = false;
    for (int ci = 0; ci < top_n; ++ci) {
        const int start = original_top[ci];
        if (start < 0 || start + top_n - 1 > 255) continue;
        bool all_exist = true;
        for (int i = 0; i < top_n; ++i) {
            if (exponent_counts[start + i] <= 0) { all_exist = false; break; }
        }
        if (!all_exist) continue;
        for (int i = 0; i < top_n; ++i) top_exponents[i] = start + i;
        found_continuous = true;
        break;
    }
    if (!found_continuous) {
        int best_start = original_top[0], max_length = 1, current_length = 1;
        for (int i = 1; i < top_n; i++) {
            if (original_top[i] == original_top[i - 1] + 1) {
                current_length++;
                if (current_length > max_length) {
                    max_length = current_length;
                    best_start = original_top[i] - current_length + 1;
                }
            } else { current_length = 1; }
        }
        for (int i = 0; i < top_n; i++) top_exponents[i] = best_start + i;
    }
    return true;
}

// ===== AllocReusableBuffers / FreeReusableBuffers =====
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

    rb->sign_mantissa = (uint8_t*)malloc(max_elems > 0 ? max_elems : 1);
    rb->compressed_full = (__nv_bfloat16*)malloc((max_elems > 0 ? max_elems : 1) * sizeof(__nv_bfloat16));
    rb->bitmap1 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    rb->bitmap2 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    rb->bitmap3 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    rb->tile_offsets = (int*)malloc(num_tiles * 2 * sizeof(int));
    rb->tile_offsets_median = (int*)malloc(num_median_tiles * 2 * sizeof(int));
    rb->tile_offsets_global = (int*)malloc((num_global_tiles + 1) * 2 * sizeof(int));

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

// ===================================================================
// Core benchmark logic
// ===================================================================

// Process all files for one benchmark point.
// For each file: split into chunks, then parallel { analyze → compress } per chunk.
void ProcessAllFiles(const std::vector<int>& work_indices,
                     const std::vector<PreloadedFile>& files,
                     CompressMethod method,
                     std::vector<CompressReusableBuffers>& thread_bufs,
                     std::vector<std::vector<__nv_bfloat16>>& thread_bf16_bufs,
                     CompressionStats* stats = nullptr) {
    for (size_t fi = 0; fi < work_indices.size(); ++fi) {
        const PreloadedFile& file = files[static_cast<size_t>(work_indices[fi])];
        const int64_t t = file.info.shape[0];
        const int64_t num_chunks = file.num_chunks;

        size_t file_orig = 0, file_comp = 0;

        #pragma omp parallel for schedule(dynamic, 1) reduction(+:file_orig,file_comp)
        for (int64_t idx = 0; idx < num_chunks; ++idx) {
            const int tid = omp_get_thread_num();
            CompressReusableBuffers& rb = thread_bufs[static_cast<size_t>(tid)];

            const int64_t seq_begin = idx * kChunkLen;
            const int64_t seq_count = std::min(kChunkLen, t - seq_begin);

            // === Analyze (reuse pre-allocated per-thread buffer) ===
            std::vector<__nv_bfloat16>& bf16_data = thread_bf16_bufs[static_cast<size_t>(tid)];
            int padded_rows = 0, padded_cols = 0;
            int top_exponents[7];
            std::string error;

            if (!LoadBf16WithAnalysis(file.info, file.raw,
                                      seq_begin, seq_count,
                                      &bf16_data,
                                      &padded_rows, &padded_cols,
                                      top_exponents, &error)) {
                continue;  // skip failed chunks
            }

            // === Compress ===
            int max_hf = 0, max_full = 0, total_hf = 0, total_full = 0;
            if (method == METHOD_MT_AVX) {
                InitBF16MatrixTripleBitmap_Reuse_SIMD(
                    bf16_data.data(), padded_rows, padded_cols,
                    8, 16, 64, 8, 64, 64,
                    top_exponents,
                    rb.sign_mantissa, rb.compressed_full,
                    rb.bitmap1, rb.bitmap2, rb.bitmap3,
                    rb.tile_offsets, rb.tile_offsets_median, rb.tile_offsets_global,
                    rb.temp_sm, rb.temp_full,
                    rb.gt_hf_count, rb.gt_full_count,
                    rb.hf_offsets, rb.full_offsets,
                    max_hf, max_full, total_hf, total_full);
            } else {
                // Single and MT both use _Reuse (thread count controls parallelism)
                InitBF16MatrixTripleBitmap_Reuse(
                    bf16_data.data(), padded_rows, padded_cols,
                    8, 16, 64, 8, 64, 64,
                    top_exponents,
                    rb.sign_mantissa, rb.compressed_full,
                    rb.bitmap1, rb.bitmap2, rb.bitmap3,
                    rb.tile_offsets, rb.tile_offsets_median, rb.tile_offsets_global,
                    rb.temp_sm, rb.temp_full,
                    rb.gt_hf_count, rb.gt_full_count,
                    rb.hf_offsets, rb.full_offsets,
                    max_hf, max_full, total_hf, total_full);
            }

            // === Collect stats (only when requested, outside timed loop) ===
            if (stats) {
                const size_t orig = static_cast<size_t>(padded_rows) * padded_cols * 2;
                const int nt  = (padded_rows / 8) * (padded_cols / 8);
                const int nmt = (padded_rows / 16) * (padded_cols / 64);
                const int ngt = (padded_rows / 64) * (padded_cols / 64);
                const size_t comp = static_cast<size_t>(total_hf)
                                  + static_cast<size_t>(total_full) * 2
                                  + static_cast<size_t>(nt) * 3 * 8
                                  + static_cast<size_t>(nt) * 2 * 4
                                  + static_cast<size_t>(nmt) * 2 * 4
                                  + static_cast<size_t>(ngt + 1) * 2 * 4;
                file_orig += orig;
                file_comp += comp;
            }
        }
        if (stats) {
            stats->total_original_bytes += file_orig;
            stats->total_compressed_bytes += file_comp;
        }
    }
}

std::string FormatDouble(double x, int precision = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << x;
    return oss.str();
}

// ===================================================================
// main
// ===================================================================
int main(int argc, char** argv) {
#ifdef _OPENMP
    std::cout << "OpenMP enabled: max_threads=" << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP disabled: single-threaded mode\n";
#endif
    const int saved_max_threads = omp_get_max_threads();

    ProgramOptions opts;
    if (!ParseOptions(argc, argv, &opts)) {
        if (argc >= 2 && std::string(argv[1]) == "--help") return 0;
        return 1;
    }

    // CUDA device setup
    {
        const cudaError_t ce = cudaSetDevice(opts.device);
        if (ce != cudaSuccess) {
            std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(ce) << "\n";
            return 1;
        }
    }

    // ---- Step 1: Collect and preload files ----
    const std::set<std::string> ext_filter = ParseExtensionFilter(opts.ext_filter);
    std::vector<std::string> file_paths;
    std::string error;
    if (!CollectInputFiles(opts, ext_filter, &file_paths, &error)) {
        std::cerr << error << "\n"; return 1;
    }
    if (file_paths.empty()) {
        std::cerr << "No files found in input path after ext filter.\n"; return 1;
    }

    std::cout << "Loading " << file_paths.size() << " files from " << opts.input_dir << " ...\n";

    std::vector<PreloadedFile> preloaded;
    preloaded.reserve(file_paths.size());
    int global_max_padded_rows = 0, global_max_padded_cols = 0;

    for (size_t i = 0; i < file_paths.size(); ++i) {
        PreloadedFile pf;
        if (!bench_common::ReadBinaryFile(file_paths[i], &pf.raw, &error)) {
            std::cerr << "Warning: skipping " << file_paths[i] << ": " << error << "\n";
            continue;
        }
        if (!ParseNpyInfo(pf.raw, &pf.info, &error)) {
            std::cerr << "Warning: skipping " << file_paths[i] << ": " << error << "\n";
            continue;
        }
        if (pf.info.fortran_order || !IsSupportedBf16Descr(pf.info.descr) || pf.info.shape.size() != 3) {
            std::cerr << "Warning: skipping " << file_paths[i] << ": unsupported format\n";
            continue;
        }
        const int64_t t = pf.info.shape[0];
        const int64_t h = pf.info.shape[1];
        const int64_t d = pf.info.shape[2];
        if (t <= 0 || h <= 0 || d <= 0) continue;

        pf.num_chunks = (t + kChunkLen - 1) / kChunkLen;

        // Compute max padded dims across all chunks in this file
        pf.max_padded_rows = 0;
        pf.max_padded_cols = RoundUpTo64(static_cast<int>(d));
        for (int64_t c = 0; c < pf.num_chunks; ++c) {
            const int64_t seq_count = std::min(kChunkLen, t - c * kChunkLen);
            const int pr = RoundUpTo64(static_cast<int>(seq_count * h));
            if (pr > pf.max_padded_rows) pf.max_padded_rows = pr;
        }
        if (pf.max_padded_rows > global_max_padded_rows) global_max_padded_rows = pf.max_padded_rows;
        if (pf.max_padded_cols > global_max_padded_cols) global_max_padded_cols = pf.max_padded_cols;

        preloaded.push_back(std::move(pf));
    }

    if (preloaded.empty()) {
        std::cerr << "No valid BF16 files loaded.\n"; return 1;
    }

    const int num_available = static_cast<int>(preloaded.size());
    std::cout << "Loaded " << num_available << " valid BF16 files.\n";
    std::cout << "Global max padded dims: [" << global_max_padded_rows << "," << global_max_padded_cols << "]\n\n";

    // ---- Step 2: Run scaling benchmark ----
    std::cout << "===== KV Cache ZipServ Compression Scaling Benchmark =====\n";
    std::cout << "warmup=" << opts.warmup
              << ", iters=" << opts.iters
              << ", device=" << opts.device
              << ", omp_max_threads=" << saved_max_threads << "\n\n";

    // Print header
    std::cout << std::left
              << std::setw(12) << "FileCount"
              << std::setw(14) << "Single(ms)"
              << std::setw(14) << "Single Comp"
              << std::setw(14) << "MT(ms)"
              << std::setw(14) << "MT Comp"
              << std::setw(14) << "MT-AVX(ms)"
              << std::setw(14) << "MT-AVX Comp"
              << "\n";
    std::cout << std::string(96, '-') << "\n";

    std::vector<ScalingResult> results;
    results.reserve(kNumFileCounts * METHOD_COUNT);

    for (int fc_idx = 0; fc_idx < kNumFileCounts; ++fc_idx) {
        const int file_count = kFileCounts[fc_idx];

        // Build work indices (cycling through available files)
        std::vector<int> work_indices(static_cast<size_t>(file_count));
        for (int i = 0; i < file_count; ++i) {
            work_indices[static_cast<size_t>(i)] = i % num_available;
        }

        double method_latencies[METHOD_COUNT] = {0.0};
        double method_ratios[METHOD_COUNT] = {0.0};

        for (int m = 0; m < METHOD_COUNT; ++m) {
            const CompressMethod method = static_cast<CompressMethod>(m);

            // Set OMP thread count
            if (method == METHOD_SINGLE) {
                omp_set_num_threads(1);
            } else {
                omp_set_num_threads(saved_max_threads);
            }

            // Allocate per-thread reusable buffers + bf16 data buffers
            const int num_threads = omp_get_max_threads();
            std::vector<CompressReusableBuffers> thread_bufs(static_cast<size_t>(num_threads));
            const size_t max_bf16_numel = static_cast<size_t>(global_max_padded_rows)
                                        * static_cast<size_t>(global_max_padded_cols);
            std::vector<std::vector<__nv_bfloat16>> thread_bf16_bufs(static_cast<size_t>(num_threads));
            for (int t = 0; t < num_threads; ++t) {
                thread_bf16_bufs[static_cast<size_t>(t)].reserve(max_bf16_numel);
            }
            bool alloc_ok = true;
            for (int t = 0; t < num_threads; ++t) {
                if (!AllocReusableBuffers(global_max_padded_rows, global_max_padded_cols,
                                          &thread_bufs[static_cast<size_t>(t)])) {
                    alloc_ok = false;
                    break;
                }
            }
            if (!alloc_ok) {
                std::cerr << "AllocReusableBuffers failed for method=" << kMethodNames[m]
                          << " file_count=" << file_count << "\n";
                for (int t = 0; t < num_threads; ++t)
                    FreeReusableBuffers(&thread_bufs[static_cast<size_t>(t)]);
                method_latencies[m] = -1.0;
                continue;
            }

            // Warmup
            for (int w = 0; w < opts.warmup; ++w) {
                ProcessAllFiles(work_indices, preloaded, method, thread_bufs, thread_bf16_bufs);
            }

            // Timed iterations
            double total_ms = 0.0;
            for (int iter = 0; iter < opts.iters; ++iter) {
                const auto wall_begin = std::chrono::high_resolution_clock::now();
                ProcessAllFiles(work_indices, preloaded, method, thread_bufs, thread_bf16_bufs);
                const auto wall_end = std::chrono::high_resolution_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
            }

            const double avg_ms = total_ms / static_cast<double>(opts.iters);
            method_latencies[m] = avg_ms;

            // Collect compression ratio (one extra untimed pass)
            CompressionStats comp_stats;
            ProcessAllFiles(work_indices, preloaded, method, thread_bufs, thread_bf16_bufs, &comp_stats);
            const double ratio = comp_stats.Ratio();
            method_ratios[m] = ratio;

            ScalingResult sr;
            sr.file_count = file_count;
            sr.method = method;
            sr.latency_ms = avg_ms;
            sr.comp_ratio = ratio;
            results.push_back(sr);

            // Cleanup
            for (int t = 0; t < num_threads; ++t)
                FreeReusableBuffers(&thread_bufs[static_cast<size_t>(t)]);
        }

        // Restore OMP threads
        omp_set_num_threads(saved_max_threads);

        // Print row
        std::cout << std::left << std::setw(12) << file_count;
        for (int m = 0; m < METHOD_COUNT; ++m) {
            if (method_latencies[m] < 0.0) {
                std::cout << std::setw(14) << "FAILED" << std::setw(14) << "-";
            } else {
                std::cout << std::setw(14) << FormatDouble(method_latencies[m])
                          << std::setw(14) << (FormatDouble(method_ratios[m], 2) + "x");
            }
        }
        std::cout << "\n";
        std::cout.flush();
    }

    std::cout << std::string(96, '-') << "\n";

    // ---- Step 3: Write CSV ----
    std::ofstream csv(opts.out_csv.c_str(), std::ios::out | std::ios::trunc);
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV output: " << opts.out_csv << "\n";
        return 1;
    }
    csv << "file_count,method,latency_ms,comp_ratio\n";
    for (size_t i = 0; i < results.size(); ++i) {
        csv << results[i].file_count << ","
            << kMethodNames[results[i].method] << ","
            << FormatDouble(results[i].latency_ms) << ","
            << FormatDouble(results[i].comp_ratio, 2) << "\n";
    }
    csv.close();
    std::cout << "CSV written to: " << opts.out_csv << "\n";

    return 0;
}
