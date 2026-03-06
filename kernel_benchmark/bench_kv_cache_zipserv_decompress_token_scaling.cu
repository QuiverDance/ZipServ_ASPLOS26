/***************************************************************************
 * KV Cache ZipServ Token Decompression Scaling Benchmark
 *
 * Measures decompression latency for the first N tokens of each KV-cache file.
 * For each token sweep point, compressed chunks are prepared before timing:
 *   - ZipServ compressed chunks
 *   - bitcomp compressed chunks
 *
 * CPU mode compares:
 *   - ZipServ-CPU (host-side reference decompressor in this benchmark)
 *   - bitcomp-CPU (bitcompHostUncompress)
 *
 * GPU mode compares:
 *   - ZipServ-GPU (BF16TripleBitmap_Decompress_API)
 *   - bitcomp-GPU (bitcompUncompress)
 *
 * Optional chunking:
 *   If --chunk_len > 0 and N > chunk_len, split the prefix into
 *   ceil(N / chunk_len) chunks. If N <= chunk_len, chunking is ignored.
 *
 * Fairness:
 *   - Compression and compressed-KV preparation happen before timing.
 *   - Compressed-KV transfer is excluded from timed regions.
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

#include <map>

#include "L_API.cuh"
#include "bench_manifest_utils.h"
#include "csv_writer.h"

#include <nvcomp/native/bitcomp.h>
#include "cuda_timer.h"

// ===== Configuration =====
static const int kDefaultChunkLen = 0;

enum CompressMethod {
    METHOD_SINGLE  = 0,
    METHOD_MT      = 1,
    METHOD_MT_AVX  = 2,
    METHOD_BITCOMP = 3,
    METHOD_COUNT   = 4
};

static const char* kMethodNames[] __attribute__((unused)) = {"Single", "MT", "MT-AVX", "bitcomp"};

enum GPUCompressMethod {
    GPU_METHOD_ZIPSERV = 0,
    GPU_METHOD_BITCOMP = 1,
    GPU_METHOD_COUNT   = 2
};

static const char* kGPUMethodNames[] __attribute__((unused)) = {"ZipServ-GPU", "bitcomp-GPU"};

// ===== Program Options =====
struct ProgramOptions {
    std::string input_dir = "~/saved_kv_cache";
    int warmup = 10;
    int iters = 100;
    int max_files = -1;
    int recursive = 0;
    std::string out_csv = "kv_cache_zipserv_decompress_token_scaling_results.csv";
    std::string ext_filter;
    int device = 0;
    std::string mode = "cpu";   // "cpu" or "gpu"
    std::string token_counts;
    int require_full_token_count = 1;
    int chunk_len = kDefaultChunkLen;
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

struct WorkItem {
    int file_index = -1;
    int64_t seq_begin = 0;
    int64_t seq_count = 0;
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

struct GpuApiEventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

struct GenericResult {
    int token_count = 0;
    int eligible_files = 0;
    int processed_files = 0;
    int requested_chunk_len = 0;
    int effective_chunk_len = 0;
    std::string method_name;
    double latency_ms = 0.0;
    double core_latency_ms = 0.0;
    double comp_ratio = 0.0;
};

int64_t ResolveChunkLenForWorkItem(int64_t seq_count, int requested_chunk_len);

// ===================================================================
// Utility functions (reused from bench_kv_cache_zipserv_compress.cu)
// ===================================================================

void PrintUsage() {
    std::cout
        << "Usage: ./bench_kv_cache_zipserv_decompress_token_scaling [options]\n"
        << "Options:\n"
        << "  --input_dir <path>      Input directory (default: ~/saved_kv_cache)\n"
        << "  --warmup <int>          Warmup iterations (default: 10)\n"
        << "  --iters <int>           Benchmark iterations (default: 100)\n"
        << "  --max_files <int>       Max files after sorting (-1: all, default: -1)\n"
        << "  --recursive <0|1>       Recursive directory traversal (default: 0)\n"
        << "  --out_csv <path>        Output CSV path (default: kv_cache_zipserv_decompress_token_scaling_results.csv)\n"
        << "  --ext_filter <csv>      Extension allow-list, e.g. npy,noext\n"
        << "  --device <int>          CUDA device index (default: 0)\n"
        << "  --mode <cpu|gpu>        Benchmark mode (default: cpu)\n"
        << "  --token_counts <csv>    Token counts for token sweep, e.g. 1,2,4,8\n"
        << "  --require_full_token_count <0|1> Require seq_len >= token_count in token mode (default: 1)\n"
        << "  --chunk_len <int>       Optional token chunk length (0: disabled, default: 0)\n"
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
        else if (arg == "--mode" && i + 1 < argc) { opts->mode = argv[++i]; }
        else if (arg == "--token_counts" && i + 1 < argc) { opts->token_counts = argv[++i]; }
        else if (arg == "--require_full_token_count" && i + 1 < argc) {
            int v = 0; if (!ParseBool01("--require_full_token_count", argv[++i], &v)) return false;
            opts->require_full_token_count = v;
        }
        else if (arg == "--chunk_len" && i + 1 < argc) { opts->chunk_len = std::atoi(argv[++i]); }
        else { std::cerr << "Unknown or incomplete argument: " << arg << "\n"; PrintUsage(); return false; }
    }
    if (opts->warmup < 0 || opts->iters <= 0) {
        std::cerr << "Error: warmup >= 0 and iters > 0 required.\n"; return false;
    }
    if (opts->max_files == 0 || opts->max_files < -1) {
        std::cerr << "Error: --max_files must be -1 or > 0.\n"; return false;
    }
    if (opts->mode != "cpu" && opts->mode != "gpu") {
        std::cerr << "Error: --mode must be 'cpu' or 'gpu', got: " << opts->mode << "\n";
        return false;
    }
    if (opts->chunk_len < 0) {
        std::cerr << "Error: --chunk_len must be >= 0, got: " << opts->chunk_len << "\n";
        return false;
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
// Load BF16 data with zero-padding only (no exponent analysis).
// Used by bitcomp which does not need ZipServ-specific exponent info.
bool LoadBf16(const NpyInfo& info,
              const std::vector<uint8_t>& raw,
              int64_t seq_begin,
              int64_t seq_count,
              std::vector<__nv_bfloat16>* out,
              int* padded_rows_out,
              int* padded_cols_out,
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

    for (int r = 0; r < mapped_rows; ++r) {
        const size_t global_row = seq_row_base + static_cast<size_t>(r);
        const uint16_t* src_row = bf16_bits + global_row * static_cast<size_t>(mapped_cols);
        uint16_t* dst_row = out_data + static_cast<size_t>(r) * static_cast<size_t>(padded_cols);
        std::memcpy(dst_row, src_row, static_cast<size_t>(mapped_cols) * sizeof(uint16_t));
    }
    return true;
}

bool LoadBf16ToBuffer(const NpyInfo& info,
                      const std::vector<uint8_t>& raw,
                      int64_t seq_begin,
                      int64_t seq_count,
                      __nv_bfloat16* out,
                      size_t out_numel,
                      int* padded_rows_out,
                      int* padded_cols_out,
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
    if (input_numel > out_numel) {
        *error = "insufficient_output_capacity";
        return false;
    }

    uint16_t* out_data = reinterpret_cast<uint16_t*>(out);
    std::memset(out_data, 0, input_numel * sizeof(uint16_t));

    const uint16_t* bf16_bits = reinterpret_cast<const uint16_t*>(raw.data() + info.data_offset);
    const size_t seq_row_base = static_cast<size_t>(seq_begin) * static_cast<size_t>(h);
    for (int r = 0; r < mapped_rows; ++r) {
        const size_t global_row = seq_row_base + static_cast<size_t>(r);
        const uint16_t* src_row = bf16_bits + global_row * static_cast<size_t>(mapped_cols);
        uint16_t* dst_row = out_data + static_cast<size_t>(r) * static_cast<size_t>(padded_cols);
        std::memcpy(dst_row, src_row, static_cast<size_t>(mapped_cols) * sizeof(uint16_t));
    }
    return true;
}

// Load BF16 data with zero-padding AND exponent analysis (for ZipServ).
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
// GPU compression buffers and helpers
// ===================================================================

struct GPUCompressBuffers {
    // Device input
    __nv_bfloat16* d_input = nullptr;
    int* d_top_exponents = nullptr;
    int* d_exponent_counts = nullptr;  // [256] exponent histogram
    int* d_phase_state = nullptr;  // [4] fused-kernel phase/control state

    // Device output buffers (same layout as CompressReusableBuffers)
    uint8_t* d_sign_mantissa = nullptr;
    __nv_bfloat16* d_compressed_full = nullptr;
    uint64_t* d_bitmap1 = nullptr;
    uint64_t* d_bitmap2 = nullptr;
    uint64_t* d_bitmap3 = nullptr;
    int* d_tile_offsets = nullptr;
    int* d_tile_offsets_median = nullptr;
    int* d_tile_offsets_global = nullptr;

    // Device temp workspace
    uint8_t* d_temp_sm = nullptr;
    __nv_bfloat16* d_temp_full = nullptr;
    int* d_gt_hf_count = nullptr;
    int* d_gt_full_count = nullptr;
    int* d_hf_offsets = nullptr;
    int* d_full_offsets = nullptr;

    // Device scalar outputs
    int* d_max_hf_count = nullptr;
    int* d_max_full_count = nullptr;
    int* d_total_hf = nullptr;
    int* d_total_full = nullptr;

    int alloc_rows = 0;
    int alloc_cols = 0;
};

bool AllocGPUCompressBuffers(int padded_rows, int padded_cols, GPUCompressBuffers* gb) {
    const int tile_m = 8, tile_m_median = 16, tile_m_global = 64;
    const int tile_k = 8, tile_k_median = 64, tile_k_global = 64;
    const int num_tiles = (padded_rows / tile_m) * (padded_cols / tile_k);
    const int num_median_tiles = (padded_rows / tile_m_median) * (padded_cols / tile_k_median);
    const int num_global_tiles = (padded_rows / tile_m_global) * (padded_cols / tile_k_global);
    const size_t max_elems = (size_t)padded_rows * padded_cols;
    const int max_elem_per_gtile = tile_m_global * tile_k_global;
    const int max_sm_per_gtile = max_elem_per_gtile + 15;
    const int max_full_per_gtile = max_elem_per_gtile + 7;

    cudaError_t ce = cudaSuccess;
    #define GPU_ALLOC(ptr, bytes) \
        ce = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
        if (ce != cudaSuccess) return false;

    GPU_ALLOC(gb->d_input, max_elems * sizeof(__nv_bfloat16));
    GPU_ALLOC(gb->d_top_exponents, 7 * sizeof(int));
    GPU_ALLOC(gb->d_exponent_counts, 256 * sizeof(int));
    GPU_ALLOC(gb->d_phase_state, 4 * sizeof(int));

    GPU_ALLOC(gb->d_sign_mantissa, max_elems > 0 ? max_elems : 1);
    GPU_ALLOC(gb->d_compressed_full, (max_elems > 0 ? max_elems : 1) * sizeof(__nv_bfloat16));
    GPU_ALLOC(gb->d_bitmap1, num_tiles * sizeof(uint64_t));
    GPU_ALLOC(gb->d_bitmap2, num_tiles * sizeof(uint64_t));
    GPU_ALLOC(gb->d_bitmap3, num_tiles * sizeof(uint64_t));
    GPU_ALLOC(gb->d_tile_offsets, num_tiles * 2 * sizeof(int));
    GPU_ALLOC(gb->d_tile_offsets_median, num_median_tiles * 2 * sizeof(int));
    GPU_ALLOC(gb->d_tile_offsets_global, (num_global_tiles + 1) * 2 * sizeof(int));

    GPU_ALLOC(gb->d_temp_sm, (size_t)num_global_tiles * max_sm_per_gtile);
    GPU_ALLOC(gb->d_temp_full, (size_t)num_global_tiles * max_full_per_gtile * sizeof(__nv_bfloat16));
    GPU_ALLOC(gb->d_gt_hf_count, num_global_tiles * sizeof(int));
    GPU_ALLOC(gb->d_gt_full_count, num_global_tiles * sizeof(int));
    GPU_ALLOC(gb->d_hf_offsets, (num_global_tiles + 1) * sizeof(int));
    GPU_ALLOC(gb->d_full_offsets, (num_global_tiles + 1) * sizeof(int));

    GPU_ALLOC(gb->d_max_hf_count, sizeof(int));
    GPU_ALLOC(gb->d_max_full_count, sizeof(int));
    GPU_ALLOC(gb->d_total_hf, sizeof(int));
    GPU_ALLOC(gb->d_total_full, sizeof(int));

    #undef GPU_ALLOC

    gb->alloc_rows = padded_rows;
    gb->alloc_cols = padded_cols;
    return true;
}

void FreeGPUCompressBuffers(GPUCompressBuffers* gb) {
    cudaFree(gb->d_input);
    cudaFree(gb->d_top_exponents);
    cudaFree(gb->d_exponent_counts);
    cudaFree(gb->d_phase_state);
    cudaFree(gb->d_sign_mantissa);
    cudaFree(gb->d_compressed_full);
    cudaFree(gb->d_bitmap1);
    cudaFree(gb->d_bitmap2);
    cudaFree(gb->d_bitmap3);
    cudaFree(gb->d_tile_offsets);
    cudaFree(gb->d_tile_offsets_median);
    cudaFree(gb->d_tile_offsets_global);
    cudaFree(gb->d_temp_sm);
    cudaFree(gb->d_temp_full);
    cudaFree(gb->d_gt_hf_count);
    cudaFree(gb->d_gt_full_count);
    cudaFree(gb->d_hf_offsets);
    cudaFree(gb->d_full_offsets);
    cudaFree(gb->d_max_hf_count);
    cudaFree(gb->d_max_full_count);
    cudaFree(gb->d_total_hf);
    cudaFree(gb->d_total_full);
    *gb = GPUCompressBuffers();
}

constexpr int kGpuPipelineDepth = 3;

struct GpuPipelineSlot {
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    GpuApiEventPair core_event;
    GPUCompressBuffers gb;
    __nv_bfloat16* h_input = nullptr;
    size_t h_input_numel = 0;
    size_t pending_input_bytes = 0;
    int pending_rows = 0;
    int pending_cols = 0;
    bool busy = false;
    bool timing_active = false;
};

bool AllocGpuPipelineSlots(int padded_rows,
                           int padded_cols,
                           std::vector<GpuPipelineSlot>* slots) {
    const size_t max_numel = static_cast<size_t>(padded_rows) * padded_cols;
    slots->assign(kGpuPipelineDepth, GpuPipelineSlot());
    for (int i = 0; i < kGpuPipelineDepth; ++i) {
        GpuPipelineSlot& slot = (*slots)[static_cast<size_t>(i)];
        if (cudaStreamCreate(&slot.stream) != cudaSuccess) return false;
        if (cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming) != cudaSuccess) return false;
        if (cudaMallocHost(reinterpret_cast<void**>(&slot.h_input),
                           (max_numel > 0 ? max_numel : 1) * sizeof(__nv_bfloat16)) != cudaSuccess) {
            return false;
        }
        slot.h_input_numel = max_numel;
        if (!AllocGPUCompressBuffers(padded_rows, padded_cols, &slot.gb)) return false;
    }
    return true;
}

void FreeGpuPipelineSlots(std::vector<GpuPipelineSlot>* slots) {
    for (size_t i = 0; i < slots->size(); ++i) {
        GpuPipelineSlot& slot = (*slots)[i];
        if (slot.core_event.start != nullptr) cudaEventDestroy(slot.core_event.start);
        if (slot.core_event.stop != nullptr) cudaEventDestroy(slot.core_event.stop);
        if (slot.done != nullptr) cudaEventDestroy(slot.done);
        if (slot.stream != nullptr) cudaStreamDestroy(slot.stream);
        if (slot.h_input != nullptr) cudaFreeHost(slot.h_input);
        FreeGPUCompressBuffers(&slot.gb);
    }
    slots->clear();
}

bool DrainZipServSlot(GpuPipelineSlot* slot,
                      CompressionStats* stats,
                      double* total_core_ms,
                      size_t* file_orig,
                      size_t* file_comp) {
    if (!slot->busy) return true;
    const cudaError_t sync_ce = cudaStreamSynchronize(slot->stream);
    if (sync_ce != cudaSuccess) {
        slot->busy = false;
        return false;
    }

    if (total_core_ms != nullptr && slot->timing_active) {
        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, slot->core_event.start, slot->core_event.stop) == cudaSuccess) {
            *total_core_ms += static_cast<double>(elapsed_ms);
        }
        cudaEventDestroy(slot->core_event.start);
        cudaEventDestroy(slot->core_event.stop);
        slot->core_event = GpuApiEventPair();
        slot->timing_active = false;
    } else if (slot->timing_active) {
        cudaEventDestroy(slot->core_event.start);
        cudaEventDestroy(slot->core_event.stop);
        slot->core_event = GpuApiEventPair();
        slot->timing_active = false;
    }

    if (stats != nullptr) {
        int total_hf = 0;
        int total_full = 0;
        cudaMemcpy(&total_hf, slot->gb.d_total_hf, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&total_full, slot->gb.d_total_full, sizeof(int), cudaMemcpyDeviceToHost);

        const int padded_rows = slot->pending_rows;
        const int padded_cols = slot->pending_cols;
        const int nt = (padded_rows / 8) * (padded_cols / 8);
        const int nmt = (padded_rows / 16) * (padded_cols / 64);
        const int ngt = (padded_rows / 64) * (padded_cols / 64);
        const size_t comp = static_cast<size_t>(total_hf)
                          + static_cast<size_t>(total_full) * 2
                          + static_cast<size_t>(nt) * 3 * 8
                          + static_cast<size_t>(nt) * 2 * 4
                          + static_cast<size_t>(nmt) * 2 * 4
                          + static_cast<size_t>(ngt + 1) * 2 * 4
                          + 7 * sizeof(int);
        *file_orig += slot->pending_input_bytes;
        *file_comp += comp;
        stats->total_original_bytes += slot->pending_input_bytes;
        stats->total_compressed_bytes += comp;
    }

    slot->busy = false;
    slot->pending_input_bytes = 0;
    slot->pending_rows = 0;
    slot->pending_cols = 0;
    return true;
}

namespace {
constexpr int kExponentBins = 256;
constexpr int kAnalyzeThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kAnalyzeWarps = kAnalyzeThreads / kWarpSize;
}

int GetAnalyzeBlockCap() {
    static int cached_cap = 0;
    if (cached_cap > 0) return cached_cap;

    int dev = 0;
    int sm_count = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
        sm_count > 0) {
        cached_cap = std::max(1, std::min(256, sm_count * 4));
    } else {
        cached_cap = 128;
    }
    return cached_cap;
}

__global__ void AnalyzeTopExponentsFusedKernel(const __nv_bfloat16* d_input,
                                               int numel,
                                               int* d_counts,
                                               int* d_top_exponents,
                                               int* d_done_counter,
                                               int expected_blocks) {
    __shared__ unsigned int s_counts[kAnalyzeWarps * kExponentBins];
    const int tid = static_cast<int>(threadIdx.x);

    for (int i = tid; i < kAnalyzeWarps * kExponentBins; i += static_cast<int>(blockDim.x)) {
        s_counts[i] = 0U;
    }
    __syncthreads();

    const int warp_id = tid / kWarpSize;
    int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(blockDim.x * gridDim.x);
    while (idx < numel) {
        const uint16_t bits = __bfloat16_as_ushort(d_input[idx]);
        const int exponent = static_cast<int>((bits >> 7) & 0xFF);
        atomicAdd(&s_counts[warp_id * kExponentBins + exponent], 1U);
        idx += stride;
    }
    __syncthreads();

    if (tid < kExponentBins) {
        unsigned int merged = 0;
        for (int w = 0; w < kAnalyzeWarps; ++w) {
            merged += s_counts[w * kExponentBins + tid];
        }
        if (merged > 0) atomicAdd(&d_counts[tid], static_cast<int>(merged));
    }
    __syncthreads();

    if (tid != 0) return;

    __threadfence();
    const int done = atomicAdd(d_done_counter, 1) + 1;
    if (done != expected_blocks) return;

    const int top_n = 7;
    int top_exp[top_n];
    int top_cnt[top_n];
    for (int i = 0; i < top_n; ++i) {
        top_exp[i] = -1;
        top_cnt[i] = -1;
    }

    for (int e = 0; e < 256; ++e) {
        int min_idx = 0;
        for (int i = 1; i < top_n; ++i) {
            if (top_cnt[i] < top_cnt[min_idx]) min_idx = i;
        }
        const int count = d_counts[e];
        if (count > top_cnt[min_idx]) {
            top_cnt[min_idx] = count;
            top_exp[min_idx] = e;
        }
    }

    int original_top[top_n];
    int n_found = 0;
    for (int i = 0; i < top_n; ++i) {
        if (top_exp[i] >= 0) original_top[n_found++] = top_exp[i];
    }
    while (n_found < top_n) {
        original_top[n_found] = 127 - n_found;
        ++n_found;
    }

    for (int i = 1; i < top_n; ++i) {
        const int key = original_top[i];
        int j = i - 1;
        while (j >= 0 && original_top[j] > key) {
            original_top[j + 1] = original_top[j];
            --j;
        }
        original_top[j + 1] = key;
    }

    for (int ci = 0; ci < top_n; ++ci) {
        const int start = original_top[ci];
        if (start < 0 || start + top_n - 1 > 255) continue;
        bool all_exist = true;
        for (int i = 0; i < top_n; ++i) {
            if (d_counts[start + i] <= 0) {
                all_exist = false;
                break;
            }
        }
        if (!all_exist) continue;
        for (int i = 0; i < top_n; ++i) d_top_exponents[i] = start + i;
        return;
    }

    int best_start = original_top[0];
    int max_length = 1;
    int current_length = 1;
    for (int i = 1; i < top_n; ++i) {
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
    if (best_start > 255 - top_n) best_start = 255 - top_n;
    for (int i = 0; i < top_n; ++i) d_top_exponents[i] = best_start + i;
}

cudaError_t AnalyzeTopExponentsGPU(cudaStream_t stream,
                                   const __nv_bfloat16* d_input,
                                   int numel,
                                   int* d_exponent_counts,
                                   int* d_top_exponents,
                                   int* d_analysis_done_counter) {
    if (numel <= 0) return cudaErrorInvalidValue;

    cudaError_t ce = cudaMemsetAsync(
        d_exponent_counts, 0, 256 * sizeof(int), stream);
    if (ce != cudaSuccess) return ce;
    ce = cudaMemsetAsync(d_analysis_done_counter, 0, sizeof(int), stream);
    if (ce != cudaSuccess) return ce;

    const int threads = kAnalyzeThreads;
    int blocks = (numel + threads - 1) / threads;
    if (blocks < 1) blocks = 1;

    const int block_cap = GetAnalyzeBlockCap();
    if (blocks > block_cap) blocks = block_cap;
    AnalyzeTopExponentsFusedKernel<<<blocks, threads, 0, stream>>>(
        d_input, numel, d_exponent_counts, d_top_exponents,
        d_analysis_done_counter, blocks);
    return cudaGetLastError();
}

// GPU bitcomp context (device-side compression)
struct BitcompGPUContext {
    void* d_input = nullptr;
    void* d_output = nullptr;
    size_t* d_compressed_bytes = nullptr;

    size_t max_input_bytes = 0;
    size_t max_output_bytes = 0;

    std::map<size_t, bitcompHandle_t> plan_cache;  // input_bytes -> handle
    cudaStream_t stream = 0;
};

bool AllocBitcompGPUContext(size_t max_input_bytes, cudaStream_t stream,
                            BitcompGPUContext* ctx, std::string* error) {
    ctx->max_input_bytes = max_input_bytes;
    ctx->max_output_bytes = bitcompMaxBuflen(max_input_bytes);
    if (ctx->max_output_bytes == 0) {
        *error = "bitcompMaxBuflen returned 0"; return false;
    }
    ctx->stream = stream;

    cudaError_t ce = cudaSuccess;
    ce = cudaMalloc(&ctx->d_input, max_input_bytes);
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(bitcomp d_input) failed: ") + cudaGetErrorString(ce);
        return false;
    }
    ce = cudaMalloc(&ctx->d_output, ctx->max_output_bytes);
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(bitcomp d_output) failed: ") + cudaGetErrorString(ce);
        return false;
    }
    ce = cudaMalloc(reinterpret_cast<void**>(&ctx->d_compressed_bytes), sizeof(size_t));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(bitcomp d_compressed_bytes) failed: ") + cudaGetErrorString(ce);
        return false;
    }

    return true;
}

bool EnsureBitcompGPUPlan(BitcompGPUContext* ctx, size_t input_bytes, std::string* error) {
    if (input_bytes == 0 || input_bytes > ctx->max_input_bytes) {
        std::ostringstream oss;
        oss << "invalid bitcomp input_bytes=" << input_bytes
            << " (max=" << ctx->max_input_bytes << ")";
        *error = oss.str();
        return false;
    }

    if (ctx->plan_cache.count(input_bytes)) return true;

    bitcompHandle_t handle = nullptr;
    bitcompResult_t st = bitcompCreatePlan(
        &handle, input_bytes,
        BITCOMP_UNSIGNED_16BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_ALGO);
    if (st != BITCOMP_SUCCESS) {
        std::ostringstream oss;
        oss << "bitcompCreatePlan failed: " << static_cast<int>(st)
            << " (input_bytes=" << input_bytes << ")";
        *error = oss.str();
        return false;
    }

    st = bitcompSetStream(handle, ctx->stream);
    if (st != BITCOMP_SUCCESS) {
        bitcompDestroyPlan(handle);
        std::ostringstream oss;
        oss << "bitcompSetStream failed: " << static_cast<int>(st)
            << " (input_bytes=" << input_bytes << ")";
        *error = oss.str();
        return false;
    }

    ctx->plan_cache[input_bytes] = handle;
    return true;
}

void FreeBitcompGPUContext(BitcompGPUContext* ctx) {
    for (std::map<size_t, bitcompHandle_t>::iterator it = ctx->plan_cache.begin();
         it != ctx->plan_cache.end(); ++it) {
        if (it->second != nullptr) bitcompDestroyPlan(it->second);
    }
    ctx->plan_cache.clear();
    cudaFree(ctx->d_input);
    cudaFree(ctx->d_output);
    cudaFree(ctx->d_compressed_bytes);
    *ctx = BitcompGPUContext();
}

// ===================================================================
// bitcomp host compression helpers (CPU-only, no GPU required)
// ===================================================================
const char* BitcompResultToString(bitcompResult_t result) {
    switch (result) {
        case BITCOMP_SUCCESS:                    return "BITCOMP_SUCCESS";
        case BITCOMP_INVALID_PARAMETER:          return "BITCOMP_INVALID_PARAMETER";
        case BITCOMP_INVALID_COMPRESSED_DATA:    return "BITCOMP_INVALID_COMPRESSED_DATA";
        case BITCOMP_INVALID_ALIGNMENT:          return "BITCOMP_INVALID_ALIGNMENT";
        case BITCOMP_INVALID_INPUT_LENGTH:       return "BITCOMP_INVALID_INPUT_LENGTH";
        case BITCOMP_CUDA_KERNEL_LAUNCH_ERROR:   return "BITCOMP_CUDA_KERNEL_LAUNCH_ERROR";
        case BITCOMP_CUDA_API_ERROR:             return "BITCOMP_CUDA_API_ERROR";
        case BITCOMP_UNKNOWN_ERROR:              return "BITCOMP_UNKNOWN_ERROR";
        default:                                 return "BITCOMP_UNRECOGNIZED_ERROR";
    }
}

bool BitcompStatusOk(bitcompResult_t status, const char* what, std::string* error) {
    if (status == BITCOMP_SUCCESS) return true;
    std::ostringstream oss;
    oss << what << " failed: " << BitcompResultToString(status)
        << " (" << static_cast<int>(status) << ")";
    *error = oss.str();
    return false;
}

// Per-thread bitcomp context (pure host memory, no GPU)
struct BitcompHostContext {
    void* output_buf = nullptr;       // 64-bit aligned host buffer
    size_t output_buf_size = 0;
    std::map<size_t, bitcompHandle_t> plan_cache;  // input_bytes → handle
};

bool AllocBitcompHostContext(size_t max_input_bytes, BitcompHostContext* ctx, std::string* error) {
    ctx->output_buf_size = bitcompMaxBuflen(max_input_bytes);
    if (ctx->output_buf_size == 0) {
        *error = "bitcompMaxBuflen returned 0";
        return false;
    }
    ctx->output_buf = aligned_alloc(8, ctx->output_buf_size);
    if (!ctx->output_buf) {
        *error = "aligned_alloc failed for bitcomp output buffer";
        return false;
    }
    return true;
}

bool EnsureBitcompPlan(BitcompHostContext* ctx, size_t input_bytes, std::string* error) {
    if (ctx->plan_cache.count(input_bytes)) return true;
    bitcompHandle_t handle = nullptr;
    bitcompResult_t st = bitcompCreatePlan(
        &handle, input_bytes,
        BITCOMP_UNSIGNED_16BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_ALGO);
    if (!BitcompStatusOk(st, "bitcompCreatePlan", error)) return false;
    ctx->plan_cache[input_bytes] = handle;
    return true;
}

void FreeBitcompHostContext(BitcompHostContext* ctx) {
    for (std::map<size_t, bitcompHandle_t>::iterator it = ctx->plan_cache.begin();
         it != ctx->plan_cache.end(); ++it) {
        if (it->second != nullptr) bitcompDestroyPlan(it->second);
    }
    ctx->plan_cache.clear();
    free(ctx->output_buf);
    ctx->output_buf = nullptr;
    ctx->output_buf_size = 0;
}

// ===================================================================
// Core benchmark logic
// ===================================================================

// Process all files for one benchmark point.
// For each file: split into chunks, then parallel { analyze → compress } per chunk.
bool ProcessCompressionChunk(const WorkItem& work,
                            const PreloadedFile& file,
                            int64_t chunk_len,
                            int64_t chunk_idx,
                            CompressMethod method,
                            CompressReusableBuffers* rb,
                            std::vector<__nv_bfloat16>* bf16_data,
                            BitcompHostContext* bitcomp_ctx,
                            bool collect_stats,
                            size_t* orig_bytes,
                            size_t* comp_bytes) {
    *orig_bytes = 0;
    *comp_bytes = 0;

    const int64_t seq_begin = work.seq_begin + chunk_idx * chunk_len;
    const int64_t seq_count = std::min<int64_t>(chunk_len, work.seq_count - chunk_idx * chunk_len);

    int padded_rows = 0;
    int padded_cols = 0;
    int top_exponents[7];
    std::string error;

    const bool need_analysis = (method != METHOD_BITCOMP);
    const bool load_ok = need_analysis
        ? LoadBf16WithAnalysis(file.info, file.raw,
                               seq_begin, seq_count,
                               bf16_data,
                               &padded_rows, &padded_cols,
                               top_exponents, &error)
        : LoadBf16(file.info, file.raw,
                   seq_begin, seq_count,
                   bf16_data,
                   &padded_rows, &padded_cols, &error);
    if (!load_ok) return false;

    int max_hf = 0;
    int max_full = 0;
    int total_hf = 0;
    int total_full = 0;
    const size_t input_bytes = static_cast<size_t>(padded_rows) * padded_cols * 2;
    if (method == METHOD_BITCOMP) {
        if (bitcomp_ctx == nullptr) return false;
        std::string bc_error;
        if (!EnsureBitcompPlan(bitcomp_ctx, input_bytes, &bc_error)) return false;
        bitcompHandle_t handle = bitcomp_ctx->plan_cache[input_bytes];
        bitcompHostCompressLossless(handle, bf16_data->data(), bitcomp_ctx->output_buf);

        if (collect_stats) {
            size_t out_bytes = 0;
            bitcompGetCompressedSize(bitcomp_ctx->output_buf, &out_bytes);
            *orig_bytes = input_bytes;
            *comp_bytes = out_bytes;
        }
        return true;
    }

    if (method == METHOD_MT_AVX) {
        InitBF16MatrixTripleBitmap_Reuse_SIMD(
            bf16_data->data(), padded_rows, padded_cols,
            8, 16, 64, 8, 64, 64,
            top_exponents,
            rb->sign_mantissa, rb->compressed_full,
            rb->bitmap1, rb->bitmap2, rb->bitmap3,
            rb->tile_offsets, rb->tile_offsets_median, rb->tile_offsets_global,
            rb->temp_sm, rb->temp_full,
            rb->gt_hf_count, rb->gt_full_count,
            rb->hf_offsets, rb->full_offsets,
            max_hf, max_full, total_hf, total_full);
    } else {
        InitBF16MatrixTripleBitmap_Reuse(
            bf16_data->data(), padded_rows, padded_cols,
            8, 16, 64, 8, 64, 64,
            top_exponents,
            rb->sign_mantissa, rb->compressed_full,
            rb->bitmap1, rb->bitmap2, rb->bitmap3,
            rb->tile_offsets, rb->tile_offsets_median, rb->tile_offsets_global,
            rb->temp_sm, rb->temp_full,
            rb->gt_hf_count, rb->gt_full_count,
            rb->hf_offsets, rb->full_offsets,
            max_hf, max_full, total_hf, total_full);
    }

    if (collect_stats) {
        const int nt  = (padded_rows / 8) * (padded_cols / 8);
        const int nmt = (padded_rows / 16) * (padded_cols / 64);
        const int ngt = (padded_rows / 64) * (padded_cols / 64);
        *orig_bytes = input_bytes;
        *comp_bytes = static_cast<size_t>(total_hf)
                    + static_cast<size_t>(total_full) * 2
                    + static_cast<size_t>(nt) * 3 * 8
                    + static_cast<size_t>(nt) * 2 * 4
                    + static_cast<size_t>(nmt) * 2 * 4
                    + static_cast<size_t>(ngt + 1) * 2 * 4
                    + 7 * sizeof(int);
    }
    return true;
}

void ProcessAllFiles(const std::vector<WorkItem>& work_items,
                     const std::vector<PreloadedFile>& files,
                     int requested_chunk_len,
                     CompressMethod method,
                     std::vector<CompressReusableBuffers>& thread_bufs,
                     std::vector<std::vector<__nv_bfloat16>>& thread_bf16_bufs,
                     CompressionStats* stats = nullptr,
                     std::vector<BitcompHostContext>* bitcomp_ctxs = nullptr,
                     int* processed_files_out = nullptr) {
    int processed_files = 0;
    for (size_t fi = 0; fi < work_items.size(); ++fi) {
        const WorkItem& work = work_items[fi];
        const PreloadedFile& file = files[static_cast<size_t>(work.file_index)];
        const int64_t chunk_len = ResolveChunkLenForWorkItem(work.seq_count, requested_chunk_len);
        const int64_t num_chunks = (work.seq_count + chunk_len - 1) / chunk_len;

        size_t file_orig = 0, file_comp = 0;
        int file_failed_chunks = 0;

        if (num_chunks <= 1 || omp_get_max_threads() <= 1) {
            size_t chunk_orig = 0;
            size_t chunk_comp = 0;
            BitcompHostContext* bitcomp_ctx = nullptr;
            if (method == METHOD_BITCOMP && bitcomp_ctxs != nullptr) {
                bitcomp_ctx = &(*bitcomp_ctxs)[0];
            }
            for (int64_t idx = 0; idx < num_chunks; ++idx) {
                if (!ProcessCompressionChunk(work, file, chunk_len, idx, method,
                                             &thread_bufs[0], &thread_bf16_bufs[0],
                                             bitcomp_ctx, stats != nullptr,
                                             &chunk_orig, &chunk_comp)) {
                    file_failed_chunks += 1;
                    continue;
                }
                if (stats != nullptr) {
                    file_orig += chunk_orig;
                    file_comp += chunk_comp;
                }
            }
        } else {
            #pragma omp parallel for schedule(dynamic, 1) reduction(+:file_orig,file_comp,file_failed_chunks)
            for (int64_t idx = 0; idx < num_chunks; ++idx) {
                const int tid = omp_get_thread_num();
                BitcompHostContext* bitcomp_ctx = nullptr;
                if (method == METHOD_BITCOMP && bitcomp_ctxs != nullptr) {
                    bitcomp_ctx = &(*bitcomp_ctxs)[static_cast<size_t>(tid)];
                }
                size_t chunk_orig = 0;
                size_t chunk_comp = 0;
                if (!ProcessCompressionChunk(work, file, chunk_len, idx, method,
                                             &thread_bufs[static_cast<size_t>(tid)],
                                             &thread_bf16_bufs[static_cast<size_t>(tid)],
                                             bitcomp_ctx, stats != nullptr,
                                             &chunk_orig, &chunk_comp)) {
                    file_failed_chunks += 1;
                    continue;
                }
                if (stats != nullptr) {
                    file_orig += chunk_orig;
                    file_comp += chunk_comp;
                }
            }
        }
        if (file_failed_chunks == 0) {
            processed_files += 1;
        }
        if (stats && file_failed_chunks == 0) {
            stats->total_original_bytes += file_orig;
            stats->total_compressed_bytes += file_comp;
        }
    }
    if (processed_files_out != nullptr) {
        *processed_files_out = processed_files;
    }
}

std::string FormatDouble(double x, int precision = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << x;
    return oss.str();
}

int64_t ResolveChunkLenForWorkItem(int64_t seq_count, int requested_chunk_len) {
    if (seq_count <= 0) return 0;
    if (requested_chunk_len <= 0) return seq_count;
    return std::min<int64_t>(seq_count, static_cast<int64_t>(requested_chunk_len));
}

bool ParsePositiveIntCsv(const std::string& spec,
                         std::vector<int>* values,
                         std::string* error) {
    values->clear();
    const std::vector<std::string> tokens = bench_common::SplitByComma(spec);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string token = bench_common::Trim(tokens[i]);
        if (token.empty()) continue;

        int value = 0;
        try {
            value = std::stoi(token);
        } catch (...) {
            *error = "invalid integer in --token_counts: " + token;
            return false;
        }
        if (value <= 0) {
            *error = "--token_counts values must be > 0, got: " + token;
            return false;
        }
        values->push_back(value);
    }
    if (values->empty()) {
        *error = "--token_counts did not contain any valid values";
        return false;
    }
    std::sort(values->begin(), values->end());
    values->erase(std::unique(values->begin(), values->end()), values->end());
    return true;
}

std::vector<int> BuildDefaultTokenCounts(const std::vector<PreloadedFile>& files) {
    std::vector<int> out;
    if (files.empty()) return out;

    int64_t common_max_tokens = files[0].info.shape[0];
    for (size_t i = 1; i < files.size(); ++i) {
        common_max_tokens = std::min(common_max_tokens, files[i].info.shape[0]);
    }
    if (common_max_tokens <= 0) return out;

    int64_t value = 1;
    while (value <= common_max_tokens) {
        out.push_back(static_cast<int>(value));
        if (value > std::numeric_limits<int>::max() / 2) break;
        value *= 2;
    }
    if (out.empty() || out.back() != common_max_tokens) {
        out.push_back(static_cast<int>(common_max_tokens));
    }
    return out;
}

bool BuildWorkItems(int token_count,
                    const std::vector<PreloadedFile>& preloaded,
                    int num_available,
                    bool require_full_token_count,
                    std::vector<WorkItem>* work_items,
                    int* eligible_files,
                    std::string* error) {
    work_items->clear();
    *eligible_files = 0;

    if (token_count <= 0) {
        *error = "token_count must be > 0";
        return false;
    }
    if (num_available <= 0) {
        *error = "no available files";
        return false;
    }

    for (int i = 0; i < num_available; ++i) {
        const int64_t total_tokens = preloaded[static_cast<size_t>(i)].info.shape[0];
        if (total_tokens <= 0) continue;
        if (require_full_token_count && total_tokens < token_count) continue;

        const int64_t seq_count = require_full_token_count
            ? static_cast<int64_t>(token_count)
            : std::min<int64_t>(total_tokens, static_cast<int64_t>(token_count));
        if (seq_count <= 0) continue;

        WorkItem item;
        item.file_index = i;
        item.seq_begin = 0;
        item.seq_count = seq_count;
        work_items->push_back(item);
    }

    *eligible_files = static_cast<int>(work_items->size());
    if (work_items->empty()) {
        std::ostringstream oss;
        oss << "no eligible files for token_count=" << token_count;
        *error = oss.str();
        return false;
    }
    return true;
}

// ===================================================================
// GPU benchmark: ProcessAllFiles_GPU
// Sequentially processes all files/chunks, uploading each chunk H2D
// and launching the GPU compression kernel on the given stream.
// ===================================================================
void ProcessAllFiles_GPU(
    const std::vector<WorkItem>& work_items,
    const std::vector<PreloadedFile>& files,
    int requested_chunk_len,
    GPUCompressMethod method,
    std::vector<GpuPipelineSlot>* zipserv_slots,
    BitcompGPUContext* bc_ctx,
    cudaStream_t bitcomp_stream,
    CompressionStats* stats = nullptr,
    double* total_core_ms = nullptr,
    int* processed_files_out = nullptr) {
    int processed_files = 0;

    for (size_t fi = 0; fi < work_items.size(); ++fi) {
        const WorkItem& work = work_items[fi];
        const PreloadedFile& file = files[static_cast<size_t>(work.file_index)];
        const int64_t chunk_len = ResolveChunkLenForWorkItem(work.seq_count, requested_chunk_len);
        const int64_t num_chunks = (work.seq_count + chunk_len - 1) / chunk_len;
        const int64_t heads = file.info.shape[1];
        const int64_t dim = file.info.shape[2];

        // Reuse host staging vectors for all chunks in this file.
        const int reserve_rows = RoundUpTo64(static_cast<int>(chunk_len * heads));
        const int reserve_cols = RoundUpTo64(static_cast<int>(dim));
        std::vector<__nv_bfloat16> bf16_data;
        bf16_data.reserve(static_cast<size_t>(reserve_rows) * reserve_cols);
        std::string error;
        std::vector<GpuApiEventPair> bitcomp_events;
        size_t file_orig = 0;
        size_t file_comp = 0;
        bool file_success = true;

        for (int64_t idx = 0; idx < num_chunks; ++idx) {
            const int64_t seq_begin = work.seq_begin + idx * chunk_len;
            const int64_t seq_count = std::min<int64_t>(chunk_len, work.seq_count - idx * chunk_len);

            // Load BF16 data on CPU
            int padded_rows = 0, padded_cols = 0;
            error.clear();

            if (method == GPU_METHOD_ZIPSERV) {
                GpuPipelineSlot& slot = (*zipserv_slots)[static_cast<size_t>(idx % zipserv_slots->size())];
                if (!DrainZipServSlot(&slot, stats, total_core_ms, &file_orig, &file_comp)) {
                    file_success = false;
                    break;
                }

                const bool load_pinned_ok = LoadBf16ToBuffer(
                    file.info, file.raw, seq_begin, seq_count,
                    slot.h_input, slot.h_input_numel,
                    &padded_rows, &padded_cols, &error);
                if (!load_pinned_ok) {
                    file_success = false;
                    break;
                }

                const size_t numel = static_cast<size_t>(padded_rows) * padded_cols;
                const size_t input_bytes = numel * sizeof(__nv_bfloat16);
                slot.pending_rows = padded_rows;
                slot.pending_cols = padded_cols;
                slot.pending_input_bytes = input_bytes;

                cudaMemcpyAsync(slot.gb.d_input, slot.h_input, input_bytes,
                                cudaMemcpyHostToDevice, slot.stream);

                if (total_core_ms != nullptr) {
                    if (cudaEventCreate(&slot.core_event.start) != cudaSuccess ||
                        cudaEventCreate(&slot.core_event.stop) != cudaSuccess ||
                        cudaEventRecord(slot.core_event.start, slot.stream) != cudaSuccess) {
                        std::cerr << "ZipServ-GPU timing setup failed\n";
                        if (slot.core_event.start != nullptr) cudaEventDestroy(slot.core_event.start);
                        if (slot.core_event.stop != nullptr) cudaEventDestroy(slot.core_event.stop);
                        slot.core_event = GpuApiEventPair();
                        total_core_ms = nullptr;
                    } else {
                        slot.timing_active = true;
                    }
                }

                const cudaError_t analyze_ce = AnalyzeTopExponentsGPU(
                    slot.stream,
                    slot.gb.d_input,
                    static_cast<int>(numel),
                    slot.gb.d_exponent_counts,
                    slot.gb.d_top_exponents,
                    slot.gb.d_phase_state);
                if (analyze_ce != cudaSuccess) {
                    std::cerr << "AnalyzeTopExponentsGPU failed: "
                              << cudaGetErrorString(analyze_ce) << "\n";
                    file_success = false;
                    break;
                }

                const cudaError_t zipserv_ce = InitBF16MatrixTripleBitmap_GPU(
                    slot.stream, slot.gb.d_input, padded_rows, padded_cols,
                    slot.gb.d_top_exponents,
                    slot.gb.d_sign_mantissa, slot.gb.d_compressed_full,
                    slot.gb.d_bitmap1, slot.gb.d_bitmap2, slot.gb.d_bitmap3,
                    slot.gb.d_tile_offsets, slot.gb.d_tile_offsets_median, slot.gb.d_tile_offsets_global,
                    slot.gb.d_temp_sm, slot.gb.d_temp_full,
                    slot.gb.d_gt_hf_count, slot.gb.d_gt_full_count,
                    slot.gb.d_hf_offsets, slot.gb.d_full_offsets,
                    slot.gb.d_max_hf_count, slot.gb.d_max_full_count,
                    slot.gb.d_total_hf, slot.gb.d_total_full);
                if (zipserv_ce != cudaSuccess) {
                    std::cerr << "InitBF16MatrixTripleBitmap_GPU failed: "
                              << cudaGetErrorString(zipserv_ce) << "\n";
                    file_success = false;
                    break;
                }

                if (slot.timing_active &&
                    cudaEventRecord(slot.core_event.stop, slot.stream) != cudaSuccess) {
                    std::cerr << "ZipServ-GPU timing stop failed\n";
                    cudaEventDestroy(slot.core_event.start);
                    cudaEventDestroy(slot.core_event.stop);
                    slot.core_event = GpuApiEventPair();
                    slot.timing_active = false;
                    total_core_ms = nullptr;
                }
                cudaEventRecord(slot.done, slot.stream);
                slot.busy = true;
            } else {
                const bool load_ok = LoadBf16(file.info, file.raw,
                                              seq_begin, seq_count,
                                              &bf16_data,
                                              &padded_rows, &padded_cols, &error);
                if (!load_ok) {
                    file_success = false;
                    break;
                }
                const size_t numel = static_cast<size_t>(padded_rows) * padded_cols;
                const size_t input_bytes = numel * sizeof(__nv_bfloat16);

                // bitcomp GPU
                std::string bc_error;
                if (!EnsureBitcompGPUPlan(bc_ctx, input_bytes, &bc_error)) {
                    std::cerr << "EnsureBitcompGPUPlan failed: " << bc_error << "\n";
                    file_success = false;
                    break;
                }
                std::map<size_t, bitcompHandle_t>::const_iterator plan_it =
                    bc_ctx->plan_cache.find(input_bytes);
                if (plan_it == bc_ctx->plan_cache.end() || plan_it->second == nullptr) {
                    std::cerr << "bitcomp GPU plan lookup failed for input_bytes="
                              << input_bytes << "\n";
                    file_success = false;
                    break;
                }
                bitcompHandle_t handle = plan_it->second;

                cudaMemcpyAsync(bc_ctx->d_input, bf16_data.data(), input_bytes,
                                cudaMemcpyHostToDevice, bitcomp_stream);
                if (total_core_ms != nullptr) {
                    GpuApiEventPair ev;
                    if (cudaEventCreate(&ev.start) != cudaSuccess ||
                        cudaEventCreate(&ev.stop) != cudaSuccess ||
                        cudaEventRecord(ev.start, bitcomp_stream) != cudaSuccess) {
                        std::cerr << "bitcomp-GPU core timing setup failed\n";
                        if (ev.start != nullptr) cudaEventDestroy(ev.start);
                        if (ev.stop != nullptr) cudaEventDestroy(ev.stop);
                        total_core_ms = nullptr;
                    } else {
                        bitcomp_events.push_back(ev);
                    }
                }
                bitcompCompressLossless(handle, bc_ctx->d_input, bc_ctx->d_output);
                if (total_core_ms != nullptr &&
                    cudaEventRecord(bitcomp_events.back().stop, bitcomp_stream) != cudaSuccess) {
                    std::cerr << "bitcomp-GPU core timing stop failed\n";
                    cudaEventDestroy(bitcomp_events.back().start);
                    cudaEventDestroy(bitcomp_events.back().stop);
                    bitcomp_events.pop_back();
                    total_core_ms = nullptr;
                }

                if (stats) {
                    size_t comp_bytes = 0;
                    bitcompGetCompressedSizeAsync(bc_ctx->d_output, bc_ctx->d_compressed_bytes, bitcomp_stream);
                    cudaMemcpyAsync(&comp_bytes, bc_ctx->d_compressed_bytes, sizeof(size_t),
                                    cudaMemcpyDeviceToHost, bitcomp_stream);
                    cudaStreamSynchronize(bitcomp_stream);
                    file_orig += input_bytes;
                    file_comp += comp_bytes;
                }
            }
        }

        if (method == GPU_METHOD_ZIPSERV) {
            for (size_t si = 0; si < zipserv_slots->size(); ++si) {
                if (!DrainZipServSlot(&(*zipserv_slots)[si], stats, total_core_ms, &file_orig, &file_comp)) {
                    file_success = false;
                }
            }
        } else {
            cudaStreamSynchronize(bitcomp_stream);
        }

        if (file_success) {
            processed_files += 1;
        }
        for (size_t i = 0; i < bitcomp_events.size(); ++i) {
            if (file_success && total_core_ms != nullptr) {
                float elapsed_ms = 0.0f;
                if (cudaEventElapsedTime(&elapsed_ms, bitcomp_events[i].start, bitcomp_events[i].stop) == cudaSuccess) {
                    *total_core_ms += static_cast<double>(elapsed_ms);
                }
            }
            cudaEventDestroy(bitcomp_events[i].start);
            cudaEventDestroy(bitcomp_events[i].stop);
        }
    }

    cudaStreamSynchronize(bitcomp_stream);
    if (processed_files_out != nullptr) {
        *processed_files_out = processed_files;
    }
}

enum CPUDecompressMethod {
    CPU_DECOMP_ZIPSERV = 0,
    CPU_DECOMP_BITCOMP = 1,
    CPU_DECOMP_COUNT   = 2
};

static const char* kCPUDecompMethodNames[] = {"ZipServ-CPU", "bitcomp-CPU"};

enum GPUDecompressMethod {
    GPU_DECOMP_ZIPSERV = 0,
    GPU_DECOMP_BITCOMP = 1,
    GPU_DECOMP_COUNT   = 2
};

static const char* kGPUDecompMethodNames[] = {"ZipServ-GPU", "bitcomp-GPU"};

struct ZipServChunk {
    int rows = 0;
    int cols = 0;
    int num_global_tiles = 0;
    int max_high_freq_count = 0;
    int max_full_count = 0;
    int high_freq_count = 0;
    int full_count = 0;
    uint8_t start_exp = 0;
    size_t input_bytes = 0;
    size_t compressed_bytes = 0;

    std::vector<uint8_t> sign_mantissa;
    std::vector<__nv_bfloat16> compressed_full;
    std::vector<uint64_t> bitmap1;
    std::vector<uint64_t> bitmap2;
    std::vector<uint64_t> bitmap3;
    std::vector<int> tile_offsets_median;
    std::vector<int> tile_offsets_global;
};

struct BitcompChunk {
    size_t input_bytes = 0;
    size_t compressed_bytes = 0;
    std::vector<uint8_t> compressed;
};

struct FileChunkRange {
    size_t begin = 0;
    size_t end = 0;
};

struct PreparedPoint {
    int token_count = 0;
    int eligible_files = 0;
    int processed_files = 0;
    int requested_chunk_len = 0;
    int effective_chunk_len = 0;
    size_t max_input_bytes = 0;
    size_t max_output_numel = 0;

    size_t zipserv_original_bytes = 0;
    size_t zipserv_compressed_bytes = 0;
    size_t bitcomp_original_bytes = 0;
    size_t bitcomp_compressed_bytes = 0;
    double zipserv_ratio = 0.0;
    double bitcomp_ratio = 0.0;

    std::vector<FileChunkRange> file_ranges;
    std::vector<ZipServChunk> zipserv_chunks;
    std::vector<BitcompChunk> bitcomp_chunks;
};

struct ZipServDeviceChunk {
    int rows = 0;
    int cols = 0;
    int max_high_freq_count = 0;
    int max_full_count = 0;
    uint8_t start_exp = 0;

    uint8_t* d_sign_mantissa = nullptr;
    __nv_bfloat16* d_compressed_full = nullptr;
    uint64_t* d_bitmap1 = nullptr;
    uint64_t* d_bitmap2 = nullptr;
    uint64_t* d_bitmap3 = nullptr;
    int* d_tile_offsets_median = nullptr;
    int* d_tile_offsets_global = nullptr;
};

struct BitcompDeviceChunk {
    size_t input_bytes = 0;
    size_t compressed_bytes = 0;
    uint8_t* d_compressed = nullptr;
};

struct PreparedPointGPU {
    size_t max_input_bytes = 0;
    __nv_bfloat16* d_output = nullptr;

    std::vector<FileChunkRange> file_ranges;
    std::vector<ZipServDeviceChunk> zipserv_chunks;
    std::vector<BitcompDeviceChunk> bitcomp_chunks;
};

struct BitcompGPUDecompressContext {
    cudaStream_t stream = nullptr;
    std::map<size_t, bitcompHandle_t> plan_cache;
};

bool CheckCuda(cudaError_t ce, const std::string& what, std::string* error) {
    if (ce == cudaSuccess) return true;
    *error = what + ": " + cudaGetErrorString(ce);
    return false;
}

size_t ComputeZipServCompressedBytes(
    int rows,
    int cols,
    int num_global_tiles,
    int total_hf,
    int total_full) {
    const int num_tiles = (rows / 8) * (cols / 8);
    const int num_median_tiles = (rows / 16) * (cols / 64);
    return static_cast<size_t>(total_hf)
         + static_cast<size_t>(total_full) * sizeof(__nv_bfloat16)
         + static_cast<size_t>(num_tiles) * 3 * sizeof(uint64_t)
         + static_cast<size_t>(num_tiles) * 2 * sizeof(int)  // small tile offsets
         + static_cast<size_t>(num_median_tiles) * 2 * sizeof(int)
         + static_cast<size_t>(num_global_tiles + 1) * 2 * sizeof(int)
         + 7 * sizeof(int);
}

bool PrepareCompressedPoint(
    int token_count,
    const std::vector<WorkItem>& work_items,
    const std::vector<PreloadedFile>& preloaded,
    int requested_chunk_len,
    int effective_chunk_len,
    int benchmark_max_padded_rows,
    int benchmark_max_padded_cols,
    PreparedPoint* out,
    std::string* error) {
    *out = PreparedPoint();
    out->token_count = token_count;
    out->eligible_files = static_cast<int>(work_items.size());
    out->requested_chunk_len = requested_chunk_len;
    out->effective_chunk_len = effective_chunk_len;

    if (work_items.empty()) {
        *error = "no work items";
        return false;
    }

    CompressReusableBuffers rb;
    if (!AllocReusableBuffers(benchmark_max_padded_rows, benchmark_max_padded_cols, &rb)) {
        *error = "AllocReusableBuffers failed";
        return false;
    }

    BitcompHostContext bitcomp_ctx;
    const size_t max_input_bytes =
        static_cast<size_t>(benchmark_max_padded_rows) *
        static_cast<size_t>(benchmark_max_padded_cols) *
        sizeof(__nv_bfloat16);
    if (!AllocBitcompHostContext(max_input_bytes, &bitcomp_ctx, error)) {
        FreeReusableBuffers(&rb);
        return false;
    }

    std::vector<__nv_bfloat16> bf16_data;
    bf16_data.reserve(static_cast<size_t>(benchmark_max_padded_rows) *
                      static_cast<size_t>(benchmark_max_padded_cols));

    for (size_t wi = 0; wi < work_items.size(); ++wi) {
        const WorkItem& work = work_items[wi];
        const PreloadedFile& file = preloaded[static_cast<size_t>(work.file_index)];
        const int64_t chunk_len = ResolveChunkLenForWorkItem(work.seq_count, requested_chunk_len);
        const int64_t num_chunks = (work.seq_count + chunk_len - 1) / chunk_len;

        FileChunkRange range;
        range.begin = out->zipserv_chunks.size();

        for (int64_t ci = 0; ci < num_chunks; ++ci) {
            const int64_t seq_begin = work.seq_begin + ci * chunk_len;
            const int64_t seq_count = std::min<int64_t>(chunk_len, work.seq_count - ci * chunk_len);

            int padded_rows = 0;
            int padded_cols = 0;
            int top_exponents[7];
            if (!LoadBf16WithAnalysis(
                    file.info,
                    file.raw,
                    seq_begin,
                    seq_count,
                    &bf16_data,
                    &padded_rows,
                    &padded_cols,
                    top_exponents,
                    error)) {
                FreeBitcompHostContext(&bitcomp_ctx);
                FreeReusableBuffers(&rb);
                return false;
            }

            int max_hf = 0;
            int max_full = 0;
            int total_hf = 0;
            int total_full = 0;
            const int num_global_tiles = InitBF16MatrixTripleBitmap_Reuse_SIMD(
                bf16_data.data(),
                padded_rows,
                padded_cols,
                8,
                16,
                64,
                8,
                64,
                64,
                top_exponents,
                rb.sign_mantissa,
                rb.compressed_full,
                rb.bitmap1,
                rb.bitmap2,
                rb.bitmap3,
                rb.tile_offsets,
                rb.tile_offsets_median,
                rb.tile_offsets_global,
                rb.temp_sm,
                rb.temp_full,
                rb.gt_hf_count,
                rb.gt_full_count,
                rb.hf_offsets,
                rb.full_offsets,
                max_hf,
                max_full,
                total_hf,
                total_full);
            if (num_global_tiles <= 0) {
                std::ostringstream oss;
                oss << "InitBF16MatrixTripleBitmap_Reuse_SIMD failed (token=" << token_count
                    << ", file_index=" << work.file_index << ", chunk_idx=" << ci << ")";
                *error = oss.str();
                FreeBitcompHostContext(&bitcomp_ctx);
                FreeReusableBuffers(&rb);
                return false;
            }

            ZipServChunk zchunk;
            zchunk.rows = padded_rows;
            zchunk.cols = padded_cols;
            zchunk.num_global_tiles = num_global_tiles;
            zchunk.max_high_freq_count = max_hf;
            zchunk.max_full_count = max_full;
            zchunk.high_freq_count = total_hf;
            zchunk.full_count = total_full;
            zchunk.start_exp = static_cast<uint8_t>(top_exponents[0] - 1);
            zchunk.input_bytes = static_cast<size_t>(padded_rows) * static_cast<size_t>(padded_cols) * sizeof(__nv_bfloat16);

            const int num_tiles = (padded_rows / 8) * (padded_cols / 8);
            const int num_median_tiles = (padded_rows / 16) * (padded_cols / 64);

            zchunk.sign_mantissa.assign(
                rb.sign_mantissa,
                rb.sign_mantissa + static_cast<size_t>(total_hf));
            zchunk.compressed_full.assign(
                rb.compressed_full,
                rb.compressed_full + static_cast<size_t>(total_full));
            zchunk.bitmap1.assign(
                rb.bitmap1,
                rb.bitmap1 + static_cast<size_t>(num_tiles));
            zchunk.bitmap2.assign(
                rb.bitmap2,
                rb.bitmap2 + static_cast<size_t>(num_tiles));
            zchunk.bitmap3.assign(
                rb.bitmap3,
                rb.bitmap3 + static_cast<size_t>(num_tiles));
            zchunk.tile_offsets_median.assign(
                rb.tile_offsets_median,
                rb.tile_offsets_median + static_cast<size_t>(num_median_tiles) * 2);
            zchunk.tile_offsets_global.assign(
                rb.tile_offsets_global,
                rb.tile_offsets_global + static_cast<size_t>(num_global_tiles + 1) * 2);
            zchunk.compressed_bytes = ComputeZipServCompressedBytes(
                padded_rows, padded_cols, num_global_tiles, total_hf, total_full);

            out->zipserv_original_bytes += zchunk.input_bytes;
            out->zipserv_compressed_bytes += zchunk.compressed_bytes;
            out->max_input_bytes = std::max(out->max_input_bytes, zchunk.input_bytes);
            out->max_output_numel = std::max(
                out->max_output_numel,
                static_cast<size_t>(padded_rows) * static_cast<size_t>(padded_cols));
            out->zipserv_chunks.push_back(std::move(zchunk));

            std::string bc_error;
            if (!EnsureBitcompPlan(&bitcomp_ctx, out->zipserv_chunks.back().input_bytes, &bc_error)) {
                *error = std::string("EnsureBitcompPlan failed: ") + bc_error;
                FreeBitcompHostContext(&bitcomp_ctx);
                FreeReusableBuffers(&rb);
                return false;
            }
            bitcompHandle_t handle = bitcomp_ctx.plan_cache[out->zipserv_chunks.back().input_bytes];
            bitcompResult_t st = bitcompHostCompressLossless(handle, bf16_data.data(), bitcomp_ctx.output_buf);
            if (!BitcompStatusOk(st, "bitcompHostCompressLossless", error)) {
                FreeBitcompHostContext(&bitcomp_ctx);
                FreeReusableBuffers(&rb);
                return false;
            }

            size_t bitcomp_size = 0;
            st = bitcompGetCompressedSize(bitcomp_ctx.output_buf, &bitcomp_size);
            if (!BitcompStatusOk(st, "bitcompGetCompressedSize", error)) {
                FreeBitcompHostContext(&bitcomp_ctx);
                FreeReusableBuffers(&rb);
                return false;
            }
            if (bitcomp_size == 0) {
                *error = "bitcompGetCompressedSize returned zero";
                FreeBitcompHostContext(&bitcomp_ctx);
                FreeReusableBuffers(&rb);
                return false;
            }

            BitcompChunk bchunk;
            bchunk.input_bytes = out->zipserv_chunks.back().input_bytes;
            bchunk.compressed_bytes = bitcomp_size;
            bchunk.compressed.resize(bitcomp_size);
            std::memcpy(bchunk.compressed.data(), bitcomp_ctx.output_buf, bitcomp_size);

            out->bitcomp_original_bytes += bchunk.input_bytes;
            out->bitcomp_compressed_bytes += bchunk.compressed_bytes;
            out->bitcomp_chunks.push_back(std::move(bchunk));
        }

        range.end = out->zipserv_chunks.size();
        if (range.end > range.begin) {
            out->file_ranges.push_back(range);
        }
    }

    out->processed_files = static_cast<int>(out->file_ranges.size());
    out->zipserv_ratio = (out->zipserv_compressed_bytes > 0)
        ? static_cast<double>(out->zipserv_original_bytes) /
          static_cast<double>(out->zipserv_compressed_bytes)
        : 0.0;
    out->bitcomp_ratio = (out->bitcomp_compressed_bytes > 0)
        ? static_cast<double>(out->bitcomp_original_bytes) /
          static_cast<double>(out->bitcomp_compressed_bytes)
        : 0.0;

    FreeBitcompHostContext(&bitcomp_ctx);
    FreeReusableBuffers(&rb);
    return true;
}

bool DecompressZipServChunkCPU(
    const ZipServChunk& chunk,
    __nv_bfloat16* output,
    size_t output_numel,
    std::string* error) {
    const size_t required_numel =
        static_cast<size_t>(chunk.rows) * static_cast<size_t>(chunk.cols);
    if (output_numel < required_numel) {
        *error = "zipserv cpu output buffer is too small";
        return false;
    }

    const int num_global_tiles_m = chunk.rows / 64;
    const int num_global_tiles_k = chunk.cols / 64;
    const int small_per_global = (64 / 8) * (64 / 8);

    for (int gtm = 0; gtm < num_global_tiles_m; ++gtm) {
        for (int gtk = 0; gtk < num_global_tiles_k; ++gtk) {
            const int global_tile_idx = gtm * num_global_tiles_k + gtk;
            const size_t global_offset_idx = static_cast<size_t>(global_tile_idx) * 2;
            if (global_offset_idx + 1 >= chunk.tile_offsets_global.size()) {
                *error = "tile_offsets_global out-of-range";
                return false;
            }

            int high_freq_idx = chunk.tile_offsets_global[global_offset_idx];
            int full_idx = chunk.tile_offsets_global[global_offset_idx + 1];
            int tile_idx = global_tile_idx * small_per_global;

            const int global_row_start = gtm * 64;
            const int global_col_start = gtk * 64;

            for (int median_tile_m = 0; median_tile_m < (64 / 16); ++median_tile_m) {
                const int median_row_start = global_row_start + median_tile_m * 16;
                const int median_col_start = global_col_start;
                for (int local_tile_m_group = 0; local_tile_m_group < (16 / 8); local_tile_m_group += 2) {
                    for (int local_tile_k_group = 0; local_tile_k_group < (64 / 8); local_tile_k_group += 2) {
                        for (int j = 0; j < 2; ++j) {
                            for (int i = 0; i < 2; ++i) {
                                const int local_tile_k = local_tile_k_group + j;
                                const int local_tile_m = local_tile_m_group + i;

                                const int row_start = median_row_start + local_tile_m * 8;
                                const int col_start = median_col_start + local_tile_k * 8;
                                if (tile_idx < 0 ||
                                    static_cast<size_t>(tile_idx) >= chunk.bitmap1.size() ||
                                    static_cast<size_t>(tile_idx) >= chunk.bitmap2.size() ||
                                    static_cast<size_t>(tile_idx) >= chunk.bitmap3.size()) {
                                    *error = "bitmap tile index out-of-range";
                                    return false;
                                }

                                const uint64_t bitmap1 = chunk.bitmap1[static_cast<size_t>(tile_idx)];
                                const uint64_t bitmap2 = chunk.bitmap2[static_cast<size_t>(tile_idx)];
                                const uint64_t bitmap3 = chunk.bitmap3[static_cast<size_t>(tile_idx)];
                                ++tile_idx;

                                const uint64_t high_freq_indicator = bitmap1 | bitmap2 | bitmap3;
                                for (int pos = 0; pos < 64; ++pos) {
                                    const int row_offset = pos / 8;
                                    const int col_offset = pos % 8;
                                    const int row = row_start + row_offset;
                                    const int col = col_start + col_offset;

                                    const size_t out_idx =
                                        static_cast<size_t>(row) * static_cast<size_t>(chunk.cols) +
                                        static_cast<size_t>(col);
                                    if (out_idx >= required_numel) {
                                        *error = "zipserv cpu output index out-of-range";
                                        return false;
                                    }

                                    const bool is_high_freq = ((high_freq_indicator >> pos) & 1ULL) != 0;
                                    if (is_high_freq) {
                                        if (high_freq_idx < 0 ||
                                            static_cast<size_t>(high_freq_idx) >= chunk.sign_mantissa.size()) {
                                            *error = "zipserv cpu sign_mantissa index out-of-range";
                                            return false;
                                        }
                                        const uint8_t combined =
                                            chunk.sign_mantissa[static_cast<size_t>(high_freq_idx++)];
                                        const uint8_t sign = (combined >> 7) & 0x1;
                                        const uint8_t mantissa = combined & 0x7F;
                                        const uint8_t code =
                                            static_cast<uint8_t>(((bitmap3 >> pos) & 1ULL) << 2 |
                                                                 ((bitmap2 >> pos) & 1ULL) << 1 |
                                                                 ((bitmap1 >> pos) & 1ULL));
                                        const uint8_t exponent =
                                            static_cast<uint8_t>(chunk.start_exp + code);
                                        const uint16_t bf16_bits =
                                            static_cast<uint16_t>((sign & 0x1) << 15) |
                                            static_cast<uint16_t>(exponent) << 7 |
                                            static_cast<uint16_t>(mantissa);
                                        output[out_idx] = __ushort_as_bfloat16(bf16_bits);
                                    } else {
                                        if (full_idx < 0 ||
                                            static_cast<size_t>(full_idx) >= chunk.compressed_full.size()) {
                                            *error = "zipserv cpu compressed_full index out-of-range";
                                            return false;
                                        }
                                        output[out_idx] =
                                            chunk.compressed_full[static_cast<size_t>(full_idx++)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool PrepareBitcompHostPlans(
    const PreparedPoint& prepared,
    BitcompHostContext* bitcomp_ctx,
    std::string* error) {
    for (size_t i = 0; i < prepared.bitcomp_chunks.size(); ++i) {
        std::string plan_error;
        if (!EnsureBitcompPlan(bitcomp_ctx, prepared.bitcomp_chunks[i].input_bytes, &plan_error)) {
            *error = std::string("EnsureBitcompPlan failed: ") + plan_error;
            return false;
        }
    }
    return true;
}

bool RunZipServCpuOnce(
    const PreparedPoint& prepared,
    __nv_bfloat16* output,
    size_t output_numel,
    int* processed_files,
    std::string* error) {
    int processed = 0;
    for (size_t fi = 0; fi < prepared.file_ranges.size(); ++fi) {
        const FileChunkRange& range = prepared.file_ranges[fi];
        bool file_ok = true;
        for (size_t ci = range.begin; ci < range.end; ++ci) {
            if (!DecompressZipServChunkCPU(prepared.zipserv_chunks[ci], output, output_numel, error)) {
                file_ok = false;
                break;
            }
        }
        if (file_ok) {
            processed += 1;
        } else {
            return false;
        }
    }
    *processed_files = processed;
    return true;
}

bool RunBitcompCpuOnce(
    const PreparedPoint& prepared,
    BitcompHostContext* bitcomp_ctx,
    uint8_t* output,
    size_t output_bytes,
    int* processed_files,
    std::string* error) {
    int processed = 0;
    for (size_t fi = 0; fi < prepared.file_ranges.size(); ++fi) {
        const FileChunkRange& range = prepared.file_ranges[fi];
        bool file_ok = true;
        for (size_t ci = range.begin; ci < range.end; ++ci) {
            const BitcompChunk& chunk = prepared.bitcomp_chunks[ci];
            if (chunk.input_bytes > output_bytes) {
                *error = "bitcomp cpu output buffer is too small";
                return false;
            }
            std::string ensure_error;
            if (!EnsureBitcompPlan(bitcomp_ctx, chunk.input_bytes, &ensure_error)) {
                *error = std::string("EnsureBitcompPlan failed: ") + ensure_error;
                return false;
            }
            bitcompHandle_t handle = bitcomp_ctx->plan_cache[chunk.input_bytes];
            bitcompResult_t st = bitcompHostUncompress(handle, chunk.compressed.data(), output);
            if (!BitcompStatusOk(st, "bitcompHostUncompress", error)) {
                file_ok = false;
                break;
            }
        }
        if (file_ok) {
            processed += 1;
        } else {
            return false;
        }
    }
    *processed_files = processed;
    return true;
}

void FreePreparedPointGPU(PreparedPointGPU* gpu) {
    for (size_t i = 0; i < gpu->zipserv_chunks.size(); ++i) {
        cudaFree(gpu->zipserv_chunks[i].d_sign_mantissa);
        cudaFree(gpu->zipserv_chunks[i].d_compressed_full);
        cudaFree(gpu->zipserv_chunks[i].d_bitmap1);
        cudaFree(gpu->zipserv_chunks[i].d_bitmap2);
        cudaFree(gpu->zipserv_chunks[i].d_bitmap3);
        cudaFree(gpu->zipserv_chunks[i].d_tile_offsets_median);
        cudaFree(gpu->zipserv_chunks[i].d_tile_offsets_global);
    }
    for (size_t i = 0; i < gpu->bitcomp_chunks.size(); ++i) {
        cudaFree(gpu->bitcomp_chunks[i].d_compressed);
    }
    cudaFree(gpu->d_output);
    *gpu = PreparedPointGPU();
}

bool PreparePointForGPU(
    const PreparedPoint& prepared,
    PreparedPointGPU* gpu,
    std::string* error) {
    *gpu = PreparedPointGPU();
    gpu->max_input_bytes = prepared.max_input_bytes;
    gpu->file_ranges = prepared.file_ranges;
    if (gpu->max_input_bytes == 0) {
        *error = "max_input_bytes is zero";
        return false;
    }

    cudaError_t ce = cudaMalloc(
        reinterpret_cast<void**>(&gpu->d_output),
        std::max<size_t>(gpu->max_input_bytes, 1));
    if (ce != cudaSuccess) {
        *error = std::string("cudaMalloc(d_output) failed: ") + cudaGetErrorString(ce);
        return false;
    }

    gpu->zipserv_chunks.reserve(prepared.zipserv_chunks.size());
    for (size_t i = 0; i < prepared.zipserv_chunks.size(); ++i) {
        const ZipServChunk& src = prepared.zipserv_chunks[i];
        ZipServDeviceChunk dst;
        dst.rows = src.rows;
        dst.cols = src.cols;
        dst.max_high_freq_count = src.max_high_freq_count;
        dst.max_full_count = src.max_full_count;
        dst.start_exp = src.start_exp;

        const size_t sign_bytes = src.sign_mantissa.size();
        const size_t full_bytes = src.compressed_full.size() * sizeof(__nv_bfloat16);
        const size_t bitmap_bytes = src.bitmap1.size() * sizeof(uint64_t);
        const size_t median_bytes = src.tile_offsets_median.size() * sizeof(int);
        const size_t global_bytes = src.tile_offsets_global.size() * sizeof(int);

        ce = cudaMalloc(reinterpret_cast<void**>(&dst.d_sign_mantissa), std::max<size_t>(sign_bytes, 1));
        if (ce != cudaSuccess) {
            *error = std::string("cudaMalloc(zipserv sign_mantissa) failed: ") + cudaGetErrorString(ce);
            FreePreparedPointGPU(gpu);
            return false;
        }
        ce = cudaMalloc(reinterpret_cast<void**>(&dst.d_compressed_full), std::max<size_t>(full_bytes, 1));
        if (ce != cudaSuccess) {
            *error = std::string("cudaMalloc(zipserv compressed_full) failed: ") + cudaGetErrorString(ce);
            FreePreparedPointGPU(gpu);
            return false;
        }
        ce = cudaMalloc(reinterpret_cast<void**>(&dst.d_bitmap1), std::max<size_t>(bitmap_bytes, 1));
        if (ce != cudaSuccess) {
            *error = std::string("cudaMalloc(zipserv bitmap1) failed: ") + cudaGetErrorString(ce);
            FreePreparedPointGPU(gpu);
            return false;
        }
        ce = cudaMalloc(reinterpret_cast<void**>(&dst.d_bitmap2), std::max<size_t>(bitmap_bytes, 1));
        if (ce != cudaSuccess) {
            *error = std::string("cudaMalloc(zipserv bitmap2) failed: ") + cudaGetErrorString(ce);
            FreePreparedPointGPU(gpu);
            return false;
        }
        ce = cudaMalloc(reinterpret_cast<void**>(&dst.d_bitmap3), std::max<size_t>(bitmap_bytes, 1));
        if (ce != cudaSuccess) {
            *error = std::string("cudaMalloc(zipserv bitmap3) failed: ") + cudaGetErrorString(ce);
            FreePreparedPointGPU(gpu);
            return false;
        }
        ce = cudaMalloc(reinterpret_cast<void**>(&dst.d_tile_offsets_median), std::max<size_t>(median_bytes, 1));
        if (ce != cudaSuccess) {
            *error = std::string("cudaMalloc(zipserv tile_offsets_median) failed: ") + cudaGetErrorString(ce);
            FreePreparedPointGPU(gpu);
            return false;
        }
        ce = cudaMalloc(reinterpret_cast<void**>(&dst.d_tile_offsets_global), std::max<size_t>(global_bytes, 1));
        if (ce != cudaSuccess) {
            *error = std::string("cudaMalloc(zipserv tile_offsets_global) failed: ") + cudaGetErrorString(ce);
            FreePreparedPointGPU(gpu);
            return false;
        }

        if (sign_bytes > 0) {
            ce = cudaMemcpy(dst.d_sign_mantissa, src.sign_mantissa.data(), sign_bytes, cudaMemcpyHostToDevice);
            if (ce != cudaSuccess) {
                *error = std::string("cudaMemcpy(zipserv sign_mantissa) failed: ") + cudaGetErrorString(ce);
                FreePreparedPointGPU(gpu);
                return false;
            }
        }
        if (full_bytes > 0) {
            ce = cudaMemcpy(dst.d_compressed_full, src.compressed_full.data(), full_bytes, cudaMemcpyHostToDevice);
            if (ce != cudaSuccess) {
                *error = std::string("cudaMemcpy(zipserv compressed_full) failed: ") + cudaGetErrorString(ce);
                FreePreparedPointGPU(gpu);
                return false;
            }
        }
        if (bitmap_bytes > 0) {
            ce = cudaMemcpy(dst.d_bitmap1, src.bitmap1.data(), bitmap_bytes, cudaMemcpyHostToDevice);
            if (ce != cudaSuccess) {
                *error = std::string("cudaMemcpy(zipserv bitmap1) failed: ") + cudaGetErrorString(ce);
                FreePreparedPointGPU(gpu);
                return false;
            }
            ce = cudaMemcpy(dst.d_bitmap2, src.bitmap2.data(), bitmap_bytes, cudaMemcpyHostToDevice);
            if (ce != cudaSuccess) {
                *error = std::string("cudaMemcpy(zipserv bitmap2) failed: ") + cudaGetErrorString(ce);
                FreePreparedPointGPU(gpu);
                return false;
            }
            ce = cudaMemcpy(dst.d_bitmap3, src.bitmap3.data(), bitmap_bytes, cudaMemcpyHostToDevice);
            if (ce != cudaSuccess) {
                *error = std::string("cudaMemcpy(zipserv bitmap3) failed: ") + cudaGetErrorString(ce);
                FreePreparedPointGPU(gpu);
                return false;
            }
        }
        if (median_bytes > 0) {
            ce = cudaMemcpy(
                dst.d_tile_offsets_median,
                src.tile_offsets_median.data(),
                median_bytes,
                cudaMemcpyHostToDevice);
            if (ce != cudaSuccess) {
                *error = std::string("cudaMemcpy(zipserv tile_offsets_median) failed: ") + cudaGetErrorString(ce);
                FreePreparedPointGPU(gpu);
                return false;
            }
        }
        if (global_bytes > 0) {
            ce = cudaMemcpy(
                dst.d_tile_offsets_global,
                src.tile_offsets_global.data(),
                global_bytes,
                cudaMemcpyHostToDevice);
            if (ce != cudaSuccess) {
                *error = std::string("cudaMemcpy(zipserv tile_offsets_global) failed: ") + cudaGetErrorString(ce);
                FreePreparedPointGPU(gpu);
                return false;
            }
        }

        gpu->zipserv_chunks.push_back(dst);
    }

    gpu->bitcomp_chunks.reserve(prepared.bitcomp_chunks.size());
    for (size_t i = 0; i < prepared.bitcomp_chunks.size(); ++i) {
        const BitcompChunk& src = prepared.bitcomp_chunks[i];
        BitcompDeviceChunk dst;
        dst.input_bytes = src.input_bytes;
        dst.compressed_bytes = src.compressed_bytes;
        ce = cudaMalloc(reinterpret_cast<void**>(&dst.d_compressed), std::max<size_t>(src.compressed_bytes, 1));
        if (ce != cudaSuccess) {
            *error = std::string("cudaMalloc(bitcomp compressed) failed: ") + cudaGetErrorString(ce);
            FreePreparedPointGPU(gpu);
            return false;
        }
        if (src.compressed_bytes > 0) {
            ce = cudaMemcpy(
                dst.d_compressed,
                src.compressed.data(),
                src.compressed_bytes,
                cudaMemcpyHostToDevice);
            if (ce != cudaSuccess) {
                *error = std::string("cudaMemcpy(bitcomp compressed) failed: ") + cudaGetErrorString(ce);
                FreePreparedPointGPU(gpu);
                return false;
            }
        }
        gpu->bitcomp_chunks.push_back(dst);
    }

    return true;
}

void FreeBitcompGPUDecompressContext(BitcompGPUDecompressContext* ctx) {
    for (std::map<size_t, bitcompHandle_t>::iterator it = ctx->plan_cache.begin();
         it != ctx->plan_cache.end(); ++it) {
        if (it->second != nullptr) {
            bitcompDestroyPlan(it->second);
        }
    }
    ctx->plan_cache.clear();
    ctx->stream = nullptr;
}

bool PrepareBitcompGPUPlans(
    const PreparedPointGPU& gpu,
    cudaStream_t stream,
    BitcompGPUDecompressContext* ctx,
    std::string* error) {
    ctx->stream = stream;
    std::set<size_t> unique_inputs;
    for (size_t i = 0; i < gpu.bitcomp_chunks.size(); ++i) {
        unique_inputs.insert(gpu.bitcomp_chunks[i].input_bytes);
    }

    for (std::set<size_t>::const_iterator it = unique_inputs.begin();
         it != unique_inputs.end(); ++it) {
        const size_t input_bytes = *it;
        bitcompHandle_t handle = nullptr;
        bitcompResult_t st = bitcompCreatePlan(
            &handle,
            input_bytes,
            BITCOMP_UNSIGNED_16BIT,
            BITCOMP_LOSSLESS,
            BITCOMP_DEFAULT_ALGO);
        if (!BitcompStatusOk(st, "bitcompCreatePlan", error)) {
            FreeBitcompGPUDecompressContext(ctx);
            return false;
        }
        st = bitcompSetStream(handle, stream);
        if (!BitcompStatusOk(st, "bitcompSetStream", error)) {
            if (handle != nullptr) bitcompDestroyPlan(handle);
            FreeBitcompGPUDecompressContext(ctx);
            return false;
        }
        ctx->plan_cache[input_bytes] = handle;
    }
    return true;
}

bool LaunchZipServGpuOnce(
    const PreparedPointGPU& gpu,
    cudaStream_t stream,
    int* processed_files,
    std::string* error) {
    int processed = 0;
    for (size_t fi = 0; fi < gpu.file_ranges.size(); ++fi) {
        const FileChunkRange& range = gpu.file_ranges[fi];
        bool file_ok = true;
        for (size_t ci = range.begin; ci < range.end; ++ci) {
            const ZipServDeviceChunk& chunk = gpu.zipserv_chunks[ci];
            const cudaError_t ce = BF16TripleBitmap_Decompress_API(
                stream,
                chunk.d_sign_mantissa,
                chunk.d_compressed_full,
                chunk.d_bitmap1,
                chunk.d_bitmap2,
                chunk.d_bitmap3,
                chunk.d_tile_offsets_median,
                chunk.d_tile_offsets_global,
                chunk.max_high_freq_count,
                chunk.max_full_count,
                chunk.start_exp,
                gpu.d_output,
                chunk.rows,
                chunk.cols);
            if (ce != cudaSuccess) {
                *error = std::string("BF16TripleBitmap_Decompress_API launch failed: ") +
                         cudaGetErrorString(ce);
                file_ok = false;
                break;
            }
        }
        if (file_ok) {
            processed += 1;
        } else {
            return false;
        }
    }
    *processed_files = processed;
    return true;
}

bool LaunchBitcompGpuOnce(
    const PreparedPointGPU& gpu,
    const BitcompGPUDecompressContext& ctx,
    int* processed_files,
    std::string* error) {
    int processed = 0;
    for (size_t fi = 0; fi < gpu.file_ranges.size(); ++fi) {
        const FileChunkRange& range = gpu.file_ranges[fi];
        bool file_ok = true;
        for (size_t ci = range.begin; ci < range.end; ++ci) {
            const BitcompDeviceChunk& chunk = gpu.bitcomp_chunks[ci];
            std::map<size_t, bitcompHandle_t>::const_iterator plan_it =
                ctx.plan_cache.find(chunk.input_bytes);
            if (plan_it == ctx.plan_cache.end() || plan_it->second == nullptr) {
                *error = "bitcomp GPU plan lookup failed";
                return false;
            }
            bitcompResult_t st = bitcompUncompress(plan_it->second, chunk.d_compressed, gpu.d_output);
            if (!BitcompStatusOk(st, "bitcompUncompress", error)) {
                file_ok = false;
                break;
            }
        }
        if (file_ok) {
            processed += 1;
        } else {
            return false;
        }
    }
    *processed_files = processed;
    return true;
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

    if (opts.mode == "gpu") {
        const cudaError_t ce = cudaSetDevice(opts.device);
        if (ce != cudaSuccess) {
            std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(ce) << "\n";
            return 1;
        }
    }

    const std::set<std::string> ext_filter = ParseExtensionFilter(opts.ext_filter);
    std::vector<std::string> file_paths;
    std::string error;
    if (!CollectInputFiles(opts, ext_filter, &file_paths, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (file_paths.empty()) {
        std::cerr << "No files found in input path after ext filter.\n";
        return 1;
    }

    std::cout << "Loading " << file_paths.size() << " files from " << opts.input_dir << " ...\n";

    std::vector<PreloadedFile> preloaded;
    preloaded.reserve(file_paths.size());
    int global_max_padded_rows = 0;
    int global_max_padded_cols = 0;

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

        pf.num_chunks = 1;
        pf.max_padded_rows = RoundUpTo64(static_cast<int>(t * h));
        pf.max_padded_cols = RoundUpTo64(static_cast<int>(d));
        if (pf.max_padded_rows > global_max_padded_rows) global_max_padded_rows = pf.max_padded_rows;
        if (pf.max_padded_cols > global_max_padded_cols) global_max_padded_cols = pf.max_padded_cols;
        preloaded.push_back(std::move(pf));
    }

    if (preloaded.empty()) {
        std::cerr << "No valid BF16 files loaded.\n";
        return 1;
    }

    const int num_available = static_cast<int>(preloaded.size());
    std::cout << "Loaded " << num_available << " valid BF16 files.\n";
    std::cout << "Global max padded dims: [" << global_max_padded_rows << "," << global_max_padded_cols << "]\n\n";

    std::vector<int> sweep_values;
    if (!opts.token_counts.empty()) {
        if (!ParsePositiveIntCsv(opts.token_counts, &sweep_values, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
    } else {
        sweep_values = BuildDefaultTokenCounts(preloaded);
        if (sweep_values.empty()) {
            std::cerr << "Failed to derive default token counts from loaded files.\n";
            return 1;
        }
    }

    int64_t max_tokens_per_chunk = 0;
    for (size_t i = 0; i < sweep_values.size(); ++i) {
        max_tokens_per_chunk = std::max<int64_t>(
            max_tokens_per_chunk,
            ResolveChunkLenForWorkItem(static_cast<int64_t>(sweep_values[i]), opts.chunk_len));
    }

    int benchmark_max_padded_rows = 0;
    int benchmark_max_padded_cols = 0;
    for (size_t i = 0; i < preloaded.size(); ++i) {
        const int64_t total_tokens = preloaded[i].info.shape[0];
        const int64_t heads = preloaded[i].info.shape[1];
        const int64_t dim = preloaded[i].info.shape[2];
        const int64_t usable_tokens = std::min<int64_t>(total_tokens, max_tokens_per_chunk);
        if (usable_tokens <= 0) continue;

        const int padded_rows = RoundUpTo64(static_cast<int>(usable_tokens * heads));
        const int padded_cols = RoundUpTo64(static_cast<int>(dim));
        if (padded_rows > benchmark_max_padded_rows) benchmark_max_padded_rows = padded_rows;
        if (padded_cols > benchmark_max_padded_cols) benchmark_max_padded_cols = padded_cols;
    }
    if (benchmark_max_padded_rows <= 0 || benchmark_max_padded_cols <= 0) {
        std::cerr << "Failed to derive benchmark padded dimensions.\n";
        return 1;
    }

    std::cout << "===== KV Cache ZipServ Token Decompression Scaling Benchmark =====\n";
    std::cout << "mode=" << opts.mode
              << ", warmup=" << opts.warmup
              << ", iters=" << opts.iters
              << ", chunk_len=" << ((opts.chunk_len <= 0) ? "disabled" : std::to_string(opts.chunk_len))
              << ", device=" << opts.device
              << ", omp_max_threads=" << saved_max_threads << "\n\n";
    std::cout << "Benchmark max padded dims: ["
              << benchmark_max_padded_rows << "," << benchmark_max_padded_cols << "]\n";
    std::cout << "[token mode] compressed chunks are prepared before timing.\n\n";

    std::vector<GenericResult> results;
    int runnable_points = 0;

    if (opts.mode == "cpu") {
        std::cout << std::left
                  << std::setw(12) << "TokenCount"
                  << std::setw(14) << "EligibleFiles"
                  << std::setw(16) << "ProcessedFiles"
                  << std::setw(12) << "ChunkLen"
                  << std::setw(18) << "ZipServ-CPU(ms)"
                  << std::setw(18) << "bitcomp-CPU(ms)"
                  << std::setw(14) << "ZipServ Comp"
                  << std::setw(14) << "bitcomp Comp"
                  << "\n";
        std::cout << std::string(122, '-') << "\n";
    } else {
        std::cout << std::left
                  << std::setw(12) << "TokenCount"
                  << std::setw(14) << "EligibleFiles"
                  << std::setw(16) << "ProcessedFiles"
                  << std::setw(12) << "ChunkLen"
                  << std::setw(22) << "ZipServ-GPU(ms)"
                  << std::setw(22) << "ZipServ-GPU API(ms)"
                  << std::setw(22) << "bitcomp-GPU(ms)"
                  << std::setw(22) << "bitcomp-GPU API(ms)"
                  << std::setw(14) << "ZipServ Comp"
                  << std::setw(14) << "bitcomp Comp"
                  << "\n";
        std::cout << std::string(170, '-') << "\n";
    }

    for (size_t sweep_idx = 0; sweep_idx < sweep_values.size(); ++sweep_idx) {
        const int token_count = sweep_values[sweep_idx];
        std::vector<WorkItem> work_items;
        int eligible_files = 0;
        std::string work_error;
        if (!BuildWorkItems(
                token_count,
                preloaded,
                num_available,
                opts.require_full_token_count != 0,
                &work_items,
                &eligible_files,
                &work_error)) {
            std::cerr << "[warning] skipping token_count=" << token_count << ": " << work_error << "\n";
            continue;
        }

        int effective_chunk_len = 0;
        for (size_t wi = 0; wi < work_items.size(); ++wi) {
            const int64_t resolved =
                ResolveChunkLenForWorkItem(work_items[wi].seq_count, opts.chunk_len);
            effective_chunk_len = std::max(effective_chunk_len, static_cast<int>(resolved));
        }

        PreparedPoint prepared;
        std::string prep_error;
        if (!PrepareCompressedPoint(
                token_count,
                work_items,
                preloaded,
                opts.chunk_len,
                effective_chunk_len,
                benchmark_max_padded_rows,
                benchmark_max_padded_cols,
                &prepared,
                &prep_error)) {
            std::cerr << "[warning] skipping token_count=" << token_count
                      << " due to preparation failure: " << prep_error << "\n";
            continue;
        }
        if (prepared.processed_files <= 0) {
            std::cerr << "[warning] skipping token_count=" << token_count
                      << ": no processed files after preparation\n";
            continue;
        }
        ++runnable_points;

        double latency_ms[2] = {-1.0, -1.0};
        double core_latency_ms[2] = {0.0, 0.0};
        int method_processed_files[2] = {0, 0};
        double method_ratio[2] = {prepared.zipserv_ratio, prepared.bitcomp_ratio};

        if (opts.mode == "cpu") {
            std::vector<__nv_bfloat16> zipserv_output(prepared.max_output_numel);
            {
                bool ok = true;
                std::string run_error;
                for (int w = 0; w < opts.warmup; ++w) {
                    int processed_iter = 0;
                    if (!RunZipServCpuOnce(
                            prepared,
                            zipserv_output.data(),
                            zipserv_output.size(),
                            &processed_iter,
                            &run_error)) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    double total_ms = 0.0;
                    int valid_iters = 0;
                    for (int iter = 0; iter < opts.iters; ++iter) {
                        int processed_iter = 0;
                        const auto wall_begin = std::chrono::high_resolution_clock::now();
                        if (!RunZipServCpuOnce(
                                prepared,
                                zipserv_output.data(),
                                zipserv_output.size(),
                                &processed_iter,
                                &run_error)) {
                            ok = false;
                            break;
                        }
                        const auto wall_end = std::chrono::high_resolution_clock::now();
                        if (processed_iter <= 0) continue;
                        const double iter_ms =
                            std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
                        total_ms += iter_ms / static_cast<double>(processed_iter);
                        valid_iters += 1;
                        method_processed_files[CPU_DECOMP_ZIPSERV] = processed_iter;
                    }
                    if (ok && valid_iters > 0) {
                        latency_ms[CPU_DECOMP_ZIPSERV] =
                            total_ms / static_cast<double>(valid_iters);
                    } else {
                        ok = false;
                    }
                }
                if (!ok) {
                    std::cerr << "[warning] ZipServ-CPU failed at token_count="
                              << token_count << ": " << run_error << "\n";
                    latency_ms[CPU_DECOMP_ZIPSERV] = -1.0;
                    method_processed_files[CPU_DECOMP_ZIPSERV] = 0;
                }
            }

            BitcompHostContext bitcomp_ctx;
            bool bitcomp_ctx_ok = AllocBitcompHostContext(
                std::max<size_t>(prepared.max_input_bytes, 1),
                &bitcomp_ctx,
                &error);
            if (!bitcomp_ctx_ok) {
                std::cerr << "[warning] bitcomp host context alloc failed at token_count="
                          << token_count << ": " << error << "\n";
                latency_ms[CPU_DECOMP_BITCOMP] = -1.0;
                method_processed_files[CPU_DECOMP_BITCOMP] = 0;
            } else {
                std::vector<uint8_t> bitcomp_output(std::max<size_t>(prepared.max_input_bytes, 1));
                bool ok = PrepareBitcompHostPlans(prepared, &bitcomp_ctx, &error);
                if (!ok) {
                    std::cerr << "[warning] bitcomp host plan prepare failed at token_count="
                              << token_count << ": " << error << "\n";
                    latency_ms[CPU_DECOMP_BITCOMP] = -1.0;
                    method_processed_files[CPU_DECOMP_BITCOMP] = 0;
                } else {
                    std::string run_error;
                    for (int w = 0; w < opts.warmup; ++w) {
                        int processed_iter = 0;
                        if (!RunBitcompCpuOnce(
                                prepared,
                                &bitcomp_ctx,
                                bitcomp_output.data(),
                                bitcomp_output.size(),
                                &processed_iter,
                                &run_error)) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        double total_ms = 0.0;
                        int valid_iters = 0;
                        for (int iter = 0; iter < opts.iters; ++iter) {
                            int processed_iter = 0;
                            const auto wall_begin = std::chrono::high_resolution_clock::now();
                            if (!RunBitcompCpuOnce(
                                    prepared,
                                    &bitcomp_ctx,
                                    bitcomp_output.data(),
                                    bitcomp_output.size(),
                                    &processed_iter,
                                    &run_error)) {
                                ok = false;
                                break;
                            }
                            const auto wall_end = std::chrono::high_resolution_clock::now();
                            if (processed_iter <= 0) continue;
                            const double iter_ms =
                                std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
                            total_ms += iter_ms / static_cast<double>(processed_iter);
                            valid_iters += 1;
                            method_processed_files[CPU_DECOMP_BITCOMP] = processed_iter;
                        }
                        if (ok && valid_iters > 0) {
                            latency_ms[CPU_DECOMP_BITCOMP] =
                                total_ms / static_cast<double>(valid_iters);
                        } else {
                            ok = false;
                        }
                    }
                    if (!ok) {
                        std::cerr << "[warning] bitcomp-CPU failed at token_count="
                                  << token_count << ": " << run_error << "\n";
                        latency_ms[CPU_DECOMP_BITCOMP] = -1.0;
                        method_processed_files[CPU_DECOMP_BITCOMP] = 0;
                    }
                }
                FreeBitcompHostContext(&bitcomp_ctx);
            }
        } else {
            PreparedPointGPU gpu;
            std::string gpu_error;
            if (!PreparePointForGPU(prepared, &gpu, &gpu_error)) {
                std::cerr << "[warning] skipping token_count=" << token_count
                          << " due to GPU prep failure: " << gpu_error << "\n";
                continue;
            }

            cudaStream_t stream = nullptr;
            if (cudaStreamCreate(&stream) != cudaSuccess) {
                std::cerr << "[warning] cudaStreamCreate failed at token_count="
                          << token_count << "\n";
                FreePreparedPointGPU(&gpu);
                continue;
            }

            BitcompGPUDecompressContext bitcomp_gpu_ctx;
            if (!PrepareBitcompGPUPlans(gpu, stream, &bitcomp_gpu_ctx, &gpu_error)) {
                std::cerr << "[warning] bitcomp GPU plan prepare failed at token_count="
                          << token_count << ": " << gpu_error << "\n";
                FreeBitcompGPUDecompressContext(&bitcomp_gpu_ctx);
                cudaStreamDestroy(stream);
                FreePreparedPointGPU(&gpu);
                continue;
            }

            for (int m = 0; m < GPU_DECOMP_COUNT; ++m) {
                bool ok = true;
                std::string run_error;

                for (int w = 0; w < opts.warmup; ++w) {
                    int processed_iter = 0;
                    if (m == GPU_DECOMP_ZIPSERV) {
                        if (!LaunchZipServGpuOnce(gpu, stream, &processed_iter, &run_error)) {
                            ok = false;
                            break;
                        }
                    } else {
                        if (!LaunchBitcompGpuOnce(gpu, bitcomp_gpu_ctx, &processed_iter, &run_error)) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok && cudaStreamSynchronize(stream) != cudaSuccess) {
                        run_error = "cudaStreamSynchronize warmup failed";
                        ok = false;
                        break;
                    }
                }

                cudaEvent_t ev_start = nullptr;
                cudaEvent_t ev_stop = nullptr;
                if (ok) {
                    if (cudaEventCreate(&ev_start) != cudaSuccess ||
                        cudaEventCreate(&ev_stop) != cudaSuccess) {
                        run_error = "cudaEventCreate failed";
                        ok = false;
                    }
                }

                if (ok) {
                    double total_ms = 0.0;
                    double total_core_ms = 0.0;
                    int valid_iters = 0;
                    for (int iter = 0; iter < opts.iters; ++iter) {
                        int processed_iter = 0;
                        const auto wall_begin = std::chrono::high_resolution_clock::now();
                        if (cudaEventRecord(ev_start, stream) != cudaSuccess) {
                            run_error = "cudaEventRecord(start) failed";
                            ok = false;
                            break;
                        }

                        if (m == GPU_DECOMP_ZIPSERV) {
                            if (!LaunchZipServGpuOnce(gpu, stream, &processed_iter, &run_error)) {
                                ok = false;
                                break;
                            }
                        } else {
                            if (!LaunchBitcompGpuOnce(gpu, bitcomp_gpu_ctx, &processed_iter, &run_error)) {
                                ok = false;
                                break;
                            }
                        }

                        if (cudaEventRecord(ev_stop, stream) != cudaSuccess) {
                            run_error = "cudaEventRecord(stop) failed";
                            ok = false;
                            break;
                        }
                        if (cudaEventSynchronize(ev_stop) != cudaSuccess) {
                            run_error = "cudaEventSynchronize(stop) failed";
                            ok = false;
                            break;
                        }

                        float iter_core_ms = 0.0f;
                        if (cudaEventElapsedTime(&iter_core_ms, ev_start, ev_stop) != cudaSuccess) {
                            run_error = "cudaEventElapsedTime failed";
                            ok = false;
                            break;
                        }

                        const auto wall_end = std::chrono::high_resolution_clock::now();
                        if (processed_iter <= 0) continue;

                        const double iter_wall_ms =
                            std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
                        total_ms += iter_wall_ms / static_cast<double>(processed_iter);
                        total_core_ms += static_cast<double>(iter_core_ms) /
                                         static_cast<double>(processed_iter);
                        valid_iters += 1;
                        method_processed_files[m] = processed_iter;
                    }
                    if (ok && valid_iters > 0) {
                        latency_ms[m] = total_ms / static_cast<double>(valid_iters);
                        core_latency_ms[m] = total_core_ms / static_cast<double>(valid_iters);
                    } else {
                        ok = false;
                    }
                }

                if (ev_start != nullptr) cudaEventDestroy(ev_start);
                if (ev_stop != nullptr) cudaEventDestroy(ev_stop);

                if (!ok) {
                    std::cerr << "[warning] " << kGPUDecompMethodNames[m]
                              << " failed at token_count=" << token_count
                              << ": " << run_error << "\n";
                    latency_ms[m] = -1.0;
                    core_latency_ms[m] = 0.0;
                    method_processed_files[m] = 0;
                }
            }

            FreeBitcompGPUDecompressContext(&bitcomp_gpu_ctx);
            cudaStreamDestroy(stream);
            FreePreparedPointGPU(&gpu);
        }

        int processed_ref = -1;
        bool processed_mismatch = false;
        const int method_count = (opts.mode == "cpu")
            ? static_cast<int>(CPU_DECOMP_COUNT)
            : static_cast<int>(GPU_DECOMP_COUNT);
        for (int m = 0; m < method_count; ++m) {
            if (latency_ms[m] < 0.0) continue;
            if (processed_ref < 0) {
                processed_ref = method_processed_files[m];
            } else if (processed_ref != method_processed_files[m]) {
                processed_mismatch = true;
            }
        }
        std::string processed_display =
            (processed_ref < 0) ? "-" : std::to_string(processed_ref);
        if (processed_mismatch) processed_display += "*";

        if (opts.mode == "cpu") {
            std::cout << std::left
                      << std::setw(12) << token_count
                      << std::setw(14) << eligible_files
                      << std::setw(16) << processed_display
                      << std::setw(12) << effective_chunk_len
                      << std::setw(18) << ((latency_ms[CPU_DECOMP_ZIPSERV] < 0.0)
                                               ? "FAILED"
                                               : FormatDouble(latency_ms[CPU_DECOMP_ZIPSERV]))
                      << std::setw(18) << ((latency_ms[CPU_DECOMP_BITCOMP] < 0.0)
                                               ? "FAILED"
                                               : FormatDouble(latency_ms[CPU_DECOMP_BITCOMP]))
                      << std::setw(14) << (FormatDouble(prepared.zipserv_ratio, 2) + "x")
                      << std::setw(14) << (FormatDouble(prepared.bitcomp_ratio, 2) + "x")
                      << "\n";
        } else {
            std::cout << std::left
                      << std::setw(12) << token_count
                      << std::setw(14) << eligible_files
                      << std::setw(16) << processed_display
                      << std::setw(12) << effective_chunk_len
                      << std::setw(22) << ((latency_ms[GPU_DECOMP_ZIPSERV] < 0.0)
                                               ? "FAILED"
                                               : FormatDouble(latency_ms[GPU_DECOMP_ZIPSERV]))
                      << std::setw(22) << ((latency_ms[GPU_DECOMP_ZIPSERV] < 0.0)
                                               ? "-"
                                               : FormatDouble(core_latency_ms[GPU_DECOMP_ZIPSERV]))
                      << std::setw(22) << ((latency_ms[GPU_DECOMP_BITCOMP] < 0.0)
                                               ? "FAILED"
                                               : FormatDouble(latency_ms[GPU_DECOMP_BITCOMP]))
                      << std::setw(22) << ((latency_ms[GPU_DECOMP_BITCOMP] < 0.0)
                                               ? "-"
                                               : FormatDouble(core_latency_ms[GPU_DECOMP_BITCOMP]))
                      << std::setw(14) << (FormatDouble(prepared.zipserv_ratio, 2) + "x")
                      << std::setw(14) << (FormatDouble(prepared.bitcomp_ratio, 2) + "x")
                      << "\n";
        }

        for (int m = 0; m < method_count; ++m) {
            GenericResult gr;
            gr.token_count = token_count;
            gr.eligible_files = eligible_files;
            gr.processed_files = method_processed_files[m];
            gr.requested_chunk_len = opts.chunk_len;
            gr.effective_chunk_len = effective_chunk_len;
            gr.method_name = (opts.mode == "cpu")
                ? kCPUDecompMethodNames[m]
                : kGPUDecompMethodNames[m];
            gr.latency_ms = latency_ms[m];
            gr.core_latency_ms = core_latency_ms[m];
            gr.comp_ratio = method_ratio[m];
            results.push_back(gr);
        }
    }

    if (opts.mode == "cpu") {
        std::cout << std::string(122, '-') << "\n";
    } else {
        std::cout << std::string(170, '-') << "\n";
    }

    std::ofstream csv(opts.out_csv.c_str(), std::ios::out | std::ios::trunc);
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV output: " << opts.out_csv << "\n";
        return 1;
    }
    if (runnable_points == 0) {
        std::cerr << "No runnable sweep points produced any benchmark results.\n";
        return 1;
    }

    csv << "token_count,eligible_files,processed_files,requested_chunk_len,effective_chunk_len,mode,method,latency_ms,core_latency_ms,comp_ratio\n";
    for (size_t i = 0; i < results.size(); ++i) {
        csv << results[i].token_count << ","
            << results[i].eligible_files << ","
            << results[i].processed_files << ","
            << results[i].requested_chunk_len << ","
            << results[i].effective_chunk_len << ","
            << opts.mode << ","
            << results[i].method_name << ","
            << FormatDouble(results[i].latency_ms) << ","
            << FormatDouble(results[i].core_latency_ms) << ","
            << FormatDouble(results[i].comp_ratio, 2) << "\n";
    }
    csv.close();
    std::cout << "CSV written to: " << opts.out_csv << "\n";
    return 0;
}
