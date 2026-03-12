#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> compress_zipserv_cuda(
    torch::Tensor input,
    int64_t logical_rows,
    int64_t logical_cols);

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
    int64_t start_exp);

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

torch::Tensor zipserv_decode_attention_cuda(
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
    double sm_scale);

namespace {

void CheckCudaTensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void CheckZipservCompressed(
    const torch::Tensor& sign_mantissa,
    const torch::Tensor& compressed_full,
    const torch::Tensor& bitmap1,
    const torch::Tensor& bitmap2,
    const torch::Tensor& bitmap3,
    const torch::Tensor& tile_offsets_median,
    const torch::Tensor& tile_offsets_global) {
    CheckCudaTensor(sign_mantissa, "sign_mantissa");
    CheckCudaTensor(compressed_full, "compressed_full");
    CheckCudaTensor(bitmap1, "bitmap1");
    CheckCudaTensor(bitmap2, "bitmap2");
    CheckCudaTensor(bitmap3, "bitmap3");
    CheckCudaTensor(tile_offsets_median, "tile_offsets_median");
    CheckCudaTensor(tile_offsets_global, "tile_offsets_global");
    TORCH_CHECK(sign_mantissa.dtype() == torch::kUInt8, "sign_mantissa must be uint8");
    TORCH_CHECK(compressed_full.dtype() == torch::kBFloat16, "compressed_full must be bfloat16");
    TORCH_CHECK(bitmap1.dtype() == torch::kUInt64, "bitmap1 must be uint64");
    TORCH_CHECK(bitmap2.dtype() == torch::kUInt64, "bitmap2 must be uint64");
    TORCH_CHECK(bitmap3.dtype() == torch::kUInt64, "bitmap3 must be uint64");
    TORCH_CHECK(tile_offsets_median.dtype() == torch::kInt32, "tile_offsets_median must be int32");
    TORCH_CHECK(tile_offsets_global.dtype() == torch::kInt32, "tile_offsets_global must be int32");
}

std::vector<torch::Tensor> compress_zipserv(
    torch::Tensor input,
    int64_t logical_rows,
    int64_t logical_cols) {
    CheckCudaTensor(input, "input");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    return compress_zipserv_cuda(input, logical_rows, logical_cols);
}

torch::Tensor decompress_zipserv(
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
    CheckCudaTensor(sign_mantissa, "sign_mantissa");
    CheckCudaTensor(compressed_full, "compressed_full");
    CheckCudaTensor(bitmap1, "bitmap1");
    CheckCudaTensor(bitmap2, "bitmap2");
    CheckCudaTensor(bitmap3, "bitmap3");
    CheckCudaTensor(tile_offsets_median, "tile_offsets_median");
    CheckCudaTensor(tile_offsets_global, "tile_offsets_global");
    CheckZipservCompressed(
        sign_mantissa,
        compressed_full,
        bitmap1,
        bitmap2,
        bitmap3,
        tile_offsets_median,
        tile_offsets_global);
    return decompress_zipserv_cuda(
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

torch::Tensor decompress_zipserv_into(
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
    CheckCudaTensor(output, "output");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bfloat16");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(output.size(0) == rows, "output rows must match rows");
    TORCH_CHECK(output.size(1) == cols, "output cols must match cols");
    CheckZipservCompressed(
        sign_mantissa,
        compressed_full,
        bitmap1,
        bitmap2,
        bitmap3,
        tile_offsets_median,
        tile_offsets_global);
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

torch::Tensor zipserv_decode_attention(
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
    CheckCudaTensor(q, "q");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(q.dim() == 2, "q must be 2D");
    TORCH_CHECK(q.size(0) == num_q_heads, "q rows must match num_q_heads");
    TORCH_CHECK(q.size(1) == head_dim, "q cols must match head_dim");
    CheckZipservCompressed(
        k_sign_mantissa,
        k_compressed_full,
        k_bitmap1,
        k_bitmap2,
        k_bitmap3,
        k_tile_offsets_median,
        k_tile_offsets_global);
    CheckZipservCompressed(
        v_sign_mantissa,
        v_compressed_full,
        v_bitmap1,
        v_bitmap2,
        v_bitmap3,
        v_tile_offsets_median,
        v_tile_offsets_global);
    return zipserv_decode_attention_cuda(
        q,
        k_sign_mantissa,
        k_compressed_full,
        k_bitmap1,
        k_bitmap2,
        k_bitmap3,
        k_tile_offsets_median,
        k_tile_offsets_global,
        k_rows,
        k_cols,
        k_max_high_freq_count,
        k_max_full_count,
        k_start_exp,
        v_sign_mantissa,
        v_compressed_full,
        v_bitmap1,
        v_bitmap2,
        v_bitmap3,
        v_tile_offsets_median,
        v_tile_offsets_global,
        v_rows,
        v_cols,
        v_max_high_freq_count,
        v_max_full_count,
        v_start_exp,
        logical_kv_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        sm_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress_zipserv", &compress_zipserv, "Compress a padded BF16 2D tensor with ZipServ");
    m.def("decompress_zipserv", &decompress_zipserv, "Decompress a ZipServ payload into a dense BF16 tensor");
    m.def("decompress_zipserv_into", &decompress_zipserv_into, "Decompress a ZipServ payload into a provided dense BF16 tensor");
    m.def("zipserv_decode_attention", &zipserv_decode_attention, "Run decode attention directly from ZipServ-compressed K/V");
}
