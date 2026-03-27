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

void CheckZipservCompressed(
    const torch::Tensor& sign_mantissa,
    const torch::Tensor& compressed_full,
    const torch::Tensor& bitmap1,
    const torch::Tensor& bitmap2,
    const torch::Tensor& bitmap3,
    const torch::Tensor& tile_offsets_median,
    const torch::Tensor& tile_offsets_global);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress_zipserv", &compress_zipserv, "Compress a padded BF16 2D tensor with ZipServ");
    m.def("decompress_zipserv", &decompress_zipserv, "Decompress a ZipServ payload into a dense BF16 tensor");
    m.def("decompress_zipserv_into", &decompress_zipserv_into, "Decompress a ZipServ payload into a provided dense BF16 tensor");
}
