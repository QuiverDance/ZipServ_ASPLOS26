#pragma once

#include "flash_common.hpp"
#include "namespace_config.h"

#include <vector>

namespace FLASH_NAMESPACE {

std::vector<at::Tensor> mha_fwd_kvcache_zipserv(
    at::Tensor &q,
    const at::Tensor &k_sign_mantissa,
    const at::Tensor &k_compressed_full,
    const at::Tensor &k_bitmap1,
    const at::Tensor &k_bitmap2,
    const at::Tensor &k_bitmap3,
    const at::Tensor &k_tile_offsets_median,
    const at::Tensor &k_tile_offsets_global,
    int64_t k_rows,
    int64_t k_cols,
    int64_t k_max_high_freq_count,
    int64_t k_max_full_count,
    int64_t k_start_exp,
    const at::Tensor &v_sign_mantissa,
    const at::Tensor &v_compressed_full,
    const at::Tensor &v_bitmap1,
    const at::Tensor &v_bitmap2,
    const at::Tensor &v_bitmap3,
    const at::Tensor &v_tile_offsets_median,
    const at::Tensor &v_tile_offsets_global,
    int64_t v_rows,
    int64_t v_cols,
    int64_t v_max_high_freq_count,
    int64_t v_max_full_count,
    int64_t v_start_exp,
    int64_t logical_kv_len,
    int64_t num_kv_heads,
    float softmax_scale);

}  // namespace FLASH_NAMESPACE
