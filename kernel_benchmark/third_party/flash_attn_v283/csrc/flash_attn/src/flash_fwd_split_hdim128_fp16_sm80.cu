// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include <c10/util/Exception.h>

#include <cutlass/numeric_types.h>

#include "flash.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    (void)params;
    (void)stream;
    TORCH_CHECK(
        false,
        "flash_attn_2_cuda benchmark build does not support fp16 split-kv decode for head_dim=128");
}

} // namespace FLASH_NAMESPACE
