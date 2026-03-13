#include "zipserv_fwd_kvcache.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ZipServ fused FlashAttention decode path";
    m.def(
        "fwd_kvcache_zipserv",
        &FLASH_NAMESPACE::mha_fwd_kvcache_zipserv,
        "Forward pass for ZipServ-compressed KV cache");
}
