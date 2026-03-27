#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_ROOT = Path("/home/pjw7200/models/llama31_70b")
DEFAULT_KV_DIR = Path("/home/pjw7200/saved_kv_cache")
DEFAULT_OUT_CSV = SCRIPT_DIR / "zipserv_decode_attention_results.csv"
EXT_NAME = "zipserv_decode_attention_ext"
FLASH_ATTN_EXT_NAME = "flash_attn_2_cuda"
PYTHON_ABI_TAG = sys.implementation.cache_tag or f"py{sys.version_info.major}{sys.version_info.minor}"
PREBUILT_EXT_ROOT = SCRIPT_DIR / ".prebuilt_extensions" / PYTHON_ABI_TAG
ZIPSERV_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / EXT_NAME
FLASH_ATTN_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / FLASH_ATTN_EXT_NAME
FLASH_ATTN_REGULAR_ONLY_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / f"{FLASH_ATTN_EXT_NAME}_regular_only"
DEFAULT_KV_LEN = 1020
FLASHINFER_PAGE_BLOCK_SIZE = 16
FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024
FUSED_LAYOUT_CHOICES = ("dense", "paged", "by_kv_head")
FLASH_ATTN_MODE = "flashattn"
FLASHINFER_MODE = "flashinfer"
STAGED_FLASH_ATTN_MODE = "staged_flashattn"
STAGED_FLASHINFER_MODE = "staged_flashinfer"
FUSED_FLASH_ATTN_MODE = "fused_flashattn"
SUPPORTED_MODES = {
    FLASH_ATTN_MODE,
    FLASHINFER_MODE,
    STAGED_FLASH_ATTN_MODE,
    STAGED_FLASHINFER_MODE,
    FUSED_FLASH_ATTN_MODE,
}

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")


@dataclass
class ZipservCompressed:
    sign_mantissa: torch.Tensor
    compressed_full: torch.Tensor
    bitmap1: torch.Tensor
    bitmap2: torch.Tensor
    bitmap3: torch.Tensor
    tile_offsets_median: torch.Tensor
    tile_offsets_global: torch.Tensor
    rows: int
    cols: int
    logical_rows: int
    logical_cols: int
    max_high_freq_count: int
    max_full_count: int
    start_exp: int
    total_high_freq: int
    total_full: int
    num_global_tiles: int
    compressed_bytes: int


@dataclass
class KVPair:
    k: torch.Tensor
    v: torch.Tensor
    k_name: str
    v_name: str


@dataclass
class BenchmarkContext:
    ext: object
    device: torch.device
    q_weight: torch.Tensor
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    kv_pairs: List[KVPair]
    kv_pair_offset: int
    fused_regular_only: bool
    fused_layout: str


def parse_csv_ints(spec: str) -> List[int]:
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("token list is empty")
    return sorted(set(values))


def print_terminal_row(row: Dict[str, object], header_state: Dict[str, bool]) -> None:
    fields = [
        ("mode", "mode", 23, "<"),
        ("batch_size", "batch", 7, ">"),
        ("kv_len", "kv_len", 8, ">"),
        ("q_heads", "q_h", 5, ">"),
        ("kv_heads", "kv_h", 5, ">"),
        ("head_dim", "dim", 5, ">"),
        ("latency_ms", "lat_ms", 10, ">"),
        ("base_path", "base", 8, ">"),
    ]
    if not header_state["printed"]:
        print(" ".join(f"{label:{align}{width}}" for _, label, width, align in fields))
        header_state["printed"] = True
    values = []
    for field, _, width, align in fields:
        value = row[field]
        if isinstance(value, float):
            if field in {"latency_ms", "base_path"}:
                values.append(f"{value:{align}{width}.3f}")
                continue
            values.append(f"{value:{align}{width}.6f}")
        else:
            values.append(f"{str(value):{align}{width}}")
    print(" ".join(values))


def baseline_ratio(row_latency_ms: float, baseline_latency_ms: float | None) -> float:
    if baseline_latency_ms is None:
        return float("nan")
    if row_latency_ms <= 0.0:
        return float("inf")
    return baseline_latency_ms / row_latency_ms


def normalize_modes(spec: str) -> List[str]:
    normalized: List[str] = []
    for token in spec.split(","):
        mode = token.strip()
        if not mode:
            continue
        if mode not in normalized:
            normalized.append(mode)
    return normalized


def zipserv_descriptor(comp: ZipservCompressed, layout: str = "by_kv_head") -> Dict[str, object]:
    return {
        "sign_mantissa": comp.sign_mantissa,
        "compressed_full": comp.compressed_full,
        "bitmap1": comp.bitmap1,
        "bitmap2": comp.bitmap2,
        "bitmap3": comp.bitmap3,
        "tile_offsets_median": comp.tile_offsets_median,
        "tile_offsets_global": comp.tile_offsets_global,
        "rows": comp.rows,
        "cols": comp.cols,
        "max_high_freq_count": comp.max_high_freq_count,
        "max_full_count": comp.max_full_count,
        "start_exp": comp.start_exp,
        "layout": layout,
    }


def load_prebuilt_extension(
    module_name: str,
    build_dir: Path,
    build_hint: str,
    *,
    reload_module: bool = False,
) -> object:
    def import_module_from_path(so_path: Path) -> object:
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, so_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load extension spec from {so_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    candidates = sorted(build_dir.glob(f"{module_name}*.so"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(
            f"Prebuilt extension '{module_name}' not found under {build_dir}. "
            f"Build it first with:\n{build_hint}"
        )
    so_path = candidates[0]
    if reload_module:
        sys.modules.pop(module_name, None)
    if module_name in sys.modules:
        return sys.modules[module_name]
    try:
        return import_module_from_path(so_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load prebuilt extension '{module_name}' from {so_path}. "
            f"Rebuild it first with:\n{build_hint}\n"
            f"Original load error: {exc}"
        ) from exc


def load_zipserv_extension() -> object:
    build_hint = (
        "python build_zipserv_decode_attention_extensions.py --target zipserv"
    )
    return load_prebuilt_extension(EXT_NAME, ZIPSERV_EXT_BUILD_DIR, build_hint)


def load_integrated_flash_attn_extension(regular_only: bool = True) -> object:
    build_hint = "python build_zipserv_decode_attention_extensions.py --target zipserv_flashattn"
    build_dir = FLASH_ATTN_REGULAR_ONLY_EXT_BUILD_DIR
    if not regular_only:
        build_hint = "python build_zipserv_decode_attention_extensions.py --target zipserv_flashattn_splitkv"
        build_dir = FLASH_ATTN_EXT_BUILD_DIR
    return load_prebuilt_extension(
        FLASH_ATTN_EXT_NAME,
        build_dir,
        build_hint,
        reload_module=True,
    )


def clear_flash_attn_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "flash_attn" or module_name.startswith("flash_attn."):
            sys.modules.pop(module_name, None)


def import_flash_attn_with_kvcache(prepend_path: Path | None = None) -> Callable[..., torch.Tensor]:
    added_path = False
    if prepend_path is not None:
        path_str = str(prepend_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            added_path = True
    clear_flash_attn_modules()
    try:
        from flash_attn import flash_attn_with_kvcache

        return flash_attn_with_kvcache
    finally:
        if prepend_path is not None and added_path:
            sys.path.remove(str(prepend_path))


def load_model_config(model_root: Path) -> Dict[str, object]:
    with (model_root / "hf_snapshot" / "config.json").open() as f:
        return json.load(f)


def load_manifest_entry(model_root: Path, layer: int) -> Dict[str, object]:
    manifest_path = model_root / "manifest" / "weights_manifest.jsonl"
    needle = f"model.layers.{layer}.self_attn.q_proj.weight"
    with manifest_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["name"] == needle:
                return row
    raise FileNotFoundError(f"Missing q_proj manifest entry for layer {layer}")


def load_bf16_bin(path: Path, shape: Sequence[int]) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16)
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(f"{path} expected {expected} elements, got {raw.size}")
    tensor = torch.from_numpy(raw.reshape(shape)).view(torch.bfloat16)
    return tensor


def load_kv_pairs(kv_dir: Path) -> List[KVPair]:
    files = sorted(p for p in kv_dir.iterdir() if p.is_file())
    if len(files) < 2:
        raise FileNotFoundError(f"Expected at least one K/V pair under {kv_dir}, found {len(files)} files")
    if len(files) % 2 != 0:
        raise ValueError(f"Expected an even number of KV files under {kv_dir}, found {len(files)}")
    pairs: List[KVPair] = []
    for pair_idx in range(0, len(files), 2):
        k_path = files[pair_idx]
        v_path = files[pair_idx + 1]
        k_np = np.load(k_path)
        v_np = np.load(v_path)
        if k_np.shape != v_np.shape:
            raise ValueError(f"Mismatched KV shapes for {k_path.name} and {v_path.name}: {k_np.shape} vs {v_np.shape}")
        k = torch.from_numpy(k_np.view(np.uint16)).view(torch.bfloat16)
        v = torch.from_numpy(v_np.view(np.uint16)).view(torch.bfloat16)
        pairs.append(KVPair(k=k, v=v, k_name=k_path.name, v_name=v_path.name))
    return pairs


def select_kv_pairs(kv_pairs: Sequence[KVPair], batch_size: int, start_pair_idx: int = 0) -> List[KVPair]:
    if not kv_pairs:
        raise ValueError("kv_pairs must not be empty")
    return [kv_pairs[(start_pair_idx + idx) % len(kv_pairs)] for idx in range(batch_size)]


def build_batched_dense_kv(
    kv_pairs: Sequence[KVPair],
    kv_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = torch.stack([pair.k[:kv_len].contiguous() for pair in kv_pairs], dim=0)
    v = torch.stack([pair.v[:kv_len].contiguous() for pair in kv_pairs], dim=0)
    return (
        k.to(device=device, dtype=torch.bfloat16).contiguous(),
        v.to(device=device, dtype=torch.bfloat16).contiguous(),
    )


def build_q(
    weight: torch.Tensor,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    hidden = torch.randn((batch_size, hidden_size), dtype=torch.float32, generator=g).to(dtype=torch.bfloat16, device=device)
    weight = weight.to(device=device, dtype=torch.bfloat16)
    q = hidden @ weight.t()
    return q.view(batch_size, num_heads, head_dim).contiguous()


def pad_batched_kv_prefix(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    batch_size, kv_len, kv_heads, logical_cols = x.shape
    logical_rows_per_batch = kv_len * kv_heads
    rows_per_batch = ((logical_rows_per_batch + 63) // 64) * 64
    cols = ((logical_cols + 63) // 64) * 64
    out = torch.zeros((batch_size, rows_per_batch, cols), dtype=x.dtype, device=x.device)
    out[:, :logical_rows_per_batch, :logical_cols] = x.contiguous().view(batch_size, logical_rows_per_batch, logical_cols)
    return out.view(batch_size * rows_per_batch, cols), logical_rows_per_batch * batch_size, rows_per_batch


def pad_batched_kv_prefix_by_kv_head(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    batch_size, kv_len, kv_heads, logical_cols = x.shape
    rows_per_head = ((kv_len + 63) // 64) * 64
    cols = ((logical_cols + 63) // 64) * 64
    out = torch.zeros((batch_size, kv_heads, rows_per_head, cols), dtype=x.dtype, device=x.device)
    out[:, :, :kv_len, :logical_cols] = x.permute(0, 2, 1, 3).contiguous()
    batch_stride_rows = kv_heads * rows_per_head
    return out.view(batch_size * batch_stride_rows, cols), batch_size * batch_stride_rows, rows_per_head


def pad_batched_kv_prefix_paged(x: torch.Tensor, page_block_size: int) -> Tuple[torch.Tensor, int, int]:
    batch_size, kv_len, kv_heads, logical_cols = x.shape
    padded_tokens = ((kv_len + page_block_size - 1) // page_block_size) * page_block_size
    rows_per_batch = padded_tokens * kv_heads
    cols = ((logical_cols + 63) // 64) * 64
    out = torch.zeros((batch_size, rows_per_batch, cols), dtype=x.dtype, device=x.device)
    out[:, : kv_len * kv_heads, :logical_cols] = x.contiguous().view(batch_size, kv_len * kv_heads, logical_cols)
    return out.view(batch_size * rows_per_batch, cols), batch_size * kv_len * kv_heads, rows_per_batch


def make_batched_paged_kv_cache(x: torch.Tensor, page_block_size: int) -> Tuple[torch.Tensor, int]:
    batch_size, kv_len, kv_heads, head_dim = x.shape
    padded_tokens = ((kv_len + page_block_size - 1) // page_block_size) * page_block_size
    num_pages_per_seq = padded_tokens // page_block_size
    out = torch.zeros((batch_size, padded_tokens, kv_heads, head_dim), dtype=x.dtype, device=x.device)
    out[:, :kv_len, :, :] = x
    return out.view(batch_size * num_pages_per_seq, page_block_size, kv_heads, head_dim).contiguous(), num_pages_per_seq


def make_zipserv_compressed_from_parts(
    sign_mantissa: torch.Tensor,
    compressed_full: torch.Tensor,
    bitmap1: torch.Tensor,
    bitmap2: torch.Tensor,
    bitmap3: torch.Tensor,
    tile_offsets_median: torch.Tensor,
    tile_offsets_global: torch.Tensor,
    meta: torch.Tensor,
) -> ZipservCompressed:
    meta_list = meta.tolist()
    return ZipservCompressed(
        sign_mantissa=sign_mantissa,
        compressed_full=compressed_full,
        bitmap1=bitmap1,
        bitmap2=bitmap2,
        bitmap3=bitmap3,
        tile_offsets_median=tile_offsets_median,
        tile_offsets_global=tile_offsets_global,
        rows=int(meta_list[0]),
        cols=int(meta_list[1]),
        logical_rows=int(meta_list[2]),
        logical_cols=int(meta_list[3]),
        max_high_freq_count=int(meta_list[4]),
        max_full_count=int(meta_list[5]),
        start_exp=int(meta_list[6]),
        total_high_freq=int(meta_list[7]),
        total_full=int(meta_list[8]),
        num_global_tiles=int(meta_list[9]),
        compressed_bytes=int(meta_list[10]),
    )


def make_zipserv_compressed(ext: object, padded_2d: torch.Tensor, logical_rows: int, logical_cols: int) -> ZipservCompressed:
    return make_zipserv_compressed_from_parts(*ext.compress_zipserv(padded_2d, logical_rows, logical_cols))


def decompress_single_to_2d(
    ext: object,
    comp: ZipservCompressed,
    output_2d: torch.Tensor | None = None,
) -> torch.Tensor:
    if output_2d is None:
        return ext.decompress_zipserv(
            comp.sign_mantissa,
            comp.compressed_full,
            comp.bitmap1,
            comp.bitmap2,
            comp.bitmap3,
            comp.tile_offsets_median,
            comp.tile_offsets_global,
            comp.rows,
            comp.cols,
            comp.max_high_freq_count,
            comp.max_full_count,
            comp.start_exp,
        )
    return ext.decompress_zipserv_into(
        output_2d,
        comp.sign_mantissa,
        comp.compressed_full,
        comp.bitmap1,
        comp.bitmap2,
        comp.bitmap3,
        comp.tile_offsets_median,
        comp.tile_offsets_global,
        comp.rows,
        comp.cols,
        comp.max_high_freq_count,
        comp.max_full_count,
        comp.start_exp,
    )


def decompress_to_batched_kv(
    ext: object,
    comp: ZipservCompressed,
    batch_size_cache: int,
    kv_len: int,
    kv_heads: int,
    head_dim: int,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    if comp.rows % batch_size_cache != 0:
        raise ValueError(f"Compressed rows {comp.rows} must be divisible by batch_size_cache {batch_size_cache}")
    rows_per_batch = comp.rows // batch_size_cache
    logical_rows_per_batch = kv_len * kv_heads
    dense = decompress_single_to_2d(ext, comp, output_2d=workspace).view(batch_size_cache, rows_per_batch, comp.cols)
    return dense[:, :logical_rows_per_batch, :head_dim].contiguous().view(batch_size_cache, kv_len, kv_heads, head_dim)


def decompress_to_paged_kv(
    ext: object,
    comp: ZipservCompressed,
    kv_heads: int,
    head_dim: int,
    page_block_size: int,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    if comp.rows % kv_heads != 0:
        raise ValueError(f"Paged workspace rows must be divisible by kv_heads, got rows={comp.rows}, kv_heads={kv_heads}")
    padded_tokens = comp.rows // kv_heads
    if padded_tokens % page_block_size != 0:
        raise ValueError(
            f"Paged workspace tokens must be divisible by page_block_size, got tokens={padded_tokens}, "
            f"page_block_size={page_block_size}"
        )
    dense = decompress_single_to_2d(ext, comp, output_2d=workspace)
    return dense[: padded_tokens * kv_heads, :head_dim].view(
        padded_tokens // page_block_size,
        page_block_size,
        kv_heads,
        head_dim,
    )


def decompress_to_batched_paged_kv(
    ext: object,
    comp: ZipservCompressed,
    batch_size_cache: int,
    kv_heads: int,
    head_dim: int,
    page_block_size: int,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    if comp.rows % batch_size_cache != 0:
        raise ValueError(f"Compressed rows {comp.rows} must be divisible by batch_size_cache {batch_size_cache}")
    rows_per_batch = comp.rows // batch_size_cache
    if rows_per_batch % kv_heads != 0:
        raise ValueError(
            f"Rows per batch must be divisible by kv_heads, got rows_per_batch={rows_per_batch}, kv_heads={kv_heads}"
        )
    padded_tokens = rows_per_batch // kv_heads
    if padded_tokens % page_block_size != 0:
        raise ValueError(
            f"Padded tokens must be divisible by page_block_size, got tokens={padded_tokens}, "
            f"page_block_size={page_block_size}"
        )
    dense = decompress_single_to_2d(ext, comp, output_2d=workspace).view(batch_size_cache, rows_per_batch, comp.cols)
    return dense[:, : padded_tokens * kv_heads, :head_dim].contiguous().view(
        batch_size_cache * (padded_tokens // page_block_size),
        page_block_size,
        kv_heads,
        head_dim,
    )


def availability_or_raise(modes: Iterable[str]) -> None:
    mode_set = set(modes)
    if mode_set & {FLASHINFER_MODE, STAGED_FLASHINFER_MODE}:
        try:
            import flashinfer.decode  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"flashinfer backend is unavailable: {exc}") from exc
    if mode_set & {FLASH_ATTN_MODE, STAGED_FLASH_ATTN_MODE}:
        load_stock_flash_attn_with_kvcache()


def load_stock_flash_attn_with_kvcache() -> Callable[..., torch.Tensor]:
    vendored_root = SCRIPT_DIR / "third_party" / "flash_attn_v283"
    sys.modules.pop(FLASH_ATTN_EXT_NAME, None)
    vendored_exc: Exception | None = None
    try:
        return import_flash_attn_with_kvcache()
    except Exception as site_exc:
        if vendored_root.is_dir():
            try:
                return import_flash_attn_with_kvcache(vendored_root)
            except Exception as exc:
                vendored_exc = exc
        cause = vendored_exc if vendored_exc is not None else site_exc
        raise RuntimeError(
            "flash_attn staged baseline is unavailable. Install flash-attn in the active environment "
            "or make the vendored flash_attn_v283 package importable."
        ) from cause


def load_integrated_flash_attn_with_kvcache(regular_only: bool = True) -> Callable[..., torch.Tensor]:
    vendored_root = SCRIPT_DIR / "third_party" / "flash_attn_v283"
    if not vendored_root.is_dir():
        raise RuntimeError(
            "ZipServ-integrated flash_attn requires the vendored flash_attn_v283 package to be present."
        )
    try:
        sys.modules.pop(FLASH_ATTN_EXT_NAME, None)
        load_integrated_flash_attn_extension(regular_only=regular_only)
        return import_flash_attn_with_kvcache(vendored_root)
    except Exception as exc:
        raise RuntimeError(
            "ZipServ-integrated flash_attn requires the vendored flash_attn_v283 package to be importable."
        ) from exc


def make_flashinfer_runner(
    q: torch.Tensor,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_block_size: int,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    import flashinfer

    batch_size = q.shape[0]
    num_pages = (kv_len + page_block_size - 1) // page_block_size
    workspace = torch.zeros(FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=True)
    indptr = torch.arange(0, (batch_size + 1) * num_pages, step=num_pages, dtype=torch.int32, device=q.device)
    indices = torch.arange(batch_size * num_pages, dtype=torch.int32, device=q.device)
    last_page_len = torch.full(
        (batch_size,),
        kv_len - (num_pages - 1) * page_block_size,
        dtype=torch.int32,
        device=q.device,
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
        kv_data_type=q.dtype,
    )

    def runner(k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        return wrapper.run(q, (k_cache, v_cache))

    return runner


def make_flash_attn_runner(
    q: torch.Tensor,
    kv_len: int,
    flash_attn_with_kvcache: Callable[..., torch.Tensor],
    force_regular: bool = True,
    kv_heads: int | None = None,
    comp_pair: Tuple[ZipservCompressed, ZipservCompressed] | None = None,
    layout: str = "by_kv_head",
) -> Callable[..., torch.Tensor]:
    batch_size, num_q_heads, head_dim = q.shape
    q_4d = q.view(batch_size, 1, num_q_heads, head_dim).contiguous()
    kwargs = {
        "cache_seqlens": torch.full((batch_size,), kv_len, dtype=torch.int32, device=q.device),
        "num_splits": 1 if force_regular else 0,
    }
    if comp_pair is None:
        if not force_regular:
            kwargs["cache_batch_idx"] = torch.arange(batch_size, dtype=torch.int32, device=q.device)

        def runner(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            out_4d = flash_attn_with_kvcache(q_4d, k, v, **kwargs)
            return out_4d.squeeze(1)

        return runner

    if kv_heads is None:
        raise ValueError("kv_heads is required for ZipServ fused FlashAttention")
    k_cache = torch.empty((batch_size, 1, kv_heads, head_dim), dtype=q.dtype, device=q.device)
    v_cache = torch.empty_like(k_cache)
    kwargs.update(
        zipserv_k=zipserv_descriptor(comp_pair[0], layout=layout),
        zipserv_v=zipserv_descriptor(comp_pair[1], layout=layout),
        zipserv_num_heads_k=kv_heads,
    )

    def runner() -> torch.Tensor:
        out_4d = flash_attn_with_kvcache(q_4d, k_cache, v_cache, **kwargs)
        return out_4d.squeeze(1)

    return runner


def make_flash_attn_stage_runner(q: torch.Tensor, kv_len: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    return make_flash_attn_runner(q, kv_len, load_stock_flash_attn_with_kvcache())


def make_flash_attn_zipserv_runner(
    q: torch.Tensor,
    kv_len: int,
    kv_heads: int,
    comp_pair: Tuple[ZipservCompressed, ZipservCompressed],
    layout: str = "by_kv_head",
    force_regular: bool = True,
) -> Callable[[], torch.Tensor]:
    return make_flash_attn_runner(
        q,
        kv_len,
        load_integrated_flash_attn_with_kvcache(regular_only=force_regular),
        force_regular=force_regular,
        kv_heads=kv_heads,
        comp_pair=comp_pair,
        layout=layout,
    )


def time_cuda(fn, warmup: int, iters: int, nvtx_label: str | None = None):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    out = None
    if nvtx_label is not None:
        torch.cuda.nvtx.range_push(f"warmup::{nvtx_label}")
    for _ in range(warmup):
        out = fn()
    if nvtx_label is not None:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if nvtx_label is not None:
        torch.cuda.nvtx.range_push(f"timed::{nvtx_label}")
    start.record()
    for _ in range(iters):
        out = fn()
    stop.record()
    stop.synchronize()
    if nvtx_label is not None:
        torch.cuda.nvtx.range_pop()
    return out, start.elapsed_time(stop) / iters


BACKEND_SPECS = (
    {
        "trigger_modes": {FLASH_ATTN_MODE, STAGED_FLASH_ATTN_MODE},
        "baseline_mode": FLASH_ATTN_MODE,
        "staged_mode": STAGED_FLASH_ATTN_MODE,
        "runner_kind": FLASH_ATTN_MODE,
        "compressed_key": "dense",
        "compressed_error": "staged_flashattn requires dense compressed K/V artifacts",
    },
    {
        "trigger_modes": {FLASHINFER_MODE, STAGED_FLASHINFER_MODE},
        "baseline_mode": FLASHINFER_MODE,
        "staged_mode": STAGED_FLASHINFER_MODE,
        "runner_kind": FLASHINFER_MODE,
        "compressed_key": "paged",
        "compressed_error": "flashinfer modes require paged K/V cache and compressed artifacts",
        "cache_key": "paged",
        "cache_error": "flashinfer modes require paged K/V cache and compressed artifacts",
        "page_block_size": FLASHINFER_PAGE_BLOCK_SIZE,
    },
)


def require_artifact_pair(
    artifacts: Dict[str, Dict[str, object]],
    kind: str,
    key: str,
    error: str,
) -> Tuple[object, object]:
    pair = artifacts[kind].get(key)
    if pair is None:
        raise RuntimeError(error)
    return pair


def benchmark_case(
    mode: str,
    batch_size: int,
    args: argparse.Namespace,
    context: BenchmarkContext,
    fn: Callable[[], torch.Tensor],
) -> Dict[str, float | str]:
    _, latency_ms = time_cuda(
        fn,
        args.warmup,
        args.iters,
        nvtx_label=(
            f"{mode}|b{batch_size}|kv{args.kv_len}|qh{context.num_q_heads}|"
            f"kvh{context.num_kv_heads}|d{context.head_dim}"
        ),
    )
    return {"latency_ms": float(latency_ms), "status": "ok"}


def make_staged_run_fn(
    layout: str,
    ext: object,
    runner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    comp_pair: Tuple[ZipservCompressed, ZipservCompressed],
    batch_size: int,
    kv_len: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    page_block_size: int | None = None,
) -> Callable[[], torch.Tensor]:
    workspaces = tuple(
        torch.empty((comp.rows, comp.cols), dtype=torch.bfloat16, device=device)
        for comp in comp_pair
    )

    def run() -> torch.Tensor:
        if layout == "dense":
            k = decompress_to_batched_kv(ext, comp_pair[0], batch_size, kv_len, kv_heads, head_dim, workspace=workspaces[0])
            v = decompress_to_batched_kv(ext, comp_pair[1], batch_size, kv_len, kv_heads, head_dim, workspace=workspaces[1])
            return runner(k, v)
        if page_block_size is None:
            raise ValueError("paged staged mode requires page_block_size")
        k = decompress_to_batched_paged_kv(
            ext,
            comp_pair[0],
            batch_size,
            kv_heads,
            head_dim,
            page_block_size,
            workspace=workspaces[0],
        )
        v = decompress_to_batched_paged_kv(
            ext,
            comp_pair[1],
            batch_size,
            kv_heads,
            head_dim,
            page_block_size,
            workspace=workspaces[1],
        )
        return runner(k, v)

    return run


def make_backend_runner(
    spec: Dict[str, object],
    context: BenchmarkContext,
    args: argparse.Namespace,
    batch_inputs: Dict[str, object],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    q = batch_inputs["q"]
    if spec["runner_kind"] == FLASH_ATTN_MODE:
        return make_flash_attn_stage_runner(q, args.kv_len)
    return make_flashinfer_runner(
        q,
        args.kv_len,
        context.num_q_heads,
        context.num_kv_heads,
        context.head_dim,
        spec["page_block_size"],
    )


def run_backend_family(
    rows: List[Dict[str, float | str | int]],
    header_state: Dict[str, bool],
    args: argparse.Namespace,
    mode_set: set[str],
    context: BenchmarkContext,
    batch_inputs: Dict[str, object],
    artifacts: Dict[str, Dict[str, object]],
    spec: Dict[str, object],
) -> None:
    if not (mode_set & spec["trigger_modes"]):
        return

    q = batch_inputs["q"]
    batch_size = int(q.shape[0])
    runner = make_backend_runner(spec, context, args, batch_inputs)
    baseline_pair = (
        batch_inputs["dense"]
        if spec["runner_kind"] == FLASH_ATTN_MODE
        else require_artifact_pair(artifacts, "cache", spec["cache_key"], spec["cache_error"])
    )
    baseline_metrics = benchmark_case(
        spec["baseline_mode"],
        batch_size,
        args,
        context,
        lambda: runner(*baseline_pair),
    )
    baseline_latency_ms = float(baseline_metrics["latency_ms"])

    if spec["baseline_mode"] in mode_set:
        append_result_row(rows, header_state, args, context, spec["baseline_mode"], batch_size, baseline_metrics, None)

    if spec["staged_mode"] in mode_set:
        comp_pair = require_artifact_pair(
            artifacts,
            "compressed",
            spec["compressed_key"],
            spec["compressed_error"],
        )
        staged_metrics = benchmark_case(
            spec["staged_mode"],
            batch_size,
            args,
            context,
            make_staged_run_fn(
                spec["compressed_key"],
                context.ext,
                runner,
                comp_pair,
                batch_size,
                args.kv_len,
                context.num_kv_heads,
                context.head_dim,
                q.device,
                page_block_size=spec.get("page_block_size"),
            ),
        )
        append_result_row(
            rows,
            header_state,
            args,
            context,
            spec["staged_mode"],
            batch_size,
            staged_metrics,
            baseline_latency_ms,
        )


def run_fused_flash_attn(
    rows: List[Dict[str, float | str | int]],
    header_state: Dict[str, bool],
    args: argparse.Namespace,
    mode_set: set[str],
    context: BenchmarkContext,
    batch_inputs: Dict[str, object],
    artifacts: Dict[str, Dict[str, object]],
) -> None:
    if FUSED_FLASH_ATTN_MODE not in mode_set:
        return
    q = batch_inputs["q"]
    batch_size = int(q.shape[0])
    flash_attn_with_kvcache = load_integrated_flash_attn_with_kvcache(
        regular_only=context.fused_regular_only
    )
    baseline_runner = make_flash_attn_runner(
        q,
        args.kv_len,
        flash_attn_with_kvcache,
        force_regular=context.fused_regular_only,
    )
    baseline_metrics = benchmark_case(
        FLASH_ATTN_MODE,
        batch_size,
        args,
        context,
        lambda: baseline_runner(*batch_inputs["dense"]),
    )
    comp_pair = require_artifact_pair(
        artifacts,
        "compressed",
        context.fused_layout,
        "fused_flashattn requires fused compressed K/V artifacts",
    )
    fused_metrics = benchmark_case(
        FUSED_FLASH_ATTN_MODE,
        batch_size,
        args,
        context,
        make_flash_attn_runner(
            q,
            args.kv_len,
            flash_attn_with_kvcache,
            force_regular=context.fused_regular_only,
            kv_heads=context.num_kv_heads,
            comp_pair=comp_pair,
            layout=context.fused_layout,
        ),
    )
    append_result_row(
        rows,
        header_state,
        args,
        context,
        FUSED_FLASH_ATTN_MODE,
        batch_size,
        fused_metrics,
        float(baseline_metrics["latency_ms"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ZipServ decode-attention benchmark")
    parser.add_argument("--model_root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--kv_dir", type=Path, default=DEFAULT_KV_DIR)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--modes",
        type=str,
        default="flashattn,flashinfer,staged_flashattn,staged_flashinfer,fused_flashattn",
    )
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument(
        "--kv_len",
        type=int,
        default=DEFAULT_KV_LEN,
        help="Logical KV token length to materialize/compress for each request",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument(
        "--fused_layout",
        type=str,
        choices=FUSED_LAYOUT_CHOICES,
        default="by_kv_head",
        help="Compressed ZipServ layout used by fused_flashattn",
    )
    parser.add_argument(
        "--fused_splitkv",
        action="store_true",
        help="Opt back into the legacy split-kv fused FlashAttention path for comparison",
    )
    return parser


def validate_modes(modes: Sequence[str]) -> None:
    for mode in modes:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}")


def load_benchmark_context(args: argparse.Namespace, modes: Sequence[str], device: torch.device) -> BenchmarkContext:
    mode_set = set(modes)
    ext = load_zipserv_extension()
    config = load_model_config(args.model_root)
    q_entry = load_manifest_entry(args.model_root, args.layer)
    q_weight = load_bf16_bin(Path(q_entry["path"]), q_entry["shape"])
    hidden_size = int(config["hidden_size"])
    num_q_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden_size // num_q_heads
    if FUSED_FLASH_ATTN_MODE in mode_set and num_kv_heads != 8:
        raise ValueError(f"{FUSED_FLASH_ATTN_MODE} currently requires num_kv_heads == 8, got {num_kv_heads}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    kv_pairs = load_kv_pairs(args.kv_dir)
    max_tokens = min(pair.k.shape[0] for pair in kv_pairs)
    if args.kv_len <= 0:
        raise ValueError(f"kv_len must be positive, got {args.kv_len}")
    if args.kv_len > max_tokens:
        raise ValueError(f"Requested kv_len {args.kv_len} exceeds available length {max_tokens}")

    return BenchmarkContext(
        ext=ext,
        device=device,
        q_weight=q_weight,
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_pairs=kv_pairs,
        kv_pair_offset=args.layer % len(kv_pairs),
        fused_regular_only=not args.fused_splitkv,
        fused_layout=args.fused_layout,
    )


def prepare_batch_inputs(
    context: BenchmarkContext,
    batch_size: int,
    kv_len: int,
    seed: int,
) -> Dict[str, object]:
    batch_kv_pairs = select_kv_pairs(context.kv_pairs, batch_size, start_pair_idx=context.kv_pair_offset)
    dense_k, dense_v = build_batched_dense_kv(batch_kv_pairs, kv_len, context.device)
    q = build_q(
        context.q_weight,
        context.hidden_size,
        context.num_q_heads,
        context.head_dim,
        seed,
        context.device,
        batch_size,
    )
    return {"q": q, "dense": (dense_k, dense_v)}


def compress_kv_pair_with_layout(
    ext: object,
    kv_pair: Tuple[torch.Tensor, torch.Tensor],
    head_dim: int,
    pad_fn: Callable[..., Tuple[torch.Tensor, int, int]],
    *pad_args: int,
) -> Tuple[ZipservCompressed, ZipservCompressed]:
    padded_k, logical_rows, _ = pad_fn(kv_pair[0], *pad_args)
    padded_v, _, _ = pad_fn(kv_pair[1], *pad_args)
    return (
        make_zipserv_compressed(ext, padded_k, logical_rows, head_dim),
        make_zipserv_compressed(ext, padded_v, logical_rows, head_dim),
    )


def build_kv_cache_pair_with_layout(
    kv_pair: Tuple[torch.Tensor, torch.Tensor],
    cache_builder: Callable[..., Tuple[torch.Tensor, int]],
    *cache_args: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k_cache, _ = cache_builder(kv_pair[0], *cache_args)
    v_cache, _ = cache_builder(kv_pair[1], *cache_args)
    return k_cache, v_cache


ARTIFACT_LAYOUT_SPECS = {
    "dense": {
        "pad": pad_batched_kv_prefix,
        "pad_args": (),
    },
    "paged": {
        "pad": pad_batched_kv_prefix_paged,
        "pad_args": (FLASHINFER_PAGE_BLOCK_SIZE,),
        "cache": make_batched_paged_kv_cache,
        "cache_args": (FLASHINFER_PAGE_BLOCK_SIZE,),
    },
    "by_kv_head": {
        "pad": pad_batched_kv_prefix_by_kv_head,
        "pad_args": (),
    },
}


def prepare_batch_artifacts(
    context: BenchmarkContext,
    modes: Sequence[str],
    batch_inputs: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    mode_set = set(modes)
    dense_pair = batch_inputs["dense"]
    artifacts = {"compressed": {}, "cache": {}}
    required_layouts = set()
    if mode_set & {FLASH_ATTN_MODE, STAGED_FLASH_ATTN_MODE}:
        required_layouts.add("dense")
    if mode_set & {FLASHINFER_MODE, STAGED_FLASHINFER_MODE}:
        required_layouts.add("paged")
    if FUSED_FLASH_ATTN_MODE in mode_set:
        required_layouts.add(context.fused_layout)
    for layout in required_layouts:
        spec = ARTIFACT_LAYOUT_SPECS[layout]
        if layout in artifacts["compressed"]:
            continue
        artifacts["compressed"][layout] = compress_kv_pair_with_layout(
            context.ext,
            dense_pair,
            context.head_dim,
            spec["pad"],
            *spec["pad_args"],
        )
        cache_builder = spec.get("cache")
        if cache_builder is not None:
            artifacts["cache"][layout] = build_kv_cache_pair_with_layout(
                dense_pair,
                cache_builder,
                *spec["cache_args"],
            )
    return artifacts


def append_result_row(
    rows: List[Dict[str, float | str | int]],
    header_state: Dict[str, bool],
    args: argparse.Namespace,
    context: BenchmarkContext,
    mode: str,
    batch_size: int,
    metrics: Dict[str, float | str | int],
    baseline_latency_ms: float | None,
) -> None:
    row = {
        "layer": args.layer,
        "mode": mode,
        "batch_size": batch_size,
        "kv_len": args.kv_len,
        "q_heads": context.num_q_heads,
        "kv_heads": context.num_kv_heads,
        "head_dim": context.head_dim,
        "base_path": baseline_ratio(float(metrics["latency_ms"]), baseline_latency_ms),
        **metrics,
    }
    if baseline_latency_ms is None:
        row["base_path"] = 1.0
    rows.append(row)
    print_terminal_row(row, header_state)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This benchmark is GPU-only.")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    modes = normalize_modes(args.modes)
    mode_set = set(modes)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    validate_modes(modes)
    availability_or_raise(modes)
    context = load_benchmark_context(args, modes, device)

    rows = []
    header_state = {"printed": False}
    for batch_size in batch_sizes:
        batch_inputs = prepare_batch_inputs(context, batch_size, args.kv_len, args.seed)
        artifacts = prepare_batch_artifacts(context, modes, batch_inputs)
        for spec in BACKEND_SPECS:
            run_backend_family(rows, header_state, args, mode_set, context, batch_inputs, artifacts, spec)
        run_fused_flash_attn(rows, header_state, args, mode_set, context, batch_inputs, artifacts)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer",
        "mode",
        "batch_size",
        "kv_len",
        "q_heads",
        "kv_heads",
        "head_dim",
        "latency_ms",
        "base_path",
        "status",
    ]
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print(
        f"layer={args.layer} kv_len={args.kv_len} kv_pair_pool={len(context.kv_pairs)} "
        f"start_pair=({context.kv_pairs[context.kv_pair_offset].k_name}, "
        f"{context.kv_pairs[context.kv_pair_offset].v_name})"
    )


if __name__ == "__main__":
    main()
