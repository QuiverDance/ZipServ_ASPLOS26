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
FLASH_ATTN_MODE = "flashattn"
FLASHINFER_MODE = "flashinfer"
STAGED_FLASH_ATTN_MODE = "staged_flashattn"
STAGED_FLASHINFER_MODE = "staged_flashinfer"
FUSED_FLASH_ATTN_MODE = "fused_flashattn"

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


def zipserv_descriptor(comp: ZipservCompressed) -> Dict[str, object]:
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
    }


def ensure_extension_built(build_target: str) -> None:
    import build_zipserv_decode_attention_extensions as ext_builder

    ext_builder.ensure_extension_built(build_target)


def load_prebuilt_extension(
    module_name: str,
    build_dir: Path,
    build_hint: str,
    build_target: str,
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
        ensure_extension_built(build_target)
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
    except Exception as first_exc:
        ensure_extension_built(build_target)
        candidates = sorted(build_dir.glob(f"{module_name}*.so"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError(
                f"Prebuilt extension '{module_name}' could not be rebuilt under {build_dir}. "
                f"Build it first with:\n{build_hint}"
            ) from first_exc
        try:
            return import_module_from_path(candidates[0])
        except Exception as second_exc:
            raise RuntimeError(
                f"Failed to load extension '{module_name}' from {candidates[0]}.\n"
                f"First load error: {first_exc}\n"
                f"After rebuild: {second_exc}"
            ) from second_exc


def load_zipserv_extension() -> object:
    build_hint = (
        "python build_zipserv_decode_attention_extensions.py --target zipserv"
    )
    return load_prebuilt_extension(EXT_NAME, ZIPSERV_EXT_BUILD_DIR, build_hint, "zipserv")


def load_integrated_flash_attn_extension(regular_only: bool = True) -> object:
    build_hint = "python build_zipserv_decode_attention_extensions.py --target zipserv_flashattn"
    build_dir = FLASH_ATTN_REGULAR_ONLY_EXT_BUILD_DIR
    build_target = "zipserv_flashattn"
    if not regular_only:
        build_hint = "python build_zipserv_decode_attention_extensions.py --target zipserv_flashattn_splitkv"
        build_dir = FLASH_ATTN_EXT_BUILD_DIR
        build_target = "zipserv_flashattn_splitkv"
    return load_prebuilt_extension(
        FLASH_ATTN_EXT_NAME,
        build_dir,
        build_hint,
        build_target,
        reload_module=True,
    )


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
    if mode_set & {FLASH_ATTN_MODE, STAGED_FLASH_ATTN_MODE} and not (mode_set & {FUSED_FLASH_ATTN_MODE}):
        load_flash_attn_with_kvcache(prefer_loaded_extension=False)


def load_flash_attn_with_kvcache(
    use_integrated_zipserv: bool = False,
    prefer_loaded_extension: bool = True,
    integrated_regular_only: bool = True,
) -> Callable[..., torch.Tensor]:
    vendored_root = SCRIPT_DIR / "third_party" / "flash_attn_v283"

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

    def using_integrated_flash_attn_extension() -> bool:
        module = sys.modules.get(FLASH_ATTN_EXT_NAME)
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            return False
        try:
            resolved_path = Path(module_path).resolve()
            return (
                resolved_path.is_relative_to(FLASH_ATTN_EXT_BUILD_DIR.resolve())
                or resolved_path.is_relative_to(FLASH_ATTN_REGULAR_ONLY_EXT_BUILD_DIR.resolve())
            )
        except AttributeError:
            resolved = str(Path(module_path).resolve())
            return (
                resolved.startswith(str(FLASH_ATTN_EXT_BUILD_DIR.resolve()))
                or resolved.startswith(str(FLASH_ATTN_REGULAR_ONLY_EXT_BUILD_DIR.resolve()))
            )

    vendored_exc: Exception | None = None
    if use_integrated_zipserv:
        if not vendored_root.is_dir():
            raise RuntimeError(
                "ZipServ-integrated flash_attn requires the vendored flash_attn_v283 package to be present."
            )
        try:
            sys.modules.pop(FLASH_ATTN_EXT_NAME, None)
            load_integrated_flash_attn_extension(regular_only=integrated_regular_only)
            return import_flash_attn_with_kvcache(vendored_root)
        except Exception as exc:
            raise RuntimeError(
                "ZipServ-integrated flash_attn requires the vendored flash_attn_v283 package to be importable."
            ) from exc

    if not prefer_loaded_extension:
        sys.modules.pop(FLASH_ATTN_EXT_NAME, None)

    if prefer_loaded_extension and using_integrated_flash_attn_extension():
        if not vendored_root.is_dir():
            raise RuntimeError(
                "Integrated flash_attn_2_cuda is loaded, but the vendored flash_attn_v283 Python wrapper is missing."
            )
        try:
            return import_flash_attn_with_kvcache(vendored_root)
        except Exception as exc:
            vendored_exc = exc

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


def make_flash_attn_stage_runner(
    q: torch.Tensor,
    kv_len: int,
    use_integrated_extension: bool = False,
    force_regular: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    flash_attn_with_kvcache = load_flash_attn_with_kvcache(
        use_integrated_zipserv=use_integrated_extension,
        prefer_loaded_extension=use_integrated_extension,
        integrated_regular_only=use_integrated_extension and force_regular,
    )
    batch_size, num_q_heads, head_dim = q.shape
    q_4d = q.view(batch_size, 1, num_q_heads, head_dim).contiguous()
    cache_seqlens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=q.device)
    cache_batch_idx = None if force_regular else torch.arange(batch_size, dtype=torch.int32, device=q.device)

    def runner(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        kwargs = {"cache_seqlens": cache_seqlens}
        if cache_batch_idx is not None:
            kwargs["cache_batch_idx"] = cache_batch_idx
        if force_regular:
            kwargs["num_splits"] = 1
        out_4d = flash_attn_with_kvcache(q_4d, k, v, **kwargs)
        return out_4d.squeeze(1)

    return runner


def make_flash_attn_zipserv_runner(
    q: torch.Tensor,
    kv_len: int,
    kv_heads: int,
    comp_k: ZipservCompressed,
    comp_v: ZipservCompressed,
    force_regular: bool = True,
) -> Callable[[], torch.Tensor]:
    flash_attn_with_kvcache = load_flash_attn_with_kvcache(
        use_integrated_zipserv=True,
        integrated_regular_only=force_regular,
    )
    batch_size, num_q_heads, head_dim = q.shape
    q_4d = q.view(batch_size, 1, num_q_heads, head_dim).contiguous()
    cache_seqlens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=q.device)
    k_cache = torch.empty((batch_size, 1, kv_heads, head_dim), dtype=q.dtype, device=q.device)
    v_cache = torch.empty_like(k_cache)
    zipserv_k = zipserv_descriptor(comp_k)
    zipserv_v = zipserv_descriptor(comp_v)

    def runner() -> torch.Tensor:
        out_4d = flash_attn_with_kvcache(
            q_4d,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            zipserv_k=zipserv_k,
            zipserv_v=zipserv_v,
            zipserv_num_heads_k=kv_heads,
            num_splits=1 if force_regular else 0,
        )
        return out_4d.squeeze(1)

    return runner


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


def benchmark_one(
    ext: object,
    mode: str,
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    comp_k: ZipservCompressed,
    comp_v: ZipservCompressed,
    kv_len: int,
    num_q_heads: int,
    kv_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    runner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    zipserv_runner: Callable[[], torch.Tensor] | None = None,
    reuse_k_workspace: torch.Tensor | None = None,
    reuse_v_workspace: torch.Tensor | None = None,
    flash_attn_ext: object | None = None,
    page_block_size: int | None = None,
    cache_batch_size: int | None = None,
) -> Tuple[torch.Tensor, Dict[str, float | str | int]]:
    out = None
    batch_size = int(dense_k.shape[0])
    nvtx_label = f"{mode}|b{batch_size}|kv{kv_len}|qh{num_q_heads}|kvh{kv_heads}|d{head_dim}"

    if mode == "direct":
        if runner is None:
            raise ValueError("direct mode requires a prepared attention runner")
        out, latency_ms = time_cuda(lambda: runner(dense_k, dense_v), warmup, iters, nvtx_label=nvtx_label)
    elif mode == "staged_dense":
        if runner is None:
            raise ValueError("staged_dense mode requires a prepared attention runner")
        if reuse_k_workspace is None or reuse_v_workspace is None:
            raise ValueError("staged_dense mode requires reusable K/V output buffers")
        batch_size_cache = cache_batch_size if cache_batch_size is not None else int(dense_k.shape[0])

        def staged_dense():
            k = decompress_to_batched_kv(
                ext,
                comp_k,
                batch_size_cache,
                kv_len,
                kv_heads,
                head_dim,
                workspace=reuse_k_workspace,
            )
            v = decompress_to_batched_kv(
                ext,
                comp_v,
                batch_size_cache,
                kv_len,
                kv_heads,
                head_dim,
                workspace=reuse_v_workspace,
            )
            return runner(k, v)

        out, latency_ms = time_cuda(staged_dense, warmup, iters, nvtx_label=nvtx_label)
    elif mode == "staged_paged":
        if runner is None:
            raise ValueError("staged_paged mode requires a prepared attention runner")
        if reuse_k_workspace is None or reuse_v_workspace is None:
            raise ValueError("staged_paged mode requires reusable dense K/V workspaces")
        if page_block_size is None:
            raise ValueError("staged_paged mode requires page_block_size")
        batch_size_cache = cache_batch_size if cache_batch_size is not None else 0
        if batch_size_cache <= 0:
            raise ValueError("staged_paged mode could not infer batch_size_cache")

        def staged_paged():
            k = decompress_to_batched_paged_kv(
                ext,
                comp_k,
                batch_size_cache,
                kv_heads,
                head_dim,
                page_block_size,
                workspace=reuse_k_workspace,
            )
            v = decompress_to_batched_paged_kv(
                ext,
                comp_v,
                batch_size_cache,
                kv_heads,
                head_dim,
                page_block_size,
                workspace=reuse_v_workspace,
            )
            return runner(k, v)

        out, latency_ms = time_cuda(staged_paged, warmup, iters, nvtx_label=nvtx_label)
    elif mode == FUSED_FLASH_ATTN_MODE:
        if zipserv_runner is None:
            raise ValueError(f"{mode} requires a prepared ZipServ fused runner")
        out, latency_ms = time_cuda(zipserv_runner, warmup, iters, nvtx_label=nvtx_label)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    metrics = {
        "latency_ms": float(latency_ms),
        "status": "ok",
    }
    return out, metrics


def main() -> None:
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
        "--fused_force_regular",
        action="store_true",
        help="Deprecated: fused_flashattn now defaults to FlashAttention's regular non-split path",
    )
    parser.add_argument(
        "--fused_splitkv",
        action="store_true",
        help="Opt back into the legacy split-kv fused FlashAttention path for comparison",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This benchmark is GPU-only.")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    modes = normalize_modes(args.modes)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    supported_modes = {
        FLASH_ATTN_MODE,
        FLASHINFER_MODE,
        STAGED_FLASH_ATTN_MODE,
        STAGED_FLASHINFER_MODE,
        FUSED_FLASH_ATTN_MODE,
    }
    for mode in modes:
        if mode not in supported_modes:
            raise ValueError(f"Unsupported mode: {mode}")
    availability_or_raise(modes)

    ext = load_zipserv_extension()
    config = load_model_config(args.model_root)
    q_entry = load_manifest_entry(args.model_root, args.layer)
    q_weight = load_bf16_bin(Path(q_entry["path"]), q_entry["shape"])
    hidden_size = int(config["hidden_size"])
    num_q_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden_size // num_q_heads
    if ({FUSED_FLASH_ATTN_MODE} & set(modes)) and num_kv_heads != 8:
        raise ValueError(f"{FUSED_FLASH_ATTN_MODE} currently requires num_kv_heads == 8, got {num_kv_heads}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    kv_pairs = load_kv_pairs(args.kv_dir)
    max_tokens = min(pair.k.shape[0] for pair in kv_pairs)
    if args.kv_len <= 0:
        raise ValueError(f"kv_len must be positive, got {args.kv_len}")
    if args.kv_len > max_tokens:
        raise ValueError(f"Requested kv_len {args.kv_len} exceeds available length {max_tokens}")
    kv_pair_offset = args.layer % len(kv_pairs)

    rows = []
    header_state = {"printed": False}
    use_integrated_flashattn_family = bool({FUSED_FLASH_ATTN_MODE} & set(modes))
    # Default fused experiments to the regular decode kernel. The split-kv ZipServ path remains
    # opt-in for comparison, but it is no longer the benchmark default or the default build.
    use_fused_regular = not args.fused_splitkv or args.fused_force_regular
    for batch_size in batch_sizes:
        batch_kv_pairs = select_kv_pairs(kv_pairs, batch_size, start_pair_idx=kv_pair_offset)
        dense_k, dense_v = build_batched_dense_kv(batch_kv_pairs, args.kv_len, device)
        q = build_q(q_weight, hidden_size, num_q_heads, head_dim, args.seed, device, batch_size)

        dense_comp_k = None
        dense_comp_v = None
        if {FLASH_ATTN_MODE, STAGED_FLASH_ATTN_MODE} & set(modes):
            padded_k, logical_rows, _ = pad_batched_kv_prefix(dense_k)
            padded_v, _, _ = pad_batched_kv_prefix(dense_v)
            dense_comp_k = make_zipserv_compressed(ext, padded_k, logical_rows, head_dim)
            dense_comp_v = make_zipserv_compressed(ext, padded_v, logical_rows, head_dim)

        flashinfer_k_cache = None
        flashinfer_v_cache = None
        flashinfer_comp_k = None
        flashinfer_comp_v = None
        if {FLASHINFER_MODE, STAGED_FLASHINFER_MODE} & set(modes):
            flashinfer_k_cache, _ = make_batched_paged_kv_cache(dense_k, FLASHINFER_PAGE_BLOCK_SIZE)
            flashinfer_v_cache, _ = make_batched_paged_kv_cache(dense_v, FLASHINFER_PAGE_BLOCK_SIZE)
            padded_k_flashinfer, flashinfer_logical_rows, _ = pad_batched_kv_prefix_paged(dense_k, FLASHINFER_PAGE_BLOCK_SIZE)
            padded_v_flashinfer, _, _ = pad_batched_kv_prefix_paged(dense_v, FLASHINFER_PAGE_BLOCK_SIZE)
            flashinfer_comp_k = make_zipserv_compressed(ext, padded_k_flashinfer, flashinfer_logical_rows, head_dim)
            flashinfer_comp_v = make_zipserv_compressed(ext, padded_v_flashinfer, flashinfer_logical_rows, head_dim)

        fused_comp_k = None
        fused_comp_v = None
        if FUSED_FLASH_ATTN_MODE in modes:
            padded_k_fused, fused_logical_rows, _ = pad_batched_kv_prefix_by_kv_head(dense_k)
            padded_v_fused, _, _ = pad_batched_kv_prefix_by_kv_head(dense_v)
            fused_comp_k = make_zipserv_compressed(ext, padded_k_fused, fused_logical_rows, head_dim)
            fused_comp_v = make_zipserv_compressed(ext, padded_v_fused, fused_logical_rows, head_dim)

        flashattn_baseline_ms = None
        flashattn_runner = None
        flashattn_modes_requested = bool({FLASH_ATTN_MODE, STAGED_FLASH_ATTN_MODE, FUSED_FLASH_ATTN_MODE} & set(modes))
        if flashattn_modes_requested:
            flashattn_runner = make_flash_attn_stage_runner(
                q,
                args.kv_len,
                use_integrated_extension=use_integrated_flashattn_family,
                force_regular=use_integrated_flashattn_family and use_fused_regular,
            )
            _, metrics = benchmark_one(
                ext,
                "direct",
                dense_k,
                dense_v,
                dense_comp_k,
                dense_comp_v,
                args.kv_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                args.warmup,
                args.iters,
                runner=flashattn_runner,
            )
            flashattn_baseline_ms = float(metrics["latency_ms"])
            if FLASH_ATTN_MODE in modes:
                rows.append({
                    "layer": args.layer,
                    "mode": FLASH_ATTN_MODE,
                    "batch_size": batch_size,
                    "kv_len": args.kv_len,
                    "q_heads": num_q_heads,
                    "kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "base_path": 1.0,
                    **metrics,
                })
                print_terminal_row(rows[-1], header_state)
            if STAGED_FLASH_ATTN_MODE in modes:
                if flashattn_runner is None:
                    raise RuntimeError("staged_flashattn requires flash_attn_with_kvcache to be available")
                flashattn_k_workspace = torch.empty((dense_comp_k.rows, dense_comp_k.cols), dtype=torch.bfloat16, device=q.device)
                flashattn_v_workspace = torch.empty((dense_comp_v.rows, dense_comp_v.cols), dtype=torch.bfloat16, device=q.device)
                _, staged_metrics = benchmark_one(
                    ext,
                    "staged_dense",
                    dense_k,
                    dense_v,
                    dense_comp_k,
                    dense_comp_v,
                    args.kv_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    args.warmup,
                    args.iters,
                    runner=flashattn_runner,
                    reuse_k_workspace=flashattn_k_workspace,
                    reuse_v_workspace=flashattn_v_workspace,
                    cache_batch_size=batch_size,
                )
                rows.append({
                    "layer": args.layer,
                    "mode": STAGED_FLASH_ATTN_MODE,
                    "batch_size": batch_size,
                    "kv_len": args.kv_len,
                    "q_heads": num_q_heads,
                    "kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "base_path": baseline_ratio(float(staged_metrics["latency_ms"]), flashattn_baseline_ms),
                    **staged_metrics,
                })
                print_terminal_row(rows[-1], header_state)
            if FUSED_FLASH_ATTN_MODE in modes:
                if flashattn_runner is None:
                    raise RuntimeError("fused_flashattn requires flash_attn_with_kvcache to be available")
                zipserv_runner = make_flash_attn_zipserv_runner(
                    q,
                    args.kv_len,
                    num_kv_heads,
                    fused_comp_k,
                    fused_comp_v,
                    force_regular=use_fused_regular,
                )
                _, fused_metrics = benchmark_one(
                    ext,
                    FUSED_FLASH_ATTN_MODE,
                    dense_k,
                    dense_v,
                    fused_comp_k,
                    fused_comp_v,
                    args.kv_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    args.warmup,
                    args.iters,
                    zipserv_runner=zipserv_runner,
                )
                rows.append({
                    "layer": args.layer,
                    "mode": FUSED_FLASH_ATTN_MODE,
                    "batch_size": batch_size,
                    "kv_len": args.kv_len,
                    "q_heads": num_q_heads,
                    "kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "base_path": baseline_ratio(float(fused_metrics["latency_ms"]), flashattn_baseline_ms),
                    **fused_metrics,
                })
                print_terminal_row(rows[-1], header_state)
        if {FLASHINFER_MODE, STAGED_FLASHINFER_MODE} & set(modes):
            flashinfer_runner = make_flashinfer_runner(
                q,
                args.kv_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                FLASHINFER_PAGE_BLOCK_SIZE,
            )
            _, flashinfer_metrics = benchmark_one(
                ext,
                "direct",
                flashinfer_k_cache,
                flashinfer_v_cache,
                flashinfer_comp_k,
                flashinfer_comp_v,
                args.kv_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                args.warmup,
                args.iters,
                runner=flashinfer_runner,
            )
            flashinfer_baseline_ms = float(flashinfer_metrics["latency_ms"])
            if FLASHINFER_MODE in modes:
                rows.append({
                    "layer": args.layer,
                    "mode": FLASHINFER_MODE,
                    "batch_size": batch_size,
                    "kv_len": args.kv_len,
                    "q_heads": num_q_heads,
                    "kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "base_path": 1.0,
                    **flashinfer_metrics,
                })
                print_terminal_row(rows[-1], header_state)
            if STAGED_FLASHINFER_MODE in modes:
                flashinfer_k_workspace = torch.empty(
                    (flashinfer_comp_k.rows, flashinfer_comp_k.cols),
                    dtype=torch.bfloat16,
                    device=q.device,
                )
                flashinfer_v_workspace = torch.empty(
                    (flashinfer_comp_v.rows, flashinfer_comp_v.cols),
                    dtype=torch.bfloat16,
                    device=q.device,
                )
                _, staged_metrics = benchmark_one(
                    ext,
                    "staged_paged",
                    flashinfer_k_cache,
                    flashinfer_v_cache,
                    flashinfer_comp_k,
                    flashinfer_comp_v,
                    args.kv_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    args.warmup,
                    args.iters,
                    runner=flashinfer_runner,
                    reuse_k_workspace=flashinfer_k_workspace,
                    reuse_v_workspace=flashinfer_v_workspace,
                    page_block_size=FLASHINFER_PAGE_BLOCK_SIZE,
                    cache_batch_size=batch_size,
                )
                rows.append({
                    "layer": args.layer,
                    "mode": STAGED_FLASHINFER_MODE,
                    "batch_size": batch_size,
                    "kv_len": args.kv_len,
                    "q_heads": num_q_heads,
                    "kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "base_path": baseline_ratio(float(staged_metrics["latency_ms"]), flashinfer_baseline_ms),
                    **staged_metrics,
                })
                print_terminal_row(rows[-1], header_state)

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
        f"layer={args.layer} kv_len={args.kv_len} kv_pair_pool={len(kv_pairs)} "
        f"start_pair=({kv_pairs[kv_pair_offset].k_name}, {kv_pairs[kv_pair_offset].v_name})"
    )


if __name__ == "__main__":
    main()
