#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import math
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
FLASH_ATTN_EXT_NAME = "zipserv_flash_attn_ext"
PYTHON_ABI_TAG = sys.implementation.cache_tag or f"py{sys.version_info.major}{sys.version_info.minor}"
PREBUILT_EXT_ROOT = SCRIPT_DIR / ".prebuilt_extensions" / PYTHON_ABI_TAG
ZIPSERV_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / EXT_NAME
FLASH_ATTN_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / FLASH_ATTN_EXT_NAME

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
        ("backend", 12),
        ("mode", 18),
        ("kv_len", 8),
        ("q_heads", 8),
        ("kv_heads", 8),
        ("head_dim", 8),
        ("latency_ms", 12),
        ("base/path", 12),
    ]
    if not header_state["printed"]:
        print(" ".join(f"{name:>{width}}" for name, width in fields))
        header_state["printed"] = True
    values = []
    for field, width in fields:
        value = row[field]
        if isinstance(value, float):
            if field in {"latency_ms", "base/path"}:
                values.append(f"{value:>{width}.3f}")
                continue
            values.append(f"{value:>{width}.6f}")
        else:
            values.append(f"{str(value):>{width}}")
    print(" ".join(values))


def baseline_ratio(row_latency_ms: float, baseline_latency_ms: float | None) -> float:
    if baseline_latency_ms is None:
        return float("nan")
    if row_latency_ms <= 0.0:
        return float("inf")
    return baseline_latency_ms / row_latency_ms


def ensure_extension_built(build_target: str) -> None:
    import build_zipserv_decode_attention_extensions as ext_builder

    ext_builder.ensure_extension_built(build_target)


def load_prebuilt_extension(module_name: str, build_dir: Path, build_hint: str, build_target: str) -> object:
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


def load_zipserv_flash_attn_extension() -> object:
    build_hint = (
        "python build_zipserv_decode_attention_extensions.py --target zipserv_flashattn"
    )
    return load_prebuilt_extension(
        FLASH_ATTN_EXT_NAME,
        FLASH_ATTN_EXT_BUILD_DIR,
        build_hint,
        "zipserv_flashattn",
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


def load_kv_pair(kv_dir: Path, layer: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    files = sorted(p for p in kv_dir.iterdir() if p.is_file())
    pair_idx = layer * 2
    if pair_idx + 1 >= len(files):
        raise IndexError(f"Layer {layer} requires files {pair_idx} and {pair_idx + 1}, found {len(files)} total")
    k_path = files[pair_idx]
    v_path = files[pair_idx + 1]
    k_np = np.load(k_path)
    v_np = np.load(v_path)
    k = torch.from_numpy(k_np.view(np.uint16)).view(torch.bfloat16)
    v = torch.from_numpy(v_np.view(np.uint16)).view(torch.bfloat16)
    return k, v, k_path.name, v_path.name


def build_q(weight: torch.Tensor, hidden_size: int, num_heads: int, head_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    hidden = torch.randn((1, hidden_size), dtype=torch.float32, generator=g).to(dtype=torch.bfloat16, device=device)
    weight = weight.to(device=device, dtype=torch.bfloat16)
    q = hidden @ weight.t()
    return q.view(num_heads, head_dim).contiguous()


def pad_kv_prefix(x: torch.Tensor, kv_len: int) -> Tuple[torch.Tensor, int]:
    prefix = x[:kv_len].contiguous()
    logical_rows = prefix.shape[0] * prefix.shape[1]
    logical_cols = prefix.shape[2]
    rows = ((logical_rows + 63) // 64) * 64
    cols = ((logical_cols + 63) // 64) * 64
    out = torch.zeros((rows, cols), dtype=torch.bfloat16)
    out[:logical_rows, :logical_cols] = prefix.view(logical_rows, logical_cols)
    return out, logical_rows


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


def decompress_to_kv(
    ext: object,
    comp: ZipservCompressed,
    kv_len: int,
    kv_heads: int,
    head_dim: int,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    dense = decompress_single_to_2d(ext, comp, output_2d=workspace)
    return dense[: comp.logical_rows, :head_dim].view(kv_len, kv_heads, head_dim)


def availability_or_raise(backends: Iterable[str]) -> None:
    for backend in backends:
        if backend == "flashinfer":
            try:
                import flashinfer.decode  # noqa: F401
            except Exception as exc:
                raise RuntimeError(f"flashinfer backend is unavailable: {exc}") from exc
        else:
            raise ValueError(f"Unknown backend: {backend}")


def make_attention_runner(backend: str, q: torch.Tensor, kv_len: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if backend == "flashinfer":
        import flashinfer.decode

        scale = 1.0 / math.sqrt(q.shape[-1])

        def runner(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flashinfer.decode.single_decode_with_kv_cache(
                q,
                k,
                v,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                use_tensor_cores=True,
                sm_scale=scale,
            )

        return runner

    raise ValueError(f"Unknown backend: {backend}")


def dense_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    kv_group_size = num_q_heads // num_kv_heads
    kv_head_index = torch.arange(num_q_heads, device=q.device) // kv_group_size
    k_grouped = k[:, kv_head_index, :].permute(1, 0, 2).contiguous().float()
    v_grouped = v[:, kv_head_index, :].permute(1, 0, 2).contiguous().float()
    q_float = q.float()
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    scores = torch.sum(q_float.unsqueeze(1) * k_grouped, dim=-1) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.sum(probs.unsqueeze(-1) * v_grouped, dim=1)


def time_cuda(fn, warmup: int, iters: int):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        out = fn()
    stop.record()
    stop.synchronize()
    return out, start.elapsed_time(stop) / iters


def compare(reference: torch.Tensor, other: torch.Tensor) -> Tuple[float, float]:
    ref = reference.float()
    val = other.float()
    diff = (ref - val).abs()
    return float(diff.max().item()), float(diff.mean().item())


def benchmark_one(
    ext: object,
    backend: str,
    mode: str,
    q: torch.Tensor,
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
    dense_reference: torch.Tensor | None,
    runner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    reuse_k_workspace: torch.Tensor | None = None,
    reuse_v_workspace: torch.Tensor | None = None,
    flash_attn_ext: object | None = None,
) -> Tuple[torch.Tensor, Dict[str, float | str | int]]:
    out = None

    if mode == "dense":
        if runner is None:
            raise ValueError("dense mode requires a prepared attention runner")
        out, latency_ms = time_cuda(lambda: runner(dense_k, dense_v), warmup, iters)
    elif mode == "staged":
        if runner is None:
            raise ValueError("staged mode requires a prepared attention runner")

        def staged():
            k = decompress_to_kv(ext, comp_k, kv_len, kv_heads, head_dim)
            v = decompress_to_kv(ext, comp_v, kv_len, kv_heads, head_dim)
            return runner(k, v)

        out, latency_ms = time_cuda(staged, warmup, iters)
    elif mode == "staged_reuse":
        if runner is None:
            raise ValueError("staged_reuse mode requires a prepared attention runner")
        if reuse_k_workspace is None or reuse_v_workspace is None:
            raise ValueError("staged_reuse mode requires reusable K/V output buffers")

        def staged_reuse():
            k = decompress_to_kv(ext, comp_k, kv_len, kv_heads, head_dim, workspace=reuse_k_workspace)
            v = decompress_to_kv(ext, comp_v, kv_len, kv_heads, head_dim, workspace=reuse_v_workspace)
            return runner(k, v)

        out, latency_ms = time_cuda(staged_reuse, warmup, iters)
    elif mode == "zipserv_native":
        def native():
            return ext.zipserv_decode_attention(
                q,
                comp_k.sign_mantissa,
                comp_k.compressed_full,
                comp_k.bitmap1,
                comp_k.bitmap2,
                comp_k.bitmap3,
                comp_k.tile_offsets_median,
                comp_k.tile_offsets_global,
                comp_k.rows,
                comp_k.cols,
                comp_k.max_high_freq_count,
                comp_k.max_full_count,
                comp_k.start_exp,
                comp_v.sign_mantissa,
                comp_v.compressed_full,
                comp_v.bitmap1,
                comp_v.bitmap2,
                comp_v.bitmap3,
                comp_v.tile_offsets_median,
                comp_v.tile_offsets_global,
                comp_v.rows,
                comp_v.cols,
                comp_v.max_high_freq_count,
                comp_v.max_full_count,
                comp_v.start_exp,
                kv_len,
                num_q_heads,
                kv_heads,
                head_dim,
                1.0 / math.sqrt(float(head_dim)),
            )

        out, latency_ms = time_cuda(native, warmup, iters)
    elif mode == "zipserv_flashattn":
        if flash_attn_ext is None:
            raise ValueError("zipserv_flashattn mode requires the vendored flash-attn extension")

        q_4d = q.view(1, 1, num_q_heads, head_dim).contiguous()

        def flash_decode():
            out_4d, _ = flash_attn_ext.fwd_kvcache_zipserv(
                q_4d,
                comp_k.sign_mantissa,
                comp_k.compressed_full,
                comp_k.bitmap1,
                comp_k.bitmap2,
                comp_k.bitmap3,
                comp_k.tile_offsets_median,
                comp_k.tile_offsets_global,
                comp_k.rows,
                comp_k.cols,
                comp_k.max_high_freq_count,
                comp_k.max_full_count,
                comp_k.start_exp,
                comp_v.sign_mantissa,
                comp_v.compressed_full,
                comp_v.bitmap1,
                comp_v.bitmap2,
                comp_v.bitmap3,
                comp_v.tile_offsets_median,
                comp_v.tile_offsets_global,
                comp_v.rows,
                comp_v.cols,
                comp_v.max_high_freq_count,
                comp_v.max_full_count,
                comp_v.start_exp,
                kv_len,
                kv_heads,
                1.0 / math.sqrt(float(head_dim)),
            )
            return out_4d.squeeze(0).squeeze(0)

        out, latency_ms = time_cuda(flash_decode, warmup, iters)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if dense_reference is None:
        max_abs_err = 0.0
        mean_abs_err = 0.0
    else:
        max_abs_err, mean_abs_err = compare(dense_reference, out)

    metrics = {
        "latency_ms": float(latency_ms),
        "max_abs_err": float(max_abs_err),
        "mean_abs_err": float(mean_abs_err),
        "status": "ok",
    }
    return out, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ZipServ decode-attention benchmark")
    parser.add_argument("--model_root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--kv_dir", type=Path, default=DEFAULT_KV_DIR)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--backend", type=str, default="all", choices=["flashinfer", "all"])
    parser.add_argument("--modes", type=str, default="dense,staged,staged_reuse")
    parser.add_argument("--token_counts", type=str, default="1,16,128,512,1024,1535")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This benchmark is GPU-only.")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    modes = [token.strip() for token in args.modes.split(",") if token.strip()]
    supported_modes = {"dense", "staged", "staged_reuse", "zipserv_native", "zipserv_flashattn"}
    for mode in modes:
        if mode not in supported_modes:
            raise ValueError(f"Unsupported mode: {mode}")

    if args.backend == "all":
        requested_backends = ["flashinfer"]
    else:
        requested_backends = [args.backend]
    backends = requested_backends if any(mode in {"dense", "staged", "staged_reuse"} for mode in modes) else []
    availability_or_raise(backends)

    ext = load_zipserv_extension()
    flash_attn_ext = load_zipserv_flash_attn_extension() if "zipserv_flashattn" in modes else None
    config = load_model_config(args.model_root)
    q_entry = load_manifest_entry(args.model_root, args.layer)
    q_weight = load_bf16_bin(Path(q_entry["path"]), q_entry["shape"])
    hidden_size = int(config["hidden_size"])
    num_q_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden_size // num_q_heads
    if any(mode in {"zipserv_native", "zipserv_flashattn"} for mode in modes) and num_kv_heads != 8:
        raise ValueError(f"zipserv fused decode paths currently require num_kv_heads == 8, got {num_kv_heads}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    q = build_q(q_weight, hidden_size, num_q_heads, head_dim, args.seed, device)

    dense_k_cpu, dense_v_cpu, k_name, v_name = load_kv_pair(args.kv_dir, args.layer)
    max_tokens = dense_k_cpu.shape[0]
    token_counts = parse_csv_ints(args.token_counts)
    if token_counts[-1] > max_tokens:
        raise ValueError(f"Requested kv_len {token_counts[-1]} exceeds available length {max_tokens}")

    rows = []
    header_state = {"printed": False}
    for kv_len in token_counts:
        dense_k_prefix_cpu = dense_k_cpu[:kv_len].contiguous()
        dense_v_prefix_cpu = dense_v_cpu[:kv_len].contiguous()
        dense_k = dense_k_prefix_cpu.to(device=device, dtype=torch.bfloat16)
        dense_v = dense_v_prefix_cpu.to(device=device, dtype=torch.bfloat16)

        padded_k_cpu, logical_rows = pad_kv_prefix(dense_k_prefix_cpu, kv_len)
        padded_v_cpu, _ = pad_kv_prefix(dense_v_prefix_cpu, kv_len)
        comp_k = make_zipserv_compressed(ext, padded_k_cpu.to(device=device, dtype=torch.bfloat16), logical_rows, head_dim)
        comp_v = make_zipserv_compressed(ext, padded_v_cpu.to(device=device, dtype=torch.bfloat16), logical_rows, head_dim)

        dense_outputs: Dict[str, torch.Tensor] = {}
        dense_latencies: Dict[str, float] = {}
        torch_dense_reference = None
        torch_dense_latency_ms = None
        if any(mode in {"zipserv_native", "zipserv_flashattn"} for mode in modes):
            torch_dense_reference, torch_dense_latency_ms = time_cuda(
                lambda: dense_decode_attention(q, dense_k, dense_v, num_q_heads, num_kv_heads),
                args.warmup,
                args.iters,
            )

        for backend in backends:
            runner = make_attention_runner(backend, q, kv_len)
            if "dense" in modes:
                dense_out, dense_metrics = benchmark_one(
                    ext,
                    backend,
                    "dense",
                    q,
                    dense_k,
                    dense_v,
                    comp_k,
                    comp_v,
                    kv_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    args.warmup,
                    args.iters,
                    None,
                    runner=runner,
                )
                dense_outputs[backend] = dense_out
                dense_latencies[backend] = float(dense_metrics["latency_ms"])
                rows.append({
                    "layer": args.layer,
                    "backend": backend,
                    "mode": "dense",
                    "kv_len": kv_len,
                    "q_heads": num_q_heads,
                    "kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "base/path": 1.0,
                    **dense_metrics,
                })
                print_terminal_row(rows[-1], header_state)
            ref = dense_outputs.get(backend)
            for mode in modes:
                if mode in {"dense", "zipserv_native", "zipserv_flashattn"}:
                    continue
                reuse_k_workspace = None
                reuse_v_workspace = None
                if mode == "staged_reuse":
                    reuse_k_workspace = torch.empty((comp_k.rows, comp_k.cols), dtype=torch.bfloat16, device=q.device)
                    reuse_v_workspace = torch.empty((comp_v.rows, comp_v.cols), dtype=torch.bfloat16, device=q.device)
                out, metrics = benchmark_one(
                    ext,
                    backend,
                    mode,
                    q,
                    dense_k,
                    dense_v,
                    comp_k,
                    comp_v,
                    kv_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    args.warmup,
                    args.iters,
                    ref,
                    runner=runner,
                    reuse_k_workspace=reuse_k_workspace,
                    reuse_v_workspace=reuse_v_workspace,
                )
                rows.append({
                    "layer": args.layer,
                    "backend": backend,
                    "mode": mode,
                    "kv_len": kv_len,
                    "q_heads": num_q_heads,
                    "kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "base/path": baseline_ratio(float(metrics["latency_ms"]), dense_latencies.get(backend)),
                    **metrics,
                })
                print_terminal_row(rows[-1], header_state)

        if "zipserv_native" in modes:
            native_ref = torch_dense_reference
            native_baseline_ms = torch_dense_latency_ms
            _, metrics = benchmark_one(
                ext,
                "zipserv_native",
                "zipserv_native",
                q,
                dense_k,
                dense_v,
                comp_k,
                comp_v,
                kv_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                args.warmup,
                args.iters,
                native_ref,
            )
            rows.append({
                "layer": args.layer,
                "backend": "zipserv_native",
                "mode": "zipserv_native",
                "kv_len": kv_len,
                "q_heads": num_q_heads,
                "kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "base/path": baseline_ratio(float(metrics["latency_ms"]), native_baseline_ms),
                **metrics,
            })
            print_terminal_row(rows[-1], header_state)

        if "zipserv_flashattn" in modes:
            flash_ref = torch_dense_reference
            flash_baseline_ms = torch_dense_latency_ms
            _, metrics = benchmark_one(
                ext,
                "flash_attn_ck",
                "zipserv_flashattn",
                q,
                dense_k,
                dense_v,
                comp_k,
                comp_v,
                kv_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                args.warmup,
                args.iters,
                flash_ref,
                flash_attn_ext=flash_attn_ext,
            )
            rows.append({
                "layer": args.layer,
                "backend": "flash_attn_ck",
                "mode": "zipserv_flashattn",
                "kv_len": kv_len,
                "q_heads": num_q_heads,
                "kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "base/path": baseline_ratio(float(metrics["latency_ms"]), flash_baseline_ms),
                **metrics,
            })
            print_terminal_row(rows[-1], header_state)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer",
        "backend",
        "mode",
        "kv_len",
        "q_heads",
        "kv_heads",
        "head_dim",
        "latency_ms",
        "base/path",
        "max_abs_err",
        "mean_abs_err",
        "status",
    ]
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print(f"layer={args.layer} kv_pair=({k_name}, {v_name})")


if __name__ == "__main__":
    main()
