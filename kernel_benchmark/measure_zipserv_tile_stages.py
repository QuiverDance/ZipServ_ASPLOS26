#!/usr/bin/env python3

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FLASH_ATTN_ROOT = SCRIPT_DIR / "third_party" / "flash_attn_v283"
MODULE_NAME = "zipserv_tile_stage_bench_ext"
PYTHON_ABI_TAG = sys.implementation.cache_tag or f"py{sys.version_info.major}{sys.version_info.minor}"
BUILD_DIR = SCRIPT_DIR / ".prebuilt_extensions" / PYTHON_ABI_TAG / MODULE_NAME

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")


def load_bench_helpers():
    path = SCRIPT_DIR / "bench_zipserv_decode_attention.py"
    spec = importlib.util.spec_from_file_location("bench_zipserv_decode_attention", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_stage_extension(verbose: bool):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return load(
        name=MODULE_NAME,
        sources=[str(SCRIPT_DIR / "zipserv_tile_stage_bench_ext.cu")],
        extra_include_paths=[
            str(REPO_ROOT / "csrc"),
            str(FLASH_ATTN_ROOT / "csrc" / "flash_attn"),
            str(FLASH_ATTN_ROOT / "csrc" / "flash_attn" / "src"),
            str(FLASH_ATTN_ROOT / "csrc" / "cutlass" / "include"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--std=c++17",
            "-lineinfo",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(BUILD_DIR),
        verbose=verbose,
    )


def mean_and_std(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=0))


def main():
    parser = argparse.ArgumentParser(description="Measure ZipServ fused-path tile load/decomp stages")
    parser.add_argument("--kv-dir", type=Path, default=Path("/home/pjw7200/saved_kv_cache"))
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--kv-len", type=int, default=1020)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    bench = load_bench_helpers()
    zipserv_ext = bench.load_zipserv_extension()
    stage_ext = load_stage_extension(verbose=args.verbose_build)
    model_root = args.model_root or bench.DEFAULT_MODEL_ROOT
    model_config = bench.load_model_config(model_root)

    kv_pairs = bench.load_kv_pairs(args.kv_dir)
    if not kv_pairs:
        raise RuntimeError(f"No KV pairs found under {args.kv_dir}")
    pair = kv_pairs[args.pair_index % len(kv_pairs)]

    kv_len = min(args.kv_len, pair.k.shape[0])
    if kv_len < 64:
        raise ValueError(f"kv_len must be at least 64 for a full fused tile block, got {kv_len}")
    measured_kv_len = (kv_len // 64) * 64
    if measured_kv_len == 0:
        raise ValueError(f"Need at least one full 64-row block to measure pure tile stages, got kv_len={kv_len}")

    k_dense, v_dense = bench.build_batched_dense_kv([pair], kv_len, device)
    kv_heads = int(k_dense.shape[2])
    head_dim = int(k_dense.shape[3])
    if head_dim != 128:
        raise ValueError(f"This microbenchmark expects head_dim == 128, got {head_dim}")
    num_q_heads = int(model_config["num_attention_heads"])
    num_kv_heads = int(model_config["num_key_value_heads"])
    if num_kv_heads != kv_heads:
        raise ValueError(f"Model config num_key_value_heads={num_kv_heads} does not match loaded KV heads={kv_heads}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    actual_seqlen_q = num_q_heads // num_kv_heads

    k_comp, v_comp = bench.compress_kv_pair_with_layout(
        zipserv_ext,
        (k_dense, v_dense),
        head_dim,
        bench.pad_batched_kv_prefix_by_kv_head,
    )

    batch_stride_rows = k_comp.rows
    head_stride_rows = k_comp.rows // kv_heads
    total_block_count = measured_kv_len // 64

    results = []
    for kv_head in range(kv_heads):
        for block_idx in range(total_block_count):
            block_row = block_idx * 64
            out = stage_ext.measure_zipserv_tile_stages(
                k_comp.sign_mantissa,
                k_comp.compressed_full,
                k_comp.bitmap1,
                k_comp.bitmap2,
                k_comp.bitmap3,
                k_comp.tile_offsets_median,
                k_comp.tile_offsets_global,
                k_comp.rows,
                k_comp.cols,
                batch_stride_rows,
                head_stride_rows,
                k_comp.max_high_freq_count,
                k_comp.max_full_count,
                k_comp.start_exp,
                0,  # ZIPSERV_LAYOUT_HEAD_MAJOR
                v_comp.sign_mantissa,
                v_comp.compressed_full,
                v_comp.bitmap1,
                v_comp.bitmap2,
                v_comp.bitmap3,
                v_comp.tile_offsets_median,
                v_comp.tile_offsets_global,
                v_comp.rows,
                v_comp.cols,
                batch_stride_rows,
                head_stride_rows,
                v_comp.max_high_freq_count,
                v_comp.max_full_count,
                v_comp.start_exp,
                0,  # ZIPSERV_LAYOUT_HEAD_MAJOR
                0,  # batch_idx
                kv_head,
                block_row,
                actual_seqlen_q,
                64,
                measured_kv_len,
                args.warmup,
                args.iters,
            )
            results.append(np.asarray(out, dtype=np.float64))

    samples = np.stack(results, axis=0)
    names = [
        "k1_load_us",
        "k1_decomp_us",
        "k2_load_us",
        "k2_decomp_us",
        "v1_load_us",
        "v1_decomp_us",
        "v2_load_us",
        "v2_decomp_us",
        "qk_gemm_us",
        "softmax_us",
        "pv_gemm_us",
        "timing_bracket_us",
    ]

    print(f"device={torch.cuda.get_device_name(args.device)}")
    print(f"kv_pair=({pair.k_name}, {pair.v_name})")
    print(
        f"kv_len={kv_len}, measured_kv_len={measured_kv_len}, kv_heads={kv_heads}, q_heads={num_q_heads}, "
        f"actual_seqlen_q={actual_seqlen_q}, head_dim={head_dim}, full_blocks={total_block_count}"
    )
    if measured_kv_len != kv_len:
        print(f"tail_tokens_excluded={kv_len - measured_kv_len} (pure stage timing uses full 64-row blocks only)")
    print(f"samples={samples.shape[0]} (heads={kv_heads} x blocks={total_block_count}), warmup={args.warmup}, iters={args.iters}")
    print("")
    for idx, name in enumerate(names):
        mean_us, std_us = mean_and_std(samples[:, idx])
        print(f"{name}: mean={mean_us:.3f} us, std={std_us:.3f} us")


if __name__ == "__main__":
    main()
