#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

from torch.utils.cpp_extension import load


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXT_NAME = "zipserv_decode_attention_ext"
FLASH_ATTN_EXT_NAME = "flash_attn_2_cuda"
ZIPSERV_FLASH_FUSED_EXT_NAME = "zipserv_flash_attn_ext"
PYTHON_ABI_TAG = sys.implementation.cache_tag or f"py{sys.version_info.major}{sys.version_info.minor}"
PREBUILT_EXT_ROOT = SCRIPT_DIR / ".prebuilt_extensions" / PYTHON_ABI_TAG
ZIPSERV_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / EXT_NAME
FLASH_ATTN_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / FLASH_ATTN_EXT_NAME
FLASH_ATTN_REGULAR_ONLY_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / f"{FLASH_ATTN_EXT_NAME}_regular_only"
ZIPSERV_FLASH_FUSED_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / ZIPSERV_FLASH_FUSED_EXT_NAME
FLASH_ATTN_ROOT = SCRIPT_DIR / "third_party" / "flash_attn_v283"

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")


def newest_shared_object(build_dir: Path, module_name: str) -> Path | None:
    candidates = sorted(build_dir.glob(f"{module_name}*.so"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def build_zipserv_extension(verbose: bool) -> Path | None:
    ZIPSERV_EXT_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    load(
        name=EXT_NAME,
        sources=[
            str(SCRIPT_DIR / "zipserv_decode_attention_ext.cpp"),
            str(SCRIPT_DIR / "zipserv_decode_attention_ext.cu"),
        ],
        extra_include_paths=[
            str(REPO_ROOT / "build"),
            str(REPO_ROOT / "csrc"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--std=c++17",
            "-lineinfo",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[
            f"-L{REPO_ROOT / 'build'}",
            "-lL_API",
            f"-Wl,-rpath,{REPO_ROOT / 'build'}",
        ],
        build_directory=str(ZIPSERV_EXT_BUILD_DIR),
        verbose=verbose,
    )
    return newest_shared_object(ZIPSERV_EXT_BUILD_DIR, EXT_NAME)


def build_flash_attn_2_cuda_extension(verbose: bool, regular_only: bool = False) -> Path | None:
    build_dir = FLASH_ATTN_REGULAR_ONLY_EXT_BUILD_DIR if regular_only else FLASH_ATTN_EXT_BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)
    flash_attn_src_dir = FLASH_ATTN_ROOT / "csrc" / "flash_attn" / "src"
    flash_attn_patterns = ["flash_fwd_hdim*_sm80.cu"] if regular_only else ["flash_*_sm80.cu"]
    flash_attn_sources = [
        str(FLASH_ATTN_ROOT / "csrc" / "flash_attn" / "flash_api.cpp"),
        *[
            str(path)
            for pattern in flash_attn_patterns
            for path in sorted(flash_attn_src_dir.glob(pattern))
        ],
    ]
    common_defines = ["-DFLASHATTENTION_DISABLE_BACKWARD"] if regular_only else []
    load(
        name=FLASH_ATTN_EXT_NAME,
        sources=flash_attn_sources,
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
            *common_defines,
            *([] if not regular_only else ["-DFLASHATTENTION_DISABLE_SPLITKV"]),
        ],
        extra_cflags=["-O3", "-std=c++17", *common_defines, *([] if not regular_only else ["-DFLASHATTENTION_DISABLE_SPLITKV"])],
        build_directory=str(build_dir),
        verbose=verbose,
    )
    return newest_shared_object(build_dir, FLASH_ATTN_EXT_NAME)


def build_zipserv_flash_fused_extension(verbose: bool) -> Path | None:
    ZIPSERV_FLASH_FUSED_EXT_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    load(
        name=ZIPSERV_FLASH_FUSED_EXT_NAME,
        sources=[
            str(FLASH_ATTN_ROOT / "csrc" / "flash_attn_ck" / "zipserv_flash_api.cpp"),
            str(FLASH_ATTN_ROOT / "csrc" / "flash_attn_ck" / "zipserv_fwd_kvcache.cu"),
        ],
        extra_include_paths=[
            str(REPO_ROOT / "csrc"),
            str(FLASH_ATTN_ROOT / "csrc" / "flash_attn" / "src"),
            str(FLASH_ATTN_ROOT / "csrc" / "flash_attn_ck"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--std=c++17",
            "-lineinfo",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(ZIPSERV_FLASH_FUSED_EXT_BUILD_DIR),
        verbose=verbose,
    )
    return newest_shared_object(ZIPSERV_FLASH_FUSED_EXT_BUILD_DIR, ZIPSERV_FLASH_FUSED_EXT_NAME)


def ensure_extension_built(target: str, verbose: bool = False) -> Path | None:
    if target == "zipserv":
        return build_zipserv_extension(verbose=verbose)
    if target == "zipserv_flashattn":
        return build_flash_attn_2_cuda_extension(verbose=verbose, regular_only=True)
    if target == "zipserv_flashattn_regular_only":
        return build_flash_attn_2_cuda_extension(verbose=verbose, regular_only=True)
    if target == "zipserv_flashattn_splitkv":
        return build_flash_attn_2_cuda_extension(verbose=verbose)
    if target == "zipserv_flashattn_ck":
        return build_zipserv_flash_fused_extension(verbose=verbose)
    if target == "all":
        build_zipserv_extension(verbose=verbose)
        build_flash_attn_2_cuda_extension(verbose=verbose, regular_only=True)
        build_zipserv_flash_fused_extension(verbose=verbose)
        return newest_shared_object(FLASH_ATTN_REGULAR_ONLY_EXT_BUILD_DIR, FLASH_ATTN_EXT_NAME)
    raise ValueError(f"Unknown target: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild decode-attention benchmark extensions without running the benchmark")
    parser.add_argument(
        "--target",
        choices=[
            "zipserv",
            "zipserv_flashattn",
            "zipserv_flashattn_regular_only",
            "zipserv_flashattn_splitkv",
            "zipserv_flashattn_ck",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    PREBUILT_EXT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"PYTHON_ABI_TAG={PYTHON_ABI_TAG}")
    print(f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST', '<unset>')}")
    print(f"MAX_JOBS={os.environ.get('MAX_JOBS', '<unset>')}")

    if args.target in {"zipserv", "all"}:
        so_path = build_zipserv_extension(verbose=args.verbose)
        print(f"{EXT_NAME}: {so_path if so_path is not None else 'built, .so path not found'}")
    if args.target in {"zipserv_flashattn", "zipserv_flashattn_regular_only", "all"}:
        so_path = build_flash_attn_2_cuda_extension(verbose=args.verbose, regular_only=True)
        print(f"{FLASH_ATTN_EXT_NAME} (regular-only): {so_path if so_path is not None else 'built, .so path not found'}")
    if args.target == "zipserv_flashattn_splitkv":
        so_path = build_flash_attn_2_cuda_extension(verbose=args.verbose)
        print(f"{FLASH_ATTN_EXT_NAME} (split-kv): {so_path if so_path is not None else 'built, .so path not found'}")
    if args.target in {"zipserv_flashattn_ck", "all"}:
        so_path = build_zipserv_flash_fused_extension(verbose=args.verbose)
        print(f"{ZIPSERV_FLASH_FUSED_EXT_NAME}: {so_path if so_path is not None else 'built, .so path not found'}")


if __name__ == "__main__":
    main()
