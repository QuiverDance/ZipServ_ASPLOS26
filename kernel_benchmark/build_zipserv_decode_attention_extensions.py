#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

from torch.utils.cpp_extension import load


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXT_NAME = "zipserv_decode_attention_ext"
FLASH_ATTN_EXT_NAME = "zipserv_flash_attn_ext"
PYTHON_ABI_TAG = sys.implementation.cache_tag or f"py{sys.version_info.major}{sys.version_info.minor}"
PREBUILT_EXT_ROOT = SCRIPT_DIR / ".prebuilt_extensions" / PYTHON_ABI_TAG
ZIPSERV_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / EXT_NAME
FLASH_ATTN_EXT_BUILD_DIR = PREBUILT_EXT_ROOT / FLASH_ATTN_EXT_NAME
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


def build_zipserv_flash_attn_extension(verbose: bool) -> Path | None:
    FLASH_ATTN_EXT_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    load(
        name=FLASH_ATTN_EXT_NAME,
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
        build_directory=str(FLASH_ATTN_EXT_BUILD_DIR),
        verbose=verbose,
    )
    return newest_shared_object(FLASH_ATTN_EXT_BUILD_DIR, FLASH_ATTN_EXT_NAME)


def ensure_extension_built(target: str, verbose: bool = False) -> Path | None:
    if target == "zipserv":
        return build_zipserv_extension(verbose=verbose)
    if target == "zipserv_flashattn":
        return build_zipserv_flash_attn_extension(verbose=verbose)
    if target == "all":
        build_zipserv_extension(verbose=verbose)
        return build_zipserv_flash_attn_extension(verbose=verbose)
    raise ValueError(f"Unknown target: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild decode-attention benchmark extensions without running the benchmark")
    parser.add_argument("--target", choices=["zipserv", "zipserv_flashattn", "all"], default="all")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    PREBUILT_EXT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"PYTHON_ABI_TAG={PYTHON_ABI_TAG}")
    print(f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST', '<unset>')}")
    print(f"MAX_JOBS={os.environ.get('MAX_JOBS', '<unset>')}")

    if args.target in {"zipserv", "all"}:
        so_path = build_zipserv_extension(verbose=args.verbose)
        print(f"{EXT_NAME}: {so_path if so_path is not None else 'built, .so path not found'}")
    if args.target in {"zipserv_flashattn", "all"}:
        so_path = build_zipserv_flash_attn_extension(verbose=args.verbose)
        print(f"{FLASH_ATTN_EXT_NAME}: {so_path if so_path is not None else 'built, .so path not found'}")


if __name__ == "__main__":
    main()
