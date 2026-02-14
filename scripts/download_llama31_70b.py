#!/usr/bin/env python3
"""Download Llama 3.1 70B weights snapshot from Hugging Face."""

import argparse
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download


DEFAULT_OUT_DIR = "~/models/llama31_70b/hf_snapshot"
DEFAULT_REPO_ID = "meta-llama/Llama-3.1-70B"
DEFAULT_ALLOW_PATTERNS = ["*.safetensors", "*.json", "tokenizer*"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HF snapshot for Llama 3.1 70B benchmark inputs.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_llama31_70b.py\n"
            "  python scripts/download_llama31_70b.py --repo-id meta-llama/Llama-3.1-70B-Instruct\n"
            "  python scripts/download_llama31_70b.py \\\n"
            "    --allow-pattern '*.safetensors' --allow-pattern 'config.json' \\\n"
            "    --append-default-patterns"
        ),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repo id.")
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where the snapshot is materialized.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help=(
            "Allow pattern for snapshot_download (repeatable). "
            "If omitted, defaults are used."
        ),
    )
    parser.add_argument(
        "--append-default-patterns",
        action="store_true",
        help=(
            "Append custom --allow-pattern values to defaults. "
            "Without this flag, custom patterns override defaults."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not attempt network access; use locally cached files only.",
    )
    return parser.parse_args()


def resolve_allow_patterns(custom_patterns: List[str], append_defaults: bool) -> List[str]:
    if not custom_patterns:
        return list(DEFAULT_ALLOW_PATTERNS)
    if not append_defaults:
        return custom_patterns
    merged: List[str] = []
    for pattern in DEFAULT_ALLOW_PATTERNS + custom_patterns:
        if pattern not in merged:
            merged.append(pattern)
    return merged


def count_materialized_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file())


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = resolve_allow_patterns(args.allow_pattern, args.append_default_patterns)
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(out_dir),
        allow_patterns=allow_patterns,
        local_files_only=args.local_files_only,
        resume_download=True,
    )

    file_count = count_materialized_files(out_dir)
    print("Download finished.")
    print(f"repo_id={args.repo_id}")
    if args.revision is not None:
        print(f"revision={args.revision}")
    print(f"snapshot_path={snapshot_path}")
    print(f"out_dir={out_dir}")
    print(f"allow_patterns={allow_patterns}")
    print(f"materialized_files={file_count}")


if __name__ == "__main__":
    main()
