#!/usr/bin/env python3
"""Export selected Llama 3.1 70B tensors into raw binary files + JSONL manifest."""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from safetensors import safe_open


DEFAULT_BASE_DIR = Path("~/models/llama31_70b")
DEFAULT_SNAPSHOT_DIR = DEFAULT_BASE_DIR / "hf_snapshot"
DEFAULT_MANIFEST_PATH = DEFAULT_BASE_DIR / "manifest" / "weights_manifest.jsonl"

CANONICAL_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


@dataclass(frozen=True)
class ExportTask:
    layer: int
    suffix_index: int
    name: str
    shard: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export selected tensors from HF snapshot to raw 16-bit binaries.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/export_llama31_70b_weights.py \\\n"
            "    --snapshot-dir ~/models/llama31_70b/hf_snapshot\n"
            "  python scripts/export_llama31_70b_weights.py --layers 0 --filter q_proj\n"
            "  python scripts/export_llama31_70b_weights.py --layers 0-79 --dtype bf16"
        ),
    )
    parser.add_argument(
        "--snapshot-dir",
        default=str(DEFAULT_SNAPSHOT_DIR),
        help="HF snapshot directory containing config/index/shards.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16"],
        default="bf16",
        help="Export dtype for output binaries (fixed to bf16 for ZipServ benchmark).",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Layer selection, e.g. 0-79 or 0,1,2 or mixed 0-3,8,10-12.",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Comma-separated substring filters for tensor names (e.g. q_proj,up_proj).",
    )
    parser.add_argument(
        "--weights-dir",
        default=None,
        help="Output directory for tensor .bin files. Defaults to weights_<dtype> under model dir.",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Output JSONL manifest path.",
    )
    return parser.parse_args()


def parse_layer_spec(spec: str, max_layer: int) -> List[int]:
    if spec is None or spec.strip() == "":
        return list(range(max_layer + 1))
    selected = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                raise ValueError(f"Invalid layer range: {token}")
            for layer in range(start, end + 1):
                selected.add(layer)
        else:
            selected.add(int(token))
    layers = sorted(selected)
    for layer in layers:
        if layer < 0 or layer > max_layer:
            raise ValueError(f"Layer {layer} is out of range [0, {max_layer}]")
    return layers


def parse_name_filters(filter_arg: str) -> List[str]:
    if filter_arg is None or filter_arg.strip() == "":
        return []
    return [term.strip() for term in filter_arg.split(",") if term.strip()]


def name_matches_filters(name: str, filters: List[str]) -> bool:
    if not filters:
        return True
    return any(term in name for term in filters)


def load_json_file(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_tasks(
    weight_map: Dict[str, str],
    layers: Iterable[int],
    filters: List[str],
) -> List[ExportTask]:
    tasks: List[ExportTask] = []
    missing: List[str] = []
    for layer in layers:
        for suffix_index, suffix in enumerate(CANONICAL_SUFFIXES):
            name = f"model.layers.{layer}.{suffix}"
            if not name_matches_filters(name, filters):
                continue
            shard = weight_map.get(name)
            if shard is None:
                missing.append(name)
                continue
            tasks.append(ExportTask(layer=layer, suffix_index=suffix_index, name=name, shard=shard))
    if missing:
        print(f"Warning: {len(missing)} target tensors were not found in index.")
        for name in missing[:10]:
            print(f"  missing: {name}")
        if len(missing) > 10:
            print("  ...")
    return tasks


def export_tensor(task: ExportTask, source: torch.Tensor, export_dtype: str, weights_dir: Path) -> Dict:
    if export_dtype != "bf16":
        raise ValueError(f"Unsupported export dtype: {export_dtype}. Expected bf16.")
    out_tensor = source.to(torch.bfloat16)
    out_tensor = out_tensor.contiguous()

    out_path = (weights_dir / f"{task.name}.bin").resolve()
    expected_nbytes = out_tensor.numel() * 2

    byte_tensor = out_tensor.view(torch.uint8).contiguous()
    with out_path.open("wb") as fp:
        fp.write(byte_tensor.numpy().tobytes())

    actual_nbytes = out_path.stat().st_size
    if actual_nbytes != expected_nbytes:
        raise RuntimeError(
            f"nbytes mismatch for {task.name}: expected {expected_nbytes}, got {actual_nbytes}"
        )

    return {
        "sort_key": (task.layer, task.suffix_index, task.name),
        "record": {
            "layer": task.layer,
            "name": task.name,
            "shape": list(out_tensor.shape),
            "dtype": export_dtype,
            "path": str(out_path),
            "nbytes": actual_nbytes,
        },
    }


def main() -> None:
    args = parse_args()

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    config_path = snapshot_dir / "config.json"
    index_path = snapshot_dir / "model.safetensors.index.json"

    config = load_json_file(config_path)
    index_data = load_json_file(index_path)
    if "weight_map" not in index_data:
        raise KeyError(f"Missing 'weight_map' in index file: {index_path}")
    weight_map: Dict[str, str] = index_data["weight_map"]

    num_layers = int(config["num_hidden_layers"])
    max_layer = num_layers - 1
    selected_layers = parse_layer_spec(args.layers, max_layer=max_layer)
    filters = parse_name_filters(args.filter)

    if args.weights_dir is None:
        weights_dir = (DEFAULT_BASE_DIR / f"weights_{args.dtype}").expanduser().resolve()
    else:
        weights_dir = Path(args.weights_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()

    weights_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(weight_map=weight_map, layers=selected_layers, filters=filters)
    if not tasks:
        raise RuntimeError("No tensors selected for export. Check --layers / --filter.")

    tasks_by_shard: Dict[str, List[ExportTask]] = defaultdict(list)
    for task in tasks:
        tasks_by_shard[task.shard].append(task)
    for shard_tasks in tasks_by_shard.values():
        shard_tasks.sort(key=lambda t: (t.layer, t.suffix_index, t.name))

    exported = []
    total_nbytes = 0
    for shard_name in sorted(tasks_by_shard.keys()):
        shard_path = snapshot_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard_reader:
            for task in tasks_by_shard[shard_name]:
                tensor = shard_reader.get_tensor(task.name)
                payload = export_tensor(
                    task=task,
                    source=tensor,
                    export_dtype=args.dtype,
                    weights_dir=weights_dir,
                )
                exported.append(payload)
                total_nbytes += payload["record"]["nbytes"]

    exported.sort(key=lambda item: item["sort_key"])
    with manifest_path.open("w", encoding="utf-8") as manifest_fp:
        for item in exported:
            manifest_fp.write(json.dumps(item["record"], separators=(",", ":")) + "\n")

    print("Export finished.")
    print(f"snapshot_dir={snapshot_dir}")
    print(f"num_hidden_layers={num_layers}")
    print(f"selected_layers={len(selected_layers)}")
    print(f"selected_tensors={len(tasks)}")
    print(f"exported_tensors={len(exported)}")
    print(f"total_bytes={total_nbytes}")
    print(f"weights_dir={weights_dir}")
    print(f"manifest_path={manifest_path}")


if __name__ == "__main__":
    main()
