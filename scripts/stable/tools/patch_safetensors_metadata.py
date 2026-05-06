import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_network_args(metadata: dict[str, str]) -> dict:
    raw = metadata.get("ss_network_args")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _build_compat_metadata(metadata: dict[str, str]) -> dict[str, str]:
    network_module = str(metadata.get("ss_network_module", "") or "").strip()
    normalized_network_module = network_module.lower()
    network_args = _parse_network_args(metadata)
    training_algo = str(metadata.get("ss_training_algo", "") or "").strip()

    if not training_algo:
        algo_value = network_args.get("algo")
        if algo_value not in (None, ""):
            training_algo = str(algo_value).strip()

    patches: dict[str, str] = {}

    if network_module:
        patches["ss_training_network_module"] = network_module

    if network_args:
        patches["ss_training_network_args"] = json.dumps(network_args, ensure_ascii=False)

    if training_algo:
        patches["ss_training_algo"] = training_algo
        patches["ss_network_type"] = training_algo

    if normalized_network_module == "lycoris.kohya":
        patches["ss_training_is_lycoris"] = "True"
        patches.setdefault("ss_network_type", "lycoris")
        lycoris_algo = str(network_args.get("algo", "") or "").strip()
        if lycoris_algo:
            patches["ss_lycoris_algo"] = lycoris_algo
            patches["ss_training_lycoris_algo"] = lycoris_algo
            patches["ss_network_type"] = lycoris_algo

    if "ss_network_type" not in patches and not training_algo:
        if normalized_network_module.startswith("networks.lora_fa"):
            patches["ss_network_type"] = "lora_fa"
        elif normalized_network_module.startswith("networks.vera"):
            patches["ss_network_type"] = "vera"
        elif normalized_network_module.startswith("networks.tlora"):
            patches["ss_network_type"] = "tlora"
        elif normalized_network_module.startswith("networks.dylora"):
            patches["ss_network_type"] = "dylora"
        elif normalized_network_module.startswith("networks.oft"):
            patches["ss_network_type"] = "oft"
        elif normalized_network_module.startswith("networks.lokr"):
            patches["ss_network_type"] = "lokr"
        elif normalized_network_module.startswith("networks.lora"):
            patches["ss_network_type"] = "lora"

    if "dora_wd" in network_args:
        patches["ss_dora_enabled"] = "True" if _boolish(network_args.get("dora_wd")) else "False"

    if "train_norm" in network_args:
        patches["ss_train_norm_enabled"] = "True" if _boolish(network_args.get("train_norm")) else "False"

    return patches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Input safetensors model path")
    parser.add_argument("--output", help="Output path; defaults to overwrite input")
    parser.add_argument("--backup", action="store_true", help="Create .bak copy before overwriting input")
    args = parser.parse_args()

    src_path = Path(args.model).resolve()
    dst_path = Path(args.output).resolve() if args.output else src_path

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(src_path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)

    patched = dict(metadata)
    patched.update(_build_compat_metadata(metadata))
    patched = {k: str(v) for k, v in patched.items()}

    if dst_path == src_path and args.backup:
        backup_path = src_path.with_suffix(src_path.suffix + ".bak")
        shutil.copy2(src_path, backup_path)
        print(f"Backup created: {backup_path}")

    save_file(tensors, str(dst_path), patched)
    print(json.dumps({
        "input": str(src_path),
        "output": str(dst_path),
        "patched_keys": sorted(set(patched.keys()) - set(metadata.keys())),
        "updated_keys": sorted(key for key in patched.keys() if metadata.get(key) != patched.get(key)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
