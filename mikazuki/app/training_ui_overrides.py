from __future__ import annotations

from mikazuki.app.training_network_overrides import (
    apply_anima_ui_overrides,
    apply_flux_tlora_ui_overrides,
    apply_stable_tlora_ui_overrides,
    normalize_conflicting_network_target_flags,
)
from mikazuki.app.training_sdxl_overrides import (
    apply_sdxl_block_swap_ui_overrides,
    apply_sdxl_low_vram_ui_overrides,
    build_sdxl_clip_skip_warning,
)

MUON_OPTIMIZER_NAMES = {"muon", "adamuon", "distributedmuon"}
ADAMUON_OPTIMIZER_NAMES = {"adamuon"}


def _is_muon_optimizer(value: object) -> bool:
    optimizer_name = str(value or "").strip().split(".")[-1].lower()
    return optimizer_name in MUON_OPTIMIZER_NAMES


def _default_muon_use_muon_arg(value: object) -> str:
    optimizer_name = str(value or "").strip().split(".")[-1].lower()
    if optimizer_name in ADAMUON_OPTIMIZER_NAMES:
        return "use_muon=False"
    return "use_muon=True"


def _normalize_arg_lines(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = value.splitlines()
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]

    lines: list[str] = []
    for item in values:
        if item is None:
            continue
        line = str(item).strip()
        if line:
            lines.append(line)
    return lines


def _replace_or_append_optimizer_arg(args: list[str], key: str, line: str) -> list[str]:
    target_key = key.strip().lower()
    replaced = False
    normalized: list[str] = []
    for arg in args:
        arg_key = arg.split("=", 1)[0].strip().lower()
        if arg_key == target_key:
            if not replaced:
                normalized.append(line)
                replaced = True
            continue
        normalized.append(arg)
    if not replaced:
        normalized.append(line)
    return normalized


def _has_optimizer_arg(args: list[str], key: str) -> bool:
    target_key = key.strip().lower()
    return any(arg.split("=", 1)[0].strip().lower() == target_key for arg in args)


def _normalize_muon_use_muon_arg(value: object) -> str | None:
    if value is None:
        return None
    line = str(value).strip()
    if not line:
        return None
    if "=" not in line:
        return f"use_muon={line}"
    key, raw_value = line.split("=", 1)
    if key.strip().lower() != "use_muon":
        return None
    return f"use_muon={raw_value.strip()}"


def apply_muon_optimizer_ui_overrides(config: dict) -> None:
    requested_use_muon = _normalize_muon_use_muon_arg(config.pop("muon_use_muon", None))
    if not _is_muon_optimizer(config.get("optimizer_type")):
        return

    optimizer_args = _normalize_arg_lines(config.get("optimizer_args"))
    optimizer_args.extend(_normalize_arg_lines(config.pop("optimizer_args_custom", None)))
    if requested_use_muon is not None:
        optimizer_args = _replace_or_append_optimizer_arg(optimizer_args, "use_muon", requested_use_muon)
    elif not _has_optimizer_arg(optimizer_args, "use_muon"):
        optimizer_args = _replace_or_append_optimizer_arg(
            optimizer_args,
            "use_muon",
            _default_muon_use_muon_arg(config.get("optimizer_type")),
        )
    config["optimizer_args"] = optimizer_args


def apply_training_ui_overrides(config: dict) -> list[str]:
    warnings = apply_sdxl_low_vram_ui_overrides(config)
    warnings.extend(apply_sdxl_block_swap_ui_overrides(config))
    apply_anima_ui_overrides(config)
    apply_flux_tlora_ui_overrides(config)
    apply_stable_tlora_ui_overrides(config)
    apply_muon_optimizer_ui_overrides(config)
    warnings.extend(normalize_conflicting_network_target_flags(config))
    return warnings
