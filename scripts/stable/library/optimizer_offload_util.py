from __future__ import annotations

from typing import Any


def normalize_optimizer_offload_mode(value: Any, *, default_value: str = "ndim_ge_2") -> str:
    raw_value = str(value or "").strip().lower()
    if not raw_value:
        return default_value
    if raw_value in {"off", "false", "none", "disabled"}:
        return "off"
    if raw_value in {"all", "always"}:
        return "all"
    if raw_value in {"ndim_ge_2", "ndim>=2", "matrix_only", "matrix"}:
        return "ndim_ge_2"
    return default_value


def should_offload_optimizer_tensor(param, *, mode: str = "ndim_ge_2") -> bool:
    normalized_mode = normalize_optimizer_offload_mode(mode)
    if normalized_mode == "off":
        return False
    if normalized_mode == "all":
        return True
    return int(getattr(param, "ndim", 0) or 0) >= 2


__all__ = [
    "normalize_optimizer_offload_mode",
    "should_offload_optimizer_tensor",
]
