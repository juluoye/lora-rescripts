from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Optional

from mikazuki.launch_utils import base_dir_path
from mikazuki.training_route_contract import extract_route_contract_metadata, resolve_training_route_contract
from mikazuki.utils.runtime_mode import infer_attention_runtime_mode


_SUPPORTED_DYNAMO_BACKENDS = {"inductor", "cudagraphs"}


@dataclass(frozen=True)
class TorchCompileCacheContext:
    enabled: bool
    cache_root: Path
    inductor_cache_dir: Path
    triton_cache_dir: Path
    manifest_path: Path
    runtime_name: str
    training_type: str
    route_kind: str
    route_label: str
    model_name: str
    model_hash: str
    torch_version: str
    python_tag: str
    backend: str
    precision: str

    @property
    def env_overrides(self) -> dict[str, str]:
        return {
            "TORCHINDUCTOR_CACHE_DIR": str(self.inductor_cache_dir),
            "TRITON_CACHE_DIR": str(self.triton_cache_dir),
            "TRITON_HOME": str(self.cache_root),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cache_root": str(self.cache_root),
            "inductor_cache_dir": str(self.inductor_cache_dir),
            "triton_cache_dir": str(self.triton_cache_dir),
            "runtime_name": self.runtime_name,
            "training_type": self.training_type,
            "route_kind": self.route_kind,
            "route_label": self.route_label,
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "torch_version": self.torch_version,
            "python_tag": self.python_tag,
            "backend": self.backend,
            "precision": self.precision,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _slugify(text: str, *, fallback: str = "unknown", max_length: int = 80) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "").strip())
    normalized = re.sub(r"-+", "-", normalized).strip("._-")
    if not normalized:
        normalized = fallback
    return normalized[:max_length].strip("._-") or fallback


def _safe_version_tag(version_text: str) -> str:
    return _slugify(version_text, fallback="unknown-version", max_length=64)


def _resolve_model_name(model_ref: str) -> str:
    raw = str(model_ref or "").strip()
    if not raw:
        return "unknown-model"

    normalized = raw.replace("\\", "/").rstrip("/")
    if "/" in normalized:
        candidate = normalized.rsplit("/", 1)[-1]
    else:
        candidate = normalized

    candidate = candidate or normalized
    candidate = candidate.rsplit(".", 1)[0] if "." in candidate else candidate
    return _slugify(candidate, fallback="unknown-model")


def _resolve_torch_version() -> str:
    try:
        return metadata.version("torch")
    except Exception:
        return "unknown"


def _resolve_route_contract(config_data: Mapping[str, Any]) -> dict[str, str]:
    contract = extract_route_contract_metadata(config_data)
    if contract:
        return contract

    route = resolve_training_route_contract(
        str(config_data.get("model_train_type", "") or ""),
        config=config_data,
    )
    return route.as_metadata_fields()


def build_torch_compile_cache_context(
    config_data: Mapping[str, Any],
    customize_env: Mapping[str, str],
    *,
    repo_root: Optional[Path] = None,
) -> Optional[TorchCompileCacheContext]:
    if not _boolish(config_data.get("torch_compile")):
        return None

    backend = str(config_data.get("dynamo_backend", "inductor") or "inductor").strip().lower()
    if backend not in _SUPPORTED_DYNAMO_BACKENDS:
        return None

    route_contract = _resolve_route_contract(config_data)
    training_type = _slugify(route_contract.get("lulynx_route_training_type", "") or config_data.get("model_train_type", "") or "training")
    route_kind = _slugify(route_contract.get("lulynx_route_kind", "") or "generic")
    route_label = str(route_contract.get("lulynx_route_label", "") or route_kind).strip() or route_kind

    runtime_name = infer_attention_runtime_mode(customize_env)
    runtime_name = _slugify(runtime_name or "system")

    model_ref = str(config_data.get("pretrained_model_name_or_path") or config_data.get("network_weights") or "")
    if not model_ref:
        model_ref = str(config_data.get("model_train_type", "") or "model")
    model_name = _resolve_model_name(model_ref)
    model_hash = hashlib.sha1(model_ref.encode("utf-8", errors="ignore")).hexdigest()[:12]

    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    torch_version = _resolve_torch_version()
    precision = "full-bf16" if _boolish(config_data.get("full_bf16")) else "full-fp16" if _boolish(config_data.get("full_fp16")) else str(config_data.get("mixed_precision", "no") or "no").strip().lower()

    root = Path(repo_root or base_dir_path()).resolve()
    cache_root = root / "cache" / "torch_compile" / runtime_name / training_type / f"{model_name}-{model_hash}" / python_tag / f"torch-{_safe_version_tag(torch_version)}" / f"precision-{_safe_version_tag(precision)}" / f"backend-{_slugify(backend)}"
    inductor_cache_dir = cache_root / "inductor"
    triton_cache_dir = cache_root / "triton"
    manifest_path = cache_root / "manifest.json"

    return TorchCompileCacheContext(
        enabled=True,
        cache_root=cache_root,
        inductor_cache_dir=inductor_cache_dir,
        triton_cache_dir=triton_cache_dir,
        manifest_path=manifest_path,
        runtime_name=runtime_name,
        training_type=training_type,
        route_kind=route_kind,
        route_label=route_label,
        model_name=model_name,
        model_hash=model_hash,
        torch_version=torch_version,
        python_tag=python_tag,
        backend=backend,
        precision=precision,
    )


def apply_torch_compile_cache_env(
    customize_env: dict[str, str],
    config_data: Mapping[str, Any],
    *,
    repo_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[TorchCompileCacheContext]:
    context = build_torch_compile_cache_context(config_data, customize_env, repo_root=repo_root)
    if context is None:
        return None

    context.cache_root.mkdir(parents=True, exist_ok=True)
    context.inductor_cache_dir.mkdir(parents=True, exist_ok=True)
    context.triton_cache_dir.mkdir(parents=True, exist_ok=True)

    customize_env.update(context.env_overrides)

    manifest_payload = context.to_manifest()
    try:
        context.manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        if logger is not None:
            logger.warning(f"[torch.compile cache] failed to write manifest: {exc}")

    if logger is not None:
        logger.info(
            "[torch.compile cache] enabled for runtime=%s route=%s model=%s backend=%s root=%s",
            context.runtime_name,
            context.route_kind,
            context.model_name,
            context.backend,
            context.cache_root,
        )
        logger.info("[torch.compile cache] TORCHINDUCTOR_CACHE_DIR=%s", context.inductor_cache_dir)
        logger.info("[torch.compile cache] TRITON_CACHE_DIR=%s", context.triton_cache_dir)

    return context
