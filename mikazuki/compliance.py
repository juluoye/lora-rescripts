from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional


PROJECT_NAME = "lora-rescripts"
OFFICIAL_REPO_URL = "https://github.com/WhitecrowAurora/lora-rescripts"
PROJECT_LICENSE = "GNU AGPL-3.0-or-later"
PROJECT_COPYRIGHT = "Copyright (C) WhitecrowAurora and contributors"
COMPLIANCE_VERSION = "lulynx-compliance-v1"
METADATA_SIGNATURE_ENV = "LULYNX_METADATA_SIGNING_SECRET"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_project_version(default: str = "unknown") -> str:
    version_path = repo_root() / "version.json"
    try:
        payload = json.loads(version_path.read_text(encoding="utf-8"))
        version = str(payload.get("version", "") or "").strip()
        if version:
            return version
    except Exception:
        pass
    return default


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return str(value)
    return str(value)


def build_runtime_banner_lines(
    *,
    script_path: Optional[str] = None,
    git_commit: Optional[str] = None,
    runtime_mode: Optional[str] = None,
    extra_notice: Optional[str] = None,
) -> list[str]:
    version = get_project_version()
    lines = [
        f"{PROJECT_NAME} {version}",
        f"Source: {OFFICIAL_REPO_URL}",
        f"License: {PROJECT_LICENSE}",
        f"Copyright: {PROJECT_COPYRIGHT}",
        "Compliance: modified builds and hosted services must provide corresponding source and preserve notices.",
        "合规提示：修改版或通过网络向他人提供服务的版本，应提供对应源码并保留来源声明。",
    ]
    if git_commit:
        lines.append(f"Commit: {git_commit}")
    if runtime_mode:
        lines.append(f"Runtime: {runtime_mode}")
    if script_path:
        lines.append(f"Entry: {script_path}")
    if extra_notice:
        lines.append(str(extra_notice))
    return lines


def emit_runtime_banner(
    *,
    printer,
    script_path: Optional[str] = None,
    git_commit: Optional[str] = None,
    runtime_mode: Optional[str] = None,
    extra_notice: Optional[str] = None,
) -> None:
    for line in build_runtime_banner_lines(
        script_path=script_path,
        git_commit=git_commit,
        runtime_mode=runtime_mode,
        extra_notice=extra_notice,
    ):
        printer(line)


def _metadata_payload_fields(metadata: Mapping[str, Any]) -> dict[str, str]:
    keys = (
        "ss_output_name",
        "ss_session_id",
        "ss_training_started_at",
        "ss_base_model_version",
        "ss_network_module",
        "ss_training_algo",
        "ss_attention_backend",
        "ss_steps",
        "ss_epoch",
    )
    return {key: _normalize_value(metadata.get(key)) for key in keys if key in metadata}


def build_lulynx_metadata_fields(
    *,
    metadata: Mapping[str, Any],
    git_commit: str,
    model_hash: Optional[str] = None,
    metadata_signing_secret: Optional[str] = None,
) -> dict[str, str]:
    version = get_project_version()
    fields: dict[str, str] = {
        "lulynx_project_name": PROJECT_NAME,
        "lulynx_project_version": version,
        "lulynx_project_repo": OFFICIAL_REPO_URL,
        "lulynx_project_license": PROJECT_LICENSE,
        "lulynx_project_commit": str(git_commit or "").strip() or "(unknown)",
        "lulynx_compliance_version": COMPLIANCE_VERSION,
        "lulynx_training_notice": (
            f"Trained/exported with {PROJECT_NAME}. Modified or hosted builds should preserve notices and provide corresponding source under {PROJECT_LICENSE}."
        ),
        "lulynx_training_notice_zh": (
            f"本模型由 {PROJECT_NAME} 训练/导出。修改版或托管服务应保留来源声明，并按 {PROJECT_LICENSE} 提供对应源码。"
        ),
    }
    if model_hash:
        fields["lulynx_weight_fingerprint_v1"] = str(model_hash)

    payload = {
        "project": PROJECT_NAME,
        "version": version,
        "repo": OFFICIAL_REPO_URL,
        "license": PROJECT_LICENSE,
        "commit": fields["lulynx_project_commit"],
        "model_hash": str(model_hash or ""),
        "metadata": _metadata_payload_fields(metadata),
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fields["lulynx_signature_payload_v1"] = payload_json

    secret = metadata_signing_secret
    if secret is None:
        secret = os.environ.get(METADATA_SIGNATURE_ENV, "")
    secret = str(secret or "").strip()
    if secret:
        signature = hmac.new(secret.encode("utf-8"), payload_json.encode("utf-8"), hashlib.sha256).hexdigest()
        fields["lulynx_signature_scheme_v1"] = "hmac-sha256"
        fields["lulynx_signature_v1"] = signature
        fields["lulynx_signature_hint_v1"] = "official-signature-present"
    else:
        fields["lulynx_signature_scheme_v1"] = "unsigned"
        fields["lulynx_signature_hint_v1"] = (
            f"Set {METADATA_SIGNATURE_ENV} to enable official metadata signatures."
        )
    return fields


def write_export_notice_file(
    output_path: str | Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    git_commit: str = "",
    model_hash: Optional[str] = None,
    export_format: Optional[str] = None,
) -> None:
    payload_metadata = dict(metadata or {})
    notice = build_lulynx_metadata_fields(
        metadata=payload_metadata,
        git_commit=git_commit,
        model_hash=model_hash,
    )
    payload: dict[str, Any] = {
        "project_name": PROJECT_NAME,
        "project_version": get_project_version(),
        "project_repo": OFFICIAL_REPO_URL,
        "project_license": PROJECT_LICENSE,
        "project_commit": str(git_commit or "").strip() or "(unknown)",
        "compliance_version": COMPLIANCE_VERSION,
        "export_format": str(export_format or "").strip(),
        "notice_en": notice.get("lulynx_training_notice", ""),
        "notice_zh": notice.get("lulynx_training_notice_zh", ""),
        "metadata_fields": notice,
        "source_metadata": {key: _normalize_value(value) for key, value in payload_metadata.items()},
    }
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "COMPLIANCE_VERSION",
    "METADATA_SIGNATURE_ENV",
    "OFFICIAL_REPO_URL",
    "PROJECT_COPYRIGHT",
    "PROJECT_LICENSE",
    "PROJECT_NAME",
    "build_lulynx_metadata_fields",
    "build_runtime_banner_lines",
    "emit_runtime_banner",
    "get_project_version",
    "repo_root",
    "write_export_notice_file",
]
