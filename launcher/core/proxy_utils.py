"""Helpers for applying launcher proxy settings to urllib/network calls."""

from __future__ import annotations

from typing import Any, Dict, Optional


def normalize_proxy_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, str]:
    settings = settings or {}
    result: Dict[str, str] = {}
    for key in ("http_proxy", "https_proxy", "all_proxy"):
        value = str(settings.get(key) or "").strip()
        if value:
            result[key] = value
    return result


def build_urllib_proxy_handler(proxy_settings: Optional[Dict[str, Any]]):
    normalized = normalize_proxy_settings(proxy_settings)
    mapping: Dict[str, str] = {}
    if normalized.get("http_proxy"):
        mapping["http"] = normalized["http_proxy"]
    if normalized.get("https_proxy"):
        mapping["https"] = normalized["https_proxy"]
    if normalized.get("all_proxy"):
        mapping.setdefault("http", normalized["all_proxy"])
        mapping.setdefault("https", normalized["all_proxy"])
        mapping["all"] = normalized["all_proxy"]
    return mapping
