"""Shared runtime coordination for plans, settings merge, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Optional

from launcher.config import DEFAULT_HOST, DEFAULT_PORT, RUNTIME_MAP, RuntimeDef
from launcher.core.diagnostics import collect_health_report
from launcher.core.launcher import LaunchOptions
from launcher.core.preflight import collect_launch_preflight
from launcher.core.recommendation import recommend_runtime
from launcher.core.runtime_detector import RuntimeStatus, detect_all, get_best_runtime
from launcher.core.task_plans import build_install_plan, build_launch_plan


@dataclass
class PreparedLaunch:
    repo_root: Path
    runtime_id: str
    runtime_def: RuntimeDef
    statuses: Dict[str, RuntimeStatus]
    status: RuntimeStatus
    settings: Dict[str, Any]
    options: LaunchOptions
    preflight: Dict[str, Any]

    def build_plan(self):
        if not self.status.python_path:
            return None
        return build_launch_plan(
            runtime_def=self.runtime_def,
            python_path=self.status.python_path,
            options=self.options,
            repo_root=self.repo_root,
        )


@dataclass
class PreparedInstall:
    repo_root: Path
    runtime_id: str
    runtime_def: RuntimeDef
    statuses: Dict[str, RuntimeStatus]
    status: RuntimeStatus
    cn_mirror: bool
    proxy_settings: Dict[str, str]

    def build_plan(self):
        return build_install_plan(
            runtime_def=self.runtime_def,
            cn_mirror=self.cn_mirror,
            proxy_settings=self.proxy_settings,
            repo_root=self.repo_root,
        )


class RuntimeCoordinator:
    """Coordinates runtime settings, detection, and plan preparation."""

    def __init__(self, repo_root: Path, settings_provider: Callable[[], Dict[str, Any]]) -> None:
        self._repo_root = repo_root
        self._settings_provider = settings_provider
        self._status_cache_ttl_seconds = 3.0
        self._status_cache_lock = threading.Condition()
        self._status_cache: Optional[tuple[float, Dict[str, RuntimeStatus]]] = None
        self._status_cache_loading = False

    def invalidate_status_cache(self) -> None:
        with self._status_cache_lock:
            self._status_cache = None

    def get_statuses(self, *, force_refresh: bool = False) -> Dict[str, RuntimeStatus]:
        while True:
            with self._status_cache_lock:
                now = time.monotonic()
                if not force_refresh and self._status_cache is not None:
                    cached_at, cached_statuses = self._status_cache
                    if now - cached_at < self._status_cache_ttl_seconds:
                        return dict(cached_statuses)

                if self._status_cache_loading:
                    self._status_cache_lock.wait()
                    continue

                self._status_cache_loading = True
                break

        try:
            statuses = detect_all(self._repo_root)
        finally:
            with self._status_cache_lock:
                if 'statuses' in locals():
                    self._status_cache = (time.monotonic(), statuses)
                self._status_cache_loading = False
                self._status_cache_lock.notify_all()

        return dict(statuses)

    def get_serialized_statuses(self) -> Dict[str, Dict[str, Any]]:
        statuses = self.get_statuses()
        result: Dict[str, Dict[str, Any]] = {}
        for runtime_id, status in statuses.items():
            result[runtime_id] = {
                "runtime_id": status.runtime_id,
                "python_exists": status.python_exists,
                "deps_installed": status.deps_installed,
                "installed": status.installed,
                "python_path": str(status.python_path) if status.python_path else None,
                "env_dir": str(status.env_dir) if status.env_dir else None,
                "integrity_ok": status.integrity_ok,
                "bootstrap_ready": status.bootstrap_ready,
                "integrity_issue_code": status.integrity_issue_code,
                "integrity_message_zh": status.integrity_message_zh,
                "integrity_message_en": status.integrity_message_en,
                "status_text": status.status_text,
            }
        return result

    def get_best_runtime_id(self) -> Optional[str]:
        return get_best_runtime(self.get_statuses())

    def get_runtime_recommendation(self) -> Dict[str, Any]:
        statuses = self.get_statuses()
        return recommend_runtime(statuses, repo_root=self._repo_root)

    def get_health_report(self, selected_runtime_id: Optional[str] = None) -> Dict[str, Any]:
        statuses = self.get_statuses()
        return collect_health_report(self._repo_root, statuses, selected_runtime_id=selected_runtime_id)

    def get_runtime_def(self, runtime_id: Optional[str]) -> Optional[RuntimeDef]:
        if not runtime_id:
            return None
        return RUNTIME_MAP.get(runtime_id)

    def merge_settings(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(self._settings_provider())
        if overrides:
            payload.update(overrides)
        return payload

    def build_launch_options(self, runtime_id: str, settings: Dict[str, Any]) -> LaunchOptions:
        return LaunchOptions(
            runtime_id=runtime_id,
            safe_mode=settings.get("safe_mode", False),
            cn_mirror=settings.get("cn_mirror", False),
            apply_proxy_to_trainer=settings.get("apply_proxy_to_trainer", False),
            http_proxy=str(settings.get("http_proxy", "") or ""),
            https_proxy=str(settings.get("https_proxy", "") or ""),
            all_proxy=str(settings.get("all_proxy", "") or ""),
            attention_policy=settings.get("attention_policy", "default"),
            host=settings.get("host", DEFAULT_HOST),
            port=settings.get("port", DEFAULT_PORT),
            listen=settings.get("listen", False),
            disable_tensorboard=settings.get("disable_tensorboard", False),
            disable_tageditor=settings.get("disable_tageditor", False),
            disable_auto_mirror=settings.get("disable_auto_mirror", False),
            dev_mode=settings.get("dev_mode", False),
        )

    def get_launch_preflight(
        self,
        runtime_id: Optional[str],
        settings_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        statuses = self.get_statuses()
        settings = self.merge_settings(settings_overrides)
        return collect_launch_preflight(self._repo_root, runtime_id, settings, statuses)

    def prepare_launch(
        self,
        runtime_id: Optional[str],
        settings_overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[PreparedLaunch]:
        runtime_def = self.get_runtime_def(runtime_id)
        if not runtime_id or runtime_def is None:
            return None
        statuses = self.get_statuses()
        status = statuses[runtime_id]
        settings = self.merge_settings(settings_overrides)
        options = self.build_launch_options(runtime_id, settings)
        preflight = collect_launch_preflight(self._repo_root, runtime_id, settings, statuses)
        return PreparedLaunch(
            repo_root=self._repo_root,
            runtime_id=runtime_id,
            runtime_def=runtime_def,
            statuses=statuses,
            status=status,
            settings=settings,
            options=options,
            preflight=preflight,
        )

    def prepare_install(
        self,
        runtime_id: Optional[str],
        *,
        cn_mirror: Optional[bool] = None,
    ) -> Optional[PreparedInstall]:
        runtime_def = self.get_runtime_def(runtime_id)
        if not runtime_id or runtime_def is None:
            return None
        statuses = self.get_statuses()
        status = statuses[runtime_id]
        resolved_cn_mirror = bool(self.merge_settings().get("cn_mirror", False) if cn_mirror is None else cn_mirror)
        return PreparedInstall(
            repo_root=self._repo_root,
            runtime_id=runtime_id,
            runtime_def=runtime_def,
            statuses=statuses,
            status=status,
            cn_mirror=resolved_cn_mirror,
            proxy_settings={
                "http_proxy": str(self.merge_settings().get("http_proxy", "") or ""),
                "https_proxy": str(self.merge_settings().get("https_proxy", "") or ""),
                "all_proxy": str(self.merge_settings().get("all_proxy", "") or ""),
            },
        )
