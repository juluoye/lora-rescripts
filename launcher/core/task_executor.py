"""Launcher task execution coordinator for launch/install/updater/stop flows."""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from launcher.config import RUNTIME_MAP
from launcher.core.api_result import error_result, ok_result
from launcher.core.dependency_cache import (
    clear_runtime_dependency_cache,
    prefetch_runtime_dependencies,
)
from launcher.core.runtime_initializer import initialize_runtime_environment
from launcher.core.task_history_store import TaskHistoryStore, TaskStateStore
from launcher.core.runtime_coordinator import RuntimeCoordinator
from launcher.core.task_plans import run_install_plan, run_launch_plan
from launcher.core.subprocess_utils import hidden_subprocess_kwargs
from launcher.core.task_state import (
    advance_task_state,
    begin_task_state,
    build_interrupted_task_result,
    build_idle_task_state,
    build_task_result,
    build_task_stage_event,
    finish_task_state,
    push_task_history,
)
from launcher.core.update_checker import run_updater
from launcher.i18n import get_language


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_TASK_LOG_LIMIT = 200
_TASK_LOG_PERSIST_INTERVAL = 10
_SIZE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}
_DOWNLOAD_PROGRESS_RE = re.compile(
    r"(?P<done>\d+(?:\.\d+)?)\s*/\s*(?P<total>\d+(?:\.\d+)?)\s*(?P<unit>B|KB|MB|GB|TB)\b",
    re.IGNORECASE,
)
_SPEED_RE = re.compile(
    r"(?P<speed>\d+(?:\.\d+)?)\s*(?P<unit>B|KB|MB|GB|TB)\s*/s\b",
    re.IGNORECASE,
)
_ETA_RE = re.compile(
    r"(?:eta[:\s]+)?(?P<eta>\d{1,2}:\d{2}(?::\d{2})?)\b",
    re.IGNORECASE,
)
_INSTALL_SECTION_SPECS = (
    {
        "patterns": ("provisioning python dev files",),
        "key": "dev_files",
        "phase": "preparing",
        "label_zh": "准备 Python 开发文件",
        "label_en": "Preparing Python dev files",
        "start": 4,
        "end": 10,
    },
    {
        "patterns": ("upgrading pip tooling",),
        "key": "pip_tooling",
        "phase": "install",
        "label_zh": "升级 pip 工具链",
        "label_en": "Upgrading pip tooling",
        "start": 10,
        "end": 16,
    },
    {
        "patterns": ("installing pytorch and torchvision", "installing pytorch stack"),
        "key": "torch_stack",
        "phase": "download",
        "label_zh": "安装 PyTorch 与 TorchVision",
        "label_en": "Installing PyTorch and TorchVision",
        "start": 16,
        "end": 58,
    },
    {
        "patterns": ("installing xformers",),
        "key": "xformers",
        "phase": "download",
        "label_zh": "安装 xformers",
        "label_en": "Installing xformers",
        "start": 58,
        "end": 70,
    },
    {
        "patterns": ("installing project dependencies",),
        "key": "requirements",
        "phase": "install",
        "label_zh": "安装项目依赖",
        "label_en": "Installing project dependencies",
        "start": 70,
        "end": 84,
    },
    {
        "patterns": ("installing triton runtime",),
        "key": "triton_runtime",
        "phase": "install",
        "label_zh": "安装 Triton 运行时",
        "label_en": "Installing Triton runtime",
        "start": 84,
        "end": 91,
    },
    {
        "patterns": ("re-enabling pkg_resources compatibility",),
        "key": "setuptools_compat",
        "phase": "finalizing",
        "label_zh": "恢复兼容组件",
        "label_en": "Restoring compatibility components",
        "start": 91,
        "end": 94,
    },
    {
        "patterns": ("verifying triton runtime", "verifying ", "checking runtime", "runtime verification"),
        "key": "verification",
        "phase": "finalizing",
        "label_zh": "校验运行时",
        "label_en": "Verifying runtime",
        "start": 94,
        "end": 98,
    },
    {
        "patterns": ("installing flashattention", "downloading flashattention wheel", "building flashattention"),
        "key": "flashattention",
        "phase": "compile",
        "label_zh": "安装 FlashAttention",
        "label_en": "Installing FlashAttention",
        "start": 86,
        "end": 96,
    },
    {
        "patterns": ("installing sageattention", "building sageattention", "downloading sageattention"),
        "key": "sageattention",
        "phase": "compile",
        "label_zh": "安装 SageAttention",
        "label_en": "Installing SageAttention",
        "start": 86,
        "end": 96,
    },
    {
        "patterns": ("installing spargeattn2", "building spargeattn2", "spas_sage_attn", "local wheel"),
        "key": "spargeattn2",
        "phase": "compile",
        "label_zh": "安装 SpargeAttn2",
        "label_en": "Installing SpargeAttn2",
        "start": 84,
        "end": 96,
    },
)
_INSTALL_SECTION_SPEC_BY_KEY = {str(spec["key"]): spec for spec in _INSTALL_SECTION_SPECS}


def _size_to_bytes(value: str, unit: str) -> Optional[int]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    multiplier = _SIZE_UNITS.get(str(unit or "").upper())
    if multiplier is None:
        return None
    return max(0, int(numeric * multiplier))


def _eta_to_seconds(value: str) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        minutes, seconds = numbers
        return max(0, minutes * 60 + seconds)
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        return max(0, hours * 3600 + minutes * 60 + seconds)
    return None

_RUNTIME_BOOTSTRAP_SITE_PACKAGE_PATTERNS = (
    "pip",
    "pip-*.dist-info",
    "setuptools",
    "setuptools-*.dist-info",
    "wheel",
    "wheel-*.dist-info",
    "_distutils_hack",
    "pkg_resources",
    "distutils-precedence.pth",
)

_RUNTIME_BOOTSTRAP_SCRIPT_PATTERNS = (
    "pip.exe",
    "pip*.exe",
)


def _matches_any_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    normalized = name.lower()
    return any(fnmatch.fnmatchcase(normalized, pattern.lower()) for pattern in patterns)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _clear_directory_contents(directory: Path, *, keep_patterns: tuple[str, ...] = ()) -> Dict[str, int]:
    removed = 0
    kept = 0
    if not directory.exists():
        return {"removed": removed, "kept": kept}

    for item in directory.iterdir():
        if keep_patterns and _matches_any_pattern(item.name, keep_patterns):
            kept += 1
            continue
        _remove_path(item)
        removed += 1

    return {"removed": removed, "kept": kept}


def _soft_uninstall_runtime_environment(env_dir: Path) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "env_dir": str(env_dir),
        "mode": "dependency_only",
        "removed_markers": 0,
        "removed_root_caches": 0,
        "site_packages_removed": 0,
        "site_packages_kept": 0,
        "scripts_removed": 0,
        "scripts_kept": 0,
    }

    deps_marker = env_dir / ".deps_installed"
    if deps_marker.exists():
        deps_marker.unlink()
        details["removed_markers"] = 1

    for cache_dir in (env_dir / "__pycache__", env_dir / "Lib" / "__pycache__", env_dir / "Scripts" / "__pycache__"):
        if cache_dir.exists():
            _remove_path(cache_dir)
            details["removed_root_caches"] += 1

    site_packages_result = _clear_directory_contents(
        env_dir / "Lib" / "site-packages",
        keep_patterns=_RUNTIME_BOOTSTRAP_SITE_PACKAGE_PATTERNS,
    )
    details["site_packages_removed"] = site_packages_result["removed"]
    details["site_packages_kept"] = site_packages_result["kept"]

    scripts_result = _clear_directory_contents(
        env_dir / "Scripts",
        keep_patterns=_RUNTIME_BOOTSTRAP_SCRIPT_PATTERNS,
    )
    details["scripts_removed"] = scripts_result["removed"]
    details["scripts_kept"] = scripts_result["kept"]
    details["python_preserved"] = bool((env_dir / "python.exe").exists())
    return details


def _collect_runtime_env_dirs(repo_root: Path, runtime_id: str, primary_env_dir: Optional[Path]) -> list[Path]:
    runtime_def = RUNTIME_MAP.get(runtime_id)
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add_candidate(path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            normalized = str(path.resolve(strict=False))
        except Exception:
            normalized = str(path)
        key = normalized.lower()
        if key in seen or not path.exists():
            return
        seen.add(key)
        candidates.append(path)

    _add_candidate(primary_env_dir)

    if runtime_def is None:
        return candidates

    env_root = repo_root / "env"
    for dir_name in runtime_def.env_dir_names:
        _add_candidate(env_root / dir_name)
        _add_candidate(repo_root / dir_name)

    return candidates


def _soft_uninstall_runtime_environment_group(repo_root: Path, runtime_id: str, primary_env_dir: Path) -> Dict[str, Any]:
    env_dirs = _collect_runtime_env_dirs(repo_root, runtime_id, primary_env_dir)
    if not env_dirs:
        return {
            "runtime_id": runtime_id,
            "env_dir": str(primary_env_dir),
            "env_dirs": [],
            "mode": "dependency_only",
            "removed_markers": 0,
            "removed_root_caches": 0,
            "site_packages_removed": 0,
            "site_packages_kept": 0,
            "scripts_removed": 0,
            "scripts_kept": 0,
            "python_preserved": False,
            "env_count": 0,
        }

    details_list = [_soft_uninstall_runtime_environment(env_dir) for env_dir in env_dirs]
    return {
        "runtime_id": runtime_id,
        "env_dir": str(env_dirs[0]),
        "env_dirs": [str(env_dir) for env_dir in env_dirs],
        "mode": "dependency_only",
        "removed_markers": sum(int(item.get("removed_markers", 0)) for item in details_list),
        "removed_root_caches": sum(int(item.get("removed_root_caches", 0)) for item in details_list),
        "site_packages_removed": sum(int(item.get("site_packages_removed", 0)) for item in details_list),
        "site_packages_kept": sum(int(item.get("site_packages_kept", 0)) for item in details_list),
        "scripts_removed": sum(int(item.get("scripts_removed", 0)) for item in details_list),
        "scripts_kept": sum(int(item.get("scripts_kept", 0)) for item in details_list),
        "python_preserved": all(bool(item.get("python_preserved", False)) for item in details_list),
        "env_count": len(env_dirs),
    }


def _read_stream_chunks(
    stream,
    on_line: Callable[[str, bool], None],
    *,
    encoding: str = "utf-8",
) -> None:
    if stream is None:
        return

    try:
        fd = stream.fileno()
    except Exception:
        return

    def _decode(raw: bytes) -> str:
        try:
            return raw.decode(encoding, errors="replace")
        except Exception:
            return raw.decode("utf-8", errors="replace")

    buf = b""
    while True:
        try:
            chunk = os.read(fd, 8192)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk

        while True:
            cr_idx = buf.find(b"\r")
            lf_idx = buf.find(b"\n")
            if cr_idx == -1 and lf_idx == -1:
                break

            if cr_idx == -1:
                idx = lf_idx
            elif lf_idx == -1:
                idx = cr_idx
            else:
                idx = min(cr_idx, lf_idx)

            is_progress = buf[idx : idx + 1] == b"\r"
            delimiter_length = 1
            if is_progress and idx + 1 < len(buf) and buf[idx + 1 : idx + 2] == b"\n":
                is_progress = False
                delimiter_length = 2

            raw_line = buf[:idx]
            buf = buf[idx + delimiter_length :]

            line = _decode(raw_line).rstrip()
            if line:
                on_line(line, is_progress)

    if buf:
        line = _decode(buf).rstrip()
        if line:
            on_line(line, False)


class LauncherTaskExecutor:
    """Coordinates long-running launcher tasks and their stage events."""

    def __init__(
        self,
        repo_root: Path,
        config_dir: Path,
        emit_callback: Callable[[str, Any], None],
        settings_provider: Callable[[], Dict[str, Any]],
        runtime_coordinator: RuntimeCoordinator,
    ) -> None:
        self._repo_root = repo_root
        self._history_store = TaskHistoryStore(config_dir)
        self._state_store = TaskStateStore(config_dir)
        self._emit = emit_callback
        self._settings_provider = settings_provider
        self._runtime_coordinator = runtime_coordinator
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._installing = False
        self._task_state: Dict[str, Any] = build_idle_task_state()
        self._task_history: list[Dict[str, Any]] = self._history_store.load()
        self._task_stage_history: list[Dict[str, Any]] = []
        self._task_command_history: list[Dict[str, Any]] = []
        self._task_log_lines: list[str] = []
        self._task_log_dirty_count = 0
        self._last_task_log_was_progress = False
        self._recover_interrupted_task_if_needed()

    def _wait_for_runtime_detection(
        self,
        runtime_id: str,
        *,
        require_installed: bool,
        timeout_seconds: float = 6.0,
        poll_interval_seconds: float = 0.4,
    ) -> None:
        """Wait briefly for runtime filesystem markers to become visible to detector logic."""
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() < deadline:
            status = self._runtime_coordinator.get_statuses(force_refresh=True).get(runtime_id)
            if not status:
                time.sleep(poll_interval_seconds)
                continue
            if require_installed:
                if status.installed:
                    return
            elif status.python_exists:
                return
            time.sleep(poll_interval_seconds)

    def get_task_state(self) -> Dict[str, Any]:
        return dict(self._task_state)

    def get_task_history(self) -> list[Dict[str, Any]]:
        return [dict(item) for item in self._task_history]

    def clear_task_history(self) -> Dict[str, Any]:
        self._task_history = []
        self._history_store.clear()
        self._emit("task_history_cleared", {"ok": True})
        return ok_result("task_history.cleared")

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def is_installing(self) -> bool:
        return self._installing

    def run_updater(self) -> Dict[str, Any]:
        self._begin_task(
            "updater",
            "updater.request_received",
            "已收到更新请求",
            "Updater request received",
        )
        if self._process is not None and self._process.poll() is None:
            message = "Stop the running trainer backend before starting the updater."
            self._finish_task(
                success=False,
                stage_code="updater.blocked_trainer_running",
                stage_label_zh="更新器被训练进程阻止",
                stage_label_en="Updater blocked by running trainer",
                code="updater.blocked_trainer_running",
                error=message,
            )
            return error_result("updater.blocked_trainer_running", message)
        if self._installing:
            message = "Wait for the current runtime installation to finish before updating."
            self._finish_task(
                success=False,
                stage_code="updater.blocked_install_running",
                stage_label_zh="更新器被安装任务阻止",
                stage_label_en="Updater blocked by running install task",
                code="updater.blocked_install_running",
                error=message,
            )
            return error_result("updater.blocked_install_running", message)

        self._advance_task(
            "updater.starting_process",
            "正在启动更新器",
            "Starting updater",
        )
        result = run_updater(
            self._repo_root,
            use_cn_mirror=bool(self._settings_provider().get("cn_mirror", False)),
            proxy_settings=self._settings_provider(),
        )
        if not result.get("ok"):
            self._finish_task(
                success=False,
                stage_code="updater.start_failed",
                stage_label_zh="更新器启动失败",
                stage_label_en="Updater start failed",
                code=str(result.get("code") or "updater.start_failed"),
                error=result.get("error"),
                details=result.get("details"),
            )
            return error_result(
                str(result.get("code") or "updater.start_failed"),
                result.get("error") or "Failed to start updater.",
                details=result.get("details"),
            )

        self._finish_task(
            success=True,
            stage_code="updater.started",
            stage_label_zh="更新器已启动",
            stage_label_en="Updater started",
            result_code=str(result.get("result_code") or "updater.started"),
            details=result.get("details"),
        )
        return ok_result(
            str(result.get("result_code") or "updater.started"),
            details=result.get("details"),
        )

    def launch(self, runtime_id: str) -> Dict[str, Any]:
        self._begin_task(
            "launch",
            "launch.request_received",
            "已收到启动请求",
            "Launch request received",
            runtime_id=runtime_id,
        )
        if self._process is not None and self._process.poll() is None:
            self._finish_task(
                success=False,
                stage_code="trainer.already_running",
                stage_label_zh="训练器已在运行",
                stage_label_en="Trainer already running",
                code="trainer.already_running",
                error="Already running",
                details={"runtime_id": runtime_id},
            )
            return error_result("trainer.already_running", "Already running")

        if runtime_id not in RUNTIME_MAP:
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error="Unknown runtime",
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", "Unknown runtime", details={"runtime_id": runtime_id})

        self._advance_task(
            "launch.validating_runtime",
            "正在校验运行时",
            "Validating runtime",
            details={"runtime_id": runtime_id},
        )
        prepared = self._runtime_coordinator.prepare_launch(runtime_id)
        if prepared is None:
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error="Unknown runtime",
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", "Unknown runtime", details={"runtime_id": runtime_id})
        status = prepared.status
        if not status or not status.installed or not status.python_path:
            self._finish_task(
                success=False,
                stage_code="runtime.not_installed",
                stage_label_zh="运行时未安装",
                stage_label_en="Runtime not installed",
                code="runtime.not_installed",
                error="Runtime not installed",
                details={"runtime_id": runtime_id},
            )
            return error_result(
                "runtime.not_installed",
                "Runtime not installed",
                details={"runtime_id": runtime_id},
            )

        self._advance_task(
            "launch.running_preflight",
            "正在执行启动前检查",
            "Running launch preflight",
            details={"runtime_id": runtime_id},
        )
        preflight = prepared.preflight
        if not preflight.get("ready", False):
            lang = get_language()
            errors = [
                issue
                for issue in preflight.get("issues", [])
                if issue.get("severity") == "error"
            ]
            if errors:
                first = errors[0]
                message = first.get("message_en") if lang == "en" else first.get("message_zh")
                details = {
                    "runtime_id": runtime_id,
                    "blocking_issue_code": first.get("code"),
                    "action_page": first.get("action_page"),
                }
                self._finish_task(
                    success=False,
                    stage_code="launch.preflight_blocked",
                    stage_label_zh="启动前检查阻止启动",
                    stage_label_en="Launch blocked by preflight",
                    code="launch.preflight_blocked",
                    error=message or "Launch preflight failed.",
                    details=details,
                )
                return error_result(
                    "launch.preflight_blocked",
                    message or "Launch preflight failed.",
                    details=details,
                    preflight=preflight,
                )

        try:
            self._advance_task(
                "launch.building_plan",
                "正在构建启动计划",
                "Building launch plan",
                details={"runtime_id": runtime_id},
            )
            plan = prepared.build_plan()
            if plan is None:
                raise RuntimeError("Launch plan could not be built because python_path is missing.")
            self._advance_task(
                "launch.spawning_process",
                "正在启动训练进程",
                "Spawning trainer process",
                details={"runtime_id": runtime_id, "command_count": len(plan.commands)},
            )
            if plan.commands:
                self._record_command_started(
                    command=plan.commands[0],
                    index=1,
                    total=len(plan.commands),
                    command_kind="launch",
                )
            self._process = run_launch_plan(plan)
            self._record_launch_process_started(self._process.pid if self._process else None)
            self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
            self._reader_thread.start()
            self._advance_task(
                "launch.process_running",
                "训练进程已启动",
                "Trainer process is running",
                details={"runtime_id": runtime_id, "pid": self._process.pid if self._process else None},
            )
            return ok_result("trainer.launch_started", runtime_id=runtime_id)
        except Exception as exc:
            if "plan" in locals() and plan and plan.commands:
                self._record_command_finished(
                    command=plan.commands[0],
                    index=1,
                    total=len(plan.commands),
                    command_kind="launch",
                    success=False,
                    error=str(exc),
                )
            self._finish_task(
                success=False,
                stage_code="launch.spawn_failed",
                stage_label_zh="训练进程启动失败",
                stage_label_en="Trainer process spawn failed",
                code="launch.spawn_failed",
                error=str(exc),
                details={"runtime_id": runtime_id},
            )
            return error_result(
                "launch.spawn_failed",
                str(exc),
                details={"runtime_id": runtime_id},
            )

    def stop(self) -> Dict[str, Any]:
        if self._process:
            self._begin_task(
                "stop",
                "trainer.stop_request_received",
                "已收到停止请求",
                "Stop request received",
            )
            self._advance_task(
                "trainer.stop_signal_sent",
                "已发送停止信号",
                "Stop signal sent",
                details={"pid": self._process.pid},
            )
            thread = threading.Thread(target=self._terminate_process, daemon=True)
            thread.start()
            return ok_result("trainer.stop_signal_sent")

        self._begin_task(
            "stop",
            "trainer.stop_request_received",
            "已收到停止请求",
            "Stop request received",
        )
        self._finish_task(
            success=True,
            stage_code="trainer.stop_no_process",
            stage_label_zh="当前没有可停止的进程",
            stage_label_en="No running process to stop",
            result_code="trainer.stop_no_process",
        )
        return ok_result("trainer.stop_no_process")

    def kill(self) -> Dict[str, Any]:
        if self._process:
            self._begin_task(
                "kill",
                "trainer.kill_request_received",
                "已收到强制结束请求",
                "Kill request received",
            )
            self._advance_task(
                "trainer.kill_signal_sent",
                "已发送强制结束信号",
                "Kill signal sent",
                details={"pid": self._process.pid},
            )
            thread = threading.Thread(target=self._force_kill_process, daemon=True)
            thread.start()
            return ok_result("trainer.kill_signal_sent")

        self._begin_task(
            "kill",
            "trainer.kill_request_received",
            "已收到强制结束请求",
            "Kill request received",
        )
        self._finish_task(
            success=True,
            stage_code="trainer.kill_no_process",
            stage_label_zh="当前没有可结束的进程",
            stage_label_en="No running process to kill",
            result_code="trainer.kill_no_process",
        )
        return ok_result("trainer.kill_no_process")

    def initialize_runtime(self, runtime_id: str) -> Dict[str, Any]:
        lang = get_language()
        self._begin_task(
            "initialize",
            "runtime_initialize.request_received",
            "已收到初始化请求",
            "Initialization request received",
            runtime_id=runtime_id,
        )
        if self._installing:
            message = "已有安装或初始化任务正在进行中。" if lang == "zh" else "Another install or initialization task is already in progress."
            self._finish_task(
                success=False,
                stage_code="runtime_initialize.already_running",
                stage_label_zh="已有初始化任务在运行",
                stage_label_en="Initialization task already running",
                code="runtime_initialize.already_running",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime_initialize.already_running", message)
        if runtime_id not in RUNTIME_MAP:
            message = "未知的运行时。" if lang == "zh" else "Unknown runtime."
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", message, details={"runtime_id": runtime_id})

        prepared = self._runtime_coordinator.prepare_install(
            runtime_id,
            cn_mirror=bool(self._settings_provider().get("cn_mirror", False)),
        )
        if prepared is None:
            message = "未知的运行时。" if lang == "zh" else "Unknown runtime."
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", message, details={"runtime_id": runtime_id})

        if prepared.status.installed:
            message = "该运行时已经安装完成，无需再次初始化。" if lang == "zh" else "This runtime is already installed and does not need initialization."
            self._finish_task(
                success=True,
                stage_code="runtime_initialize.already_prepared",
                stage_label_zh="运行时已可用",
                stage_label_en="Runtime is already ready",
                result_code="runtime_initialize.already_prepared",
                details={"runtime_id": runtime_id},
            )
            return ok_result("runtime_initialize.already_prepared", runtime_id=runtime_id, message=message)

        self._installing = True

        def _run():
            success = False
            final_code = "runtime_initialize.failed"
            error_message = None
            try:
                self._advance_task(
                    "runtime_initialize.preparing_runtime",
                    "正在准备运行时目录",
                    "Preparing runtime directory",
                    details={"runtime_id": runtime_id},
                )
                initialize_runtime_environment(
                    prepared.runtime_def,
                    repo_root=self._repo_root,
                    statuses=prepared.statuses,
                    cn_mirror=prepared.cn_mirror,
                    log_callback=lambda line: self._append_task_log_line(line, event_name="install_log"),
                )
                success = True
                final_code = "runtime_initialize.completed"
            except Exception as exc:
                final_code = "runtime_initialize.execution_failed"
                error_message = str(exc)
                self._append_task_log_line(f"[Launcher] Runtime initialization failed: {exc}", event_name="install_log")
            finally:
                self._installing = False
                payload: Dict[str, Any] = {
                    "runtime_id": runtime_id,
                    "success": success,
                    "action": "initialize",
                }
                self._runtime_coordinator.invalidate_status_cache()
                if success:
                    self._wait_for_runtime_detection(runtime_id, require_installed=False)
                    self._finish_task(
                        success=True,
                        stage_code="runtime_initialize.completed",
                        stage_label_zh="运行时初始化完成",
                        stage_label_en="Runtime initialization completed",
                        result_code=final_code,
                        details={"runtime_id": runtime_id},
                    )
                    payload["result_code"] = final_code
                else:
                    self._finish_task(
                        success=False,
                        stage_code=final_code,
                        stage_label_zh="运行时初始化失败",
                        stage_label_en="Runtime initialization failed",
                        code=final_code,
                        error=error_message,
                        details={"runtime_id": runtime_id},
                    )
                    payload["code"] = final_code
                    if error_message:
                        payload["error"] = error_message
                self._emit("install_done", payload)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return ok_result("runtime_initialize.started", runtime_id=runtime_id)

    def install_runtime(self, runtime_id: str) -> Dict[str, Any]:
        lang = get_language()
        self._begin_task(
            "install",
            "runtime_install.request_received",
            "已收到安装请求",
            "Install request received",
            runtime_id=runtime_id,
        )
        if self._installing:
            message = "已有安装任务正在进行中。" if lang == "zh" else "Another installation is already in progress."
            self._finish_task(
                success=False,
                stage_code="runtime_install.already_running",
                stage_label_zh="已有安装任务在运行",
                stage_label_en="Install task already running",
                code="runtime_install.already_running",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime_install.already_running", message)
        if runtime_id not in RUNTIME_MAP:
            message = "未知的运行时。" if lang == "zh" else "Unknown runtime."
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", message, details={"runtime_id": runtime_id})

        self._advance_task(
            "runtime_install.validating_runtime",
            "正在校验安装前置条件",
            "Validating install prerequisites",
            details={"runtime_id": runtime_id},
        )
        if shutil.which("powershell.exe") is None:
            message = (
                "系统中未找到 PowerShell，无法启动运行时安装器。"
                if lang == "zh"
                else "PowerShell was not found on this system, so the runtime installer cannot start."
            )
            self._finish_task(
                success=False,
                stage_code="runtime_install.powershell_missing",
                stage_label_zh="系统缺少 PowerShell",
                stage_label_en="PowerShell is missing",
                code="runtime_install.powershell_missing",
                error=message,
            )
            return error_result("runtime_install.powershell_missing", message)

        runtime_def = RUNTIME_MAP[runtime_id]
        missing_scripts = [
            script_name
            for script_name in runtime_def.install_scripts
            if not (self._repo_root / script_name).exists()
        ]
        if missing_scripts:
            message = (
                f"缺少安装脚本：{', '.join(missing_scripts)}"
                if lang == "zh"
                else f"Missing install script(s): {', '.join(missing_scripts)}"
            )
            self._finish_task(
                success=False,
                stage_code="runtime_install.scripts_missing",
                stage_label_zh="安装脚本缺失",
                stage_label_en="Install scripts missing",
                code="runtime_install.scripts_missing",
                error=message,
                details={"missing_scripts": missing_scripts},
            )
            return error_result(
                "runtime_install.scripts_missing",
                message,
                details={"missing_scripts": missing_scripts},
            )

        prepared = self._runtime_coordinator.prepare_install(
            runtime_id,
            cn_mirror=bool(self._settings_provider().get("cn_mirror", False)),
        )
        if prepared is None:
            message = "未知的运行时。" if lang == "zh" else "Unknown runtime."
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", message, details={"runtime_id": runtime_id})
        status = prepared.status
        if not status or not status.python_exists or not status.python_path:
            preferred_dirs = [f".\\env\\{name}" for name in prepared.runtime_def.env_dir_names]
            legacy_dirs = [f".\\{name}" for name in prepared.runtime_def.env_dir_names]
            suggested_locations = ", ".join(preferred_dirs + legacy_dirs)
            message = (
                (
                    "这个运行时在安装前需要先准备好项目内本地 Python 环境。"
                    f"请先放到以下任一位置：{suggested_locations}"
                )
                if lang == "zh"
                else (
                    "This runtime needs a prepared project-local Python environment before installation. "
                    f"Place it in one of these locations first: {suggested_locations}"
                )
            )
            details = {
                "preferred_dirs": preferred_dirs,
                "legacy_dirs": legacy_dirs,
                "runtime_id": runtime_id,
            }
            self._finish_task(
                success=False,
                stage_code="runtime_install.python_missing",
                stage_label_zh="缺少运行时 Python 环境",
                stage_label_en="Runtime Python environment missing",
                code="runtime_install.python_missing",
                error=message,
                details=details,
            )
            return error_result(
                "runtime_install.python_missing",
                message,
                details=details,
            )
        if not status.integrity_ok:
            message = (
                (
                    f"检测到 {runtime_id} 的 Python 目录存在，但环境骨架不完整，当前不能直接安装依赖。"
                    f"{status.integrity_message_zh or ''}"
                )
                if lang == "zh"
                else (
                    f"The {runtime_id} Python directory exists, but the runtime skeleton is incomplete and dependencies cannot be installed yet. "
                    f"{status.integrity_message_en or ''}"
                )
            ).strip()
            self._finish_task(
                success=False,
                stage_code="runtime_install.runtime_broken",
                stage_label_zh="运行时环境损坏",
                stage_label_en="Runtime environment is broken",
                code="runtime_install.runtime_broken",
                error=message,
                details={"runtime_id": runtime_id, "issue_code": status.integrity_issue_code},
            )
            return error_result(
                "runtime_install.runtime_broken",
                message,
                details={"runtime_id": runtime_id, "issue_code": status.integrity_issue_code},
            )

        self._installing = True

        def _run():
            success = False
            final_code = "runtime_install.failed"
            error_message = None
            try:
                self._advance_task(
                    "runtime_install.building_plan",
                    "正在构建安装计划",
                    "Building install plan",
                    details={"runtime_id": runtime_id},
                )
                plan = prepared.build_plan()
                self._advance_task(
                    "runtime_install.executing_scripts",
                    "正在执行安装脚本",
                    "Executing install scripts",
                    details={"runtime_id": runtime_id, "script_count": len(plan.commands)},
                )
                success = run_install_plan(
                    plan,
                    output_callback=lambda line, progress: self._handle_install_output_line(line, progress=progress),
                    stage_callback=lambda command, index, total: (
                        self._record_command_started(
                            command=command,
                            index=index,
                            total=total,
                            command_kind="install",
                        ),
                        self._advance_task(
                            "runtime_install.running_script",
                            f"正在执行安装脚本 {index}/{total}",
                            f"Running install script {index}/{total}",
                            details={
                                "runtime_id": runtime_id,
                                "script_index": index,
                                "script_total": total,
                                "progress_phase": "preparing",
                                "progress_phase_label_zh": "准备脚本",
                                "progress_phase_label_en": "Preparing script",
                                "command_label_zh": command.label_zh,
                                "command_label_en": command.label_en,
                                "progress_item_label_zh": command.label_zh,
                                "progress_item_label_en": command.label_en,
                                "command_preview": command.to_public_dict().get("command_preview"),
                            },
                        ),
                    ),
                    result_callback=lambda command, index, total, command_success: self._record_command_finished(
                        command=command,
                        index=index,
                        total=total,
                        command_kind="install",
                        success=command_success,
                    ),
                )
                if success:
                    final_code = "runtime_install.completed"
            except Exception as exc:
                final_code = "runtime_install.execution_failed"
                error_message = str(exc)
                self._append_task_log_line(f"[Launcher] Install plan failed: {exc}", event_name="install_log")
            finally:
                self._installing = False
                payload: Dict[str, Any] = {
                    "runtime_id": runtime_id,
                    "success": success,
                    "action": "install",
                }
                self._runtime_coordinator.invalidate_status_cache()
                if success:
                    self._wait_for_runtime_detection(runtime_id, require_installed=True)
                    self._finish_task(
                        success=True,
                        stage_code="runtime_install.completed",
                        stage_label_zh="安装已完成",
                        stage_label_en="Installation completed",
                        result_code=final_code,
                        details={"runtime_id": runtime_id},
                    )
                    payload["result_code"] = final_code
                else:
                    self._finish_task(
                        success=False,
                        stage_code=final_code,
                        stage_label_zh="安装失败",
                        stage_label_en="Installation failed",
                        code=final_code,
                        error=error_message,
                        details={"runtime_id": runtime_id},
                    )
                    payload["code"] = final_code
                    if error_message:
                        payload["error"] = error_message
                self._emit("install_done", payload)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return ok_result("runtime_install.started", runtime_id=runtime_id)

    def prefetch_runtime_dependencies(self, runtime_id: str) -> Dict[str, Any]:
        lang = get_language()
        self._begin_task(
            "dependency_cache",
            "dependency_cache.request_received",
            "已收到依赖缓存请求",
            "Dependency cache request received",
            runtime_id=runtime_id,
        )
        if self._installing:
            message = "已有运行时任务正在进行中。" if lang == "zh" else "Another runtime task is already in progress."
            self._finish_task(
                success=False,
                stage_code="dependency_cache.already_running",
                stage_label_zh="已有运行时任务在运行",
                stage_label_en="Another runtime task is already running",
                code="dependency_cache.already_running",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("dependency_cache.already_running", message)
        if runtime_id not in RUNTIME_MAP:
            message = "未知的运行时。" if lang == "zh" else "Unknown runtime."
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", message, details={"runtime_id": runtime_id})

        prepared = self._runtime_coordinator.prepare_install(
            runtime_id,
            cn_mirror=bool(self._settings_provider().get("cn_mirror", False)),
        )
        if prepared is None:
            message = "未知的运行时。" if lang == "zh" else "Unknown runtime."
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", message, details={"runtime_id": runtime_id})

        status = prepared.status
        if not status.python_exists or not status.python_path:
            message = (
                "请先初始化该运行时的本地 Python 环境，再开始缓存依赖。"
                if lang == "zh"
                else "Initialize this runtime's local Python environment before prefetching dependencies."
            )
            self._finish_task(
                success=False,
                stage_code="dependency_cache.python_missing",
                stage_label_zh="缺少运行时 Python 环境",
                stage_label_en="Runtime Python environment missing",
                code="dependency_cache.python_missing",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("dependency_cache.python_missing", message, details={"runtime_id": runtime_id})
        if not status.integrity_ok or not status.bootstrap_ready:
            message = (
                "当前运行时环境还不完整，请先完成初始化或修复损坏的 Python 骨架。"
                if lang == "zh"
                else "This runtime is not ready yet. Finish initialization or repair the broken Python skeleton first."
            )
            self._finish_task(
                success=False,
                stage_code="dependency_cache.runtime_not_ready",
                stage_label_zh="运行时尚未就绪",
                stage_label_en="Runtime is not ready",
                code="dependency_cache.runtime_not_ready",
                error=message,
                details={"runtime_id": runtime_id, "issue_code": status.integrity_issue_code},
            )
            return error_result("dependency_cache.runtime_not_ready", message, details={"runtime_id": runtime_id})

        self._installing = True

        def _run() -> None:
            success = False
            final_code = "dependency_cache.failed"
            error_message = None
            final_details: Dict[str, Any] = {"runtime_id": runtime_id}
            try:
                self._advance_task(
                    "dependency_cache.preparing",
                    "正在准备依赖缓存目录",
                    "Preparing dependency cache directory",
                    details={"runtime_id": runtime_id},
                )
                final_state = prefetch_runtime_dependencies(
                    self._repo_root,
                    runtime_id,
                    status.python_path,
                    cn_mirror=prepared.cn_mirror,
                    proxy_settings=prepared.proxy_settings,
                    log_callback=lambda line: self._append_task_log_line(line, event_name="install_log"),
                    progress_callback=lambda payload: self._advance_task(
                        "dependency_cache.downloading",
                        f"正在缓存依赖 {payload.get('completed_items', 0) + (0 if payload.get('state') == 'succeeded' else 1)}/{payload.get('total_items', 0)}",
                        f"Caching dependencies {payload.get('completed_items', 0) + (0 if payload.get('state') == 'succeeded' else 1)}/{payload.get('total_items', 0)}",
                        details=payload,
                    ),
                )
                success = True
                final_code = "dependency_cache.completed"
                final_details = {
                    "runtime_id": runtime_id,
                    "cache_dir": final_state.get("cache_dir"),
                    "cached_items": final_state.get("cached_items"),
                    "total_items": final_state.get("total_items"),
                    "total_bytes": final_state.get("total_bytes"),
                }
            except Exception as exc:
                final_code = "dependency_cache.execution_failed"
                error_message = str(exc)
                self._append_task_log_line(f"[Launcher] Dependency cache failed: {exc}", event_name="install_log")
            finally:
                self._installing = False
                payload: Dict[str, Any] = {
                    "runtime_id": runtime_id,
                    "success": success,
                    "action": "cache",
                    "details": final_details,
                }
                if success:
                    self._finish_task(
                        success=True,
                        stage_code="dependency_cache.completed",
                        stage_label_zh="依赖缓存完成",
                        stage_label_en="Dependency cache completed",
                        result_code=final_code,
                        details=final_details,
                    )
                    payload["result_code"] = final_code
                else:
                    self._finish_task(
                        success=False,
                        stage_code=final_code,
                        stage_label_zh="依赖缓存失败",
                        stage_label_en="Dependency cache failed",
                        code=final_code,
                        error=error_message,
                        details=final_details,
                    )
                    payload["code"] = final_code
                    if error_message:
                        payload["error"] = error_message
                self._emit("install_done", payload)

        threading.Thread(target=_run, daemon=True).start()
        return ok_result("dependency_cache.started", runtime_id=runtime_id)

    def clear_runtime_dependency_cache(self, runtime_id: str) -> Dict[str, Any]:
        lang = get_language()
        self._begin_task(
            "dependency_cache_clear",
            "dependency_cache_clear.request_received",
            "已收到清理依赖缓存请求",
            "Dependency cache clear request received",
            runtime_id=runtime_id,
        )
        if self._installing:
            message = "已有运行时任务正在进行中。" if lang == "zh" else "Another runtime task is already in progress."
            self._finish_task(
                success=False,
                stage_code="dependency_cache_clear.already_running",
                stage_label_zh="已有运行时任务在运行",
                stage_label_en="Another runtime task is already running",
                code="dependency_cache_clear.already_running",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("dependency_cache_clear.already_running", message)
        if runtime_id not in RUNTIME_MAP:
            message = "未知的运行时。" if lang == "zh" else "Unknown runtime."
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", message, details={"runtime_id": runtime_id})

        try:
            state = clear_runtime_dependency_cache(self._repo_root, runtime_id)
            self._finish_task(
                success=True,
                stage_code="dependency_cache_clear.completed",
                stage_label_zh="依赖缓存已清理",
                stage_label_en="Dependency cache cleared",
                result_code="dependency_cache_clear.completed",
                details={"runtime_id": runtime_id, "cache_dir": state.get("cache_dir")},
            )
            return ok_result(
                "dependency_cache_clear.completed",
                runtime_id=runtime_id,
                cache_state=state,
            )
        except Exception as exc:
            self._finish_task(
                success=False,
                stage_code="dependency_cache_clear.failed",
                stage_label_zh="依赖缓存清理失败",
                stage_label_en="Dependency cache clear failed",
                code="dependency_cache_clear.failed",
                error=str(exc),
                details={"runtime_id": runtime_id},
            )
            return error_result("dependency_cache_clear.failed", str(exc), details={"runtime_id": runtime_id})

    def uninstall_runtime(self, runtime_id: str) -> Dict[str, Any]:
        lang = get_language()
        self._begin_task(
            "uninstall",
            "runtime_uninstall.request_received",
            "已收到依赖卸载请求",
            "Dependency uninstall request received",
            runtime_id=runtime_id,
        )
        if self._installing:
            message = "已有安装、初始化或依赖卸载任务正在进行中。" if lang == "zh" else "Another install, initialization, or dependency uninstall task is already in progress."
            self._finish_task(
                success=False,
                stage_code="runtime_uninstall.already_running",
                stage_label_zh="已有运行时任务在运行",
                stage_label_en="Another runtime task is already running",
                code="runtime_uninstall.already_running",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime_uninstall.already_running", message)
        if self._process is not None and self._process.poll() is None:
            message = "请先停止当前训练进程，再卸载运行时。" if lang == "zh" else "Stop the running trainer process before uninstalling a runtime."
            self._finish_task(
                success=False,
                stage_code="runtime_uninstall.trainer_running",
                stage_label_zh="训练进程正在运行",
                stage_label_en="Trainer process is running",
                code="runtime_uninstall.trainer_running",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime_uninstall.trainer_running", message)
        if runtime_id not in RUNTIME_MAP:
            message = "未知的运行时。" if lang == "zh" else "Unknown runtime."
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", message, details={"runtime_id": runtime_id})

        prepared = self._runtime_coordinator.prepare_install(
            runtime_id,
            cn_mirror=bool(self._settings_provider().get("cn_mirror", False)),
        )
        if prepared is None:
            message = "未知的运行时。" if lang == "zh" else "Unknown runtime."
            self._finish_task(
                success=False,
                stage_code="runtime.unknown",
                stage_label_zh="未知运行时",
                stage_label_en="Unknown runtime",
                code="runtime.unknown",
                error=message,
                details={"runtime_id": runtime_id},
            )
            return error_result("runtime.unknown", message, details={"runtime_id": runtime_id})

        status = prepared.status
        env_dir = status.env_dir
        if env_dir is None or not env_dir.exists():
            message = "没有检测到可卸载的运行时目录。" if lang == "zh" else "No runtime directory was found to uninstall."
            self._finish_task(
                success=True,
                stage_code="runtime_uninstall.not_installed",
                stage_label_zh="运行时未安装",
                stage_label_en="Runtime is not installed",
                result_code="runtime_uninstall.not_installed",
                details={"runtime_id": runtime_id},
            )
            return ok_result("runtime_uninstall.not_installed", runtime_id=runtime_id, message=message)

        self._installing = True

        def _run():
            success = False
            final_code = "runtime_uninstall.failed"
            error_message = None
            final_details: Dict[str, Any] = {"runtime_id": runtime_id, "env_dir": str(env_dir), "mode": "dependency_only"}
            try:
                self._advance_task(
                    "runtime_uninstall.removing_dependencies",
                    "正在卸载运行时依赖",
                    "Removing runtime dependencies",
                    details={"runtime_id": runtime_id, "env_dir": str(env_dir)},
                )
                self._append_task_log_line(
                    f"[Launcher] Uninstalling runtime dependencies from: {env_dir}",
                    event_name="install_log",
                )
                final_details = _soft_uninstall_runtime_environment_group(self._repo_root, runtime_id, env_dir)
                success = True
                final_code = "runtime_uninstall.completed"
                self._append_task_log_line(
                    (
                        "[Launcher] Runtime dependencies removed while preserving the local Python skeleton: "
                        f"{', '.join(final_details.get('env_dirs', [str(env_dir)]))} "
                        f"(site-packages removed: {final_details.get('site_packages_removed', 0)}, "
                        f"scripts removed: {final_details.get('scripts_removed', 0)})"
                    ),
                    event_name="install_log",
                )
            except Exception as exc:
                final_code = "runtime_uninstall.execution_failed"
                error_message = str(exc)
                self._append_task_log_line(
                    f"[Launcher] Runtime dependency uninstall failed: {exc}",
                    event_name="install_log",
                )
            finally:
                self._installing = False
                payload: Dict[str, Any] = {
                    "runtime_id": runtime_id,
                    "success": success,
                    "action": "uninstall",
                    "details": final_details,
                }
                self._runtime_coordinator.invalidate_status_cache()
                if success:
                    self._wait_for_runtime_detection(runtime_id, require_installed=False)
                    self._finish_task(
                        success=True,
                        stage_code="runtime_uninstall.completed",
                        stage_label_zh="运行时依赖已卸载",
                        stage_label_en="Runtime dependencies removed",
                        result_code=final_code,
                        details=final_details,
                    )
                    payload["result_code"] = final_code
                else:
                    self._finish_task(
                        success=False,
                        stage_code=final_code,
                        stage_label_zh="运行时依赖卸载失败",
                        stage_label_en="Runtime dependency uninstall failed",
                        code=final_code,
                        error=error_message,
                        details=final_details,
                    )
                    payload["code"] = final_code
                    if error_message:
                        payload["error"] = error_message
                self._emit("install_done", payload)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return ok_result("runtime_uninstall.started", runtime_id=runtime_id)

    def _emit_task_snapshot(self, include_stage_event: bool = True) -> None:
        self._emit("task_state", dict(self._task_state))
        if include_stage_event and self._task_state.get("task_type") != "idle":
            event = build_task_stage_event(self._task_state)
            self._task_stage_history.append(event)
            if len(self._task_stage_history) > 64:
                self._task_stage_history = self._task_stage_history[-64:]
            self._emit("task_stage", event)

    def _active_task_snapshot(self) -> Dict[str, Any]:
        snapshot = dict(self._task_state)
        snapshot["stage_history"] = [dict(item) for item in self._task_stage_history]
        snapshot["command_records"] = [dict(item) for item in self._task_command_history]
        snapshot["log_lines"] = list(self._task_log_lines)
        return snapshot

    def _persist_active_task_state(self) -> None:
        if self._task_state.get("task_type") == "idle":
            return
        if self._task_state.get("state") not in {"pending", "running"}:
            return
        self._state_store.save(self._active_task_snapshot())

    def _find_command_record(self, command_preview: str, index: int) -> Optional[Dict[str, Any]]:
        for record in reversed(self._task_command_history):
            if record.get("command_preview") == command_preview and record.get("index") == index:
                return record
        return None

    def _record_command_started(
        self,
        *,
        command: Any,
        index: int,
        total: int,
        command_kind: str,
    ) -> None:
        public = command.to_public_dict()
        record = dict(public)
        record.update(
            {
                "index": index,
                "total": total,
                "command_kind": command_kind,
                "status": "running",
                "started_at": _now_iso(),
                "finished_at": None,
                "duration_ms": None,
                "exit_code": None,
                "pid": None,
                "error": None,
            }
        )
        self._task_command_history.append(record)
        if len(self._task_command_history) > 32:
            self._task_command_history = self._task_command_history[-32:]
        self._persist_active_task_state()

    def _record_command_finished(
        self,
        *,
        command: Any,
        index: int,
        success: bool,
        total: int = 1,
        command_kind: Optional[str] = None,
        exit_code: Optional[int] = None,
        pid: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        public = command.to_public_dict()
        record = self._find_command_record(public.get("command_preview", ""), index)
        if record is None:
            self._record_command_started(
                command=command,
                index=index,
                total=total,
                command_kind=command_kind or self._task_state.get("task_type", "task"),
            )
            record = self._find_command_record(public.get("command_preview", ""), index)
            if record is None:
                return
        record.update(
            {
                "status": "succeeded" if success else "failed",
                "finished_at": _now_iso(),
                "exit_code": exit_code,
                "pid": pid,
                "error": error,
            }
        )
        started_at = record.get("started_at")
        finished_at = record.get("finished_at")
        if isinstance(started_at, str) and isinstance(finished_at, str):
            try:
                started = datetime.fromisoformat(started_at)
                finished = datetime.fromisoformat(finished_at)
                record["duration_ms"] = max(0, int((finished - started).total_seconds() * 1000))
            except ValueError:
                record["duration_ms"] = None
        self._persist_active_task_state()

    def _record_launch_process_started(self, pid: Optional[int]) -> None:
        if not self._task_command_history:
            return
        self._task_command_history[-1].update({"pid": pid})
        self._persist_active_task_state()

    def _record_launch_process_exited(self, code: int) -> None:
        if not self._task_command_history:
            return
        self._task_command_history[-1].update(
            {
                "status": "succeeded" if code == 0 else "failed",
                "finished_at": _now_iso(),
                "exit_code": code,
                "error": None if code == 0 else f"Process exited with code {code}",
            }
        )
        started_at = self._task_command_history[-1].get("started_at")
        finished_at = self._task_command_history[-1].get("finished_at")
        if isinstance(started_at, str) and isinstance(finished_at, str):
            try:
                started = datetime.fromisoformat(started_at)
                finished = datetime.fromisoformat(finished_at)
                self._task_command_history[-1]["duration_ms"] = max(0, int((finished - started).total_seconds() * 1000))
            except ValueError:
                self._task_command_history[-1]["duration_ms"] = None

    def _append_task_log_line(self, line: str, *, event_name: str, progress: bool = False) -> None:
        self._emit(event_name, line)
        if self._task_state.get("task_type") == "idle":
            return
        if progress and self._last_task_log_was_progress and self._task_log_lines:
            self._task_log_lines[-1] = line
        elif not progress and self._last_task_log_was_progress and self._task_log_lines and self._task_log_lines[-1] == line:
            pass
        else:
            self._task_log_lines.append(line)
        if len(self._task_log_lines) > _TASK_LOG_LIMIT:
            self._task_log_lines = self._task_log_lines[-_TASK_LOG_LIMIT:]
        self._task_log_dirty_count += 1
        self._last_task_log_was_progress = progress
        if self._task_log_dirty_count >= _TASK_LOG_PERSIST_INTERVAL:
            self._task_log_dirty_count = 0
            self._persist_active_task_state()

    def _update_task_details(self, patch: Dict[str, Any]) -> None:
        if self._task_state.get("task_type") == "idle":
            return
        current_details = dict(self._task_state.get("details") or {})
        next_details = dict(current_details)
        changed = False
        for key, value in patch.items():
            if next_details.get(key) != value:
                next_details[key] = value
                changed = True
        if not changed:
            return
        self._task_state["details"] = next_details
        self._task_state["updated_at"] = _now_iso()
        self._emit_task_snapshot(include_stage_event=False)
        self._persist_active_task_state()

    def _parse_install_progress_patch(self, line: str) -> Optional[Dict[str, Any]]:
        stripped = str(line or "").strip()
        lowered = stripped.lower()
        if not stripped:
            return None

        patch: Dict[str, Any] = {}
        current_details = dict(self._task_state.get("details") or {})

        for spec in _INSTALL_SECTION_SPECS:
            if any(pattern in lowered for pattern in spec["patterns"]):
                patch.update(
                    {
                        "progress_section_key": spec["key"],
                        "progress_section_start_percent": spec["start"],
                        "progress_section_end_percent": spec["end"],
                        "progress_phase": spec["phase"],
                        "progress_phase_label_zh": spec["label_zh"],
                        "progress_phase_label_en": spec["label_en"],
                        "progress_item_label_zh": stripped,
                        "progress_item_label_en": stripped,
                        "progress_downloaded_bytes": 0,
                        "progress_total_bytes": 0,
                        "progress_speed_bytes_per_sec": 0,
                        "progress_eta_seconds": None,
                        "progress_item_percent": 0,
                    }
                )
                break

        download_match = _DOWNLOAD_PROGRESS_RE.search(stripped)
        if download_match:
            downloaded_bytes = _size_to_bytes(download_match.group("done"), download_match.group("unit"))
            total_bytes = _size_to_bytes(download_match.group("total"), download_match.group("unit"))
            if downloaded_bytes is not None:
                patch["progress_downloaded_bytes"] = downloaded_bytes
            if total_bytes is not None:
                patch["progress_total_bytes"] = total_bytes
                if downloaded_bytes is not None and total_bytes > 0:
                    patch["progress_item_percent"] = max(0.0, min(100.0, (downloaded_bytes / total_bytes) * 100.0))
            patch["progress_phase"] = "download"
            patch["progress_phase_label_zh"] = "下载中"
            patch["progress_phase_label_en"] = "Downloading"

            speed_match = _SPEED_RE.search(stripped)
            if speed_match:
                speed_bytes = _size_to_bytes(speed_match.group("speed"), speed_match.group("unit"))
                if speed_bytes is not None:
                    patch["progress_speed_bytes_per_sec"] = speed_bytes
            eta_match = _ETA_RE.search(stripped)
            if eta_match:
                eta_seconds = _eta_to_seconds(eta_match.group("eta"))
                if eta_seconds is not None:
                    patch["progress_eta_seconds"] = eta_seconds

        if stripped.startswith("Downloading "):
            item_label = stripped[len("Downloading ") :].strip().rstrip(".")
            patch.update(
                {
                    "progress_phase": "download",
                    "progress_phase_label_zh": "下载中",
                    "progress_phase_label_en": "Downloading",
                    "progress_item_label_zh": item_label,
                    "progress_item_label_en": item_label,
                }
            )
        elif lowered.startswith("collecting "):
            item_label = stripped[len("Collecting ") :].strip()
            patch.update(
                {
                    "progress_phase": "install",
                    "progress_phase_label_zh": "安装依赖中",
                    "progress_phase_label_en": "Installing dependencies",
                    "progress_item_label_zh": item_label,
                    "progress_item_label_en": item_label,
                }
            )
        elif lowered.startswith("installing collected packages"):
            item_label = stripped.partition(":")[2].strip() or stripped
            patch.update(
                {
                    "progress_phase": "install",
                    "progress_phase_label_zh": "安装依赖中",
                    "progress_phase_label_en": "Installing dependencies",
                    "progress_item_label_zh": item_label,
                    "progress_item_label_en": item_label,
                }
            )
        elif "preparing metadata" in lowered or "installing build dependencies" in lowered or "getting requirements to build wheel" in lowered:
            patch.update(
                {
                    "progress_phase": "install",
                    "progress_phase_label_zh": "准备安装环境",
                    "progress_phase_label_en": "Preparing install environment",
                    "progress_item_label_zh": stripped,
                    "progress_item_label_en": stripped,
                }
            )
        elif "building wheel for" in lowered or "building wheels for collected packages" in lowered or "running build_ext" in lowered or "ninja:" in lowered or "compiling" in lowered:
            patch.update(
                {
                    "progress_phase": "compile",
                    "progress_phase_label_zh": "编译中",
                    "progress_phase_label_en": "Compiling",
                    "progress_item_label_zh": stripped,
                    "progress_item_label_en": stripped,
                }
            )
        elif lowered.startswith("successfully installed"):
            current_section_key = str(current_details.get("progress_section_key") or "").strip()
            current_section_spec = _INSTALL_SECTION_SPEC_BY_KEY.get(current_section_key)
            success_phase = "install"
            if current_section_spec is not None:
                section_phase = str(current_section_spec.get("phase") or "").strip()
                if section_phase in {"compile", "finalizing"}:
                    success_phase = section_phase
            elif str(current_details.get("progress_phase") or "").strip() in {"compile", "finalizing"}:
                success_phase = str(current_details.get("progress_phase") or "").strip()

            if success_phase == "compile":
                phase_label_zh = "编译中"
                phase_label_en = "Compiling"
            elif success_phase == "finalizing":
                phase_label_zh = "收尾中"
                phase_label_en = "Finalizing"
            else:
                phase_label_zh = "安装依赖中"
                phase_label_en = "Installing dependencies"

            patch.update(
                {
                    "progress_phase": success_phase,
                    "progress_phase_label_zh": phase_label_zh,
                    "progress_phase_label_en": phase_label_en,
                    "progress_item_label_zh": stripped,
                    "progress_item_label_en": stripped,
                    "progress_eta_seconds": 0,
                    "progress_item_percent": 100,
                }
            )
        elif lowered.startswith("requirement already satisfied"):
            patch.update(
                {
                    "progress_phase": "install",
                    "progress_phase_label_zh": "检查依赖中",
                    "progress_phase_label_en": "Checking dependencies",
                    "progress_item_label_zh": stripped,
                    "progress_item_label_en": stripped,
                }
            )
        elif lowered.startswith("using cached "):
            patch.update(
                {
                    "progress_item_label_zh": stripped,
                    "progress_item_label_en": stripped,
                }
            )

        return patch or None

    def _handle_install_output_line(self, line: str, *, progress: bool) -> None:
        self._append_task_log_line(line, event_name="install_log", progress=progress)
        if self._task_state.get("task_type") != "install":
            return
        patch = self._parse_install_progress_patch(line)
        if patch:
            self._update_task_details(patch)

    def _build_log_analysis(self) -> Dict[str, Any]:
        lines = list(self._task_log_lines)
        lowered_lines = [line.lower() for line in lines]
        warning_count = sum(1 for line in lowered_lines if "warning" in line or "deprecated" in line)
        error_count = sum(
            1
            for line in lowered_lines
            if "traceback" in line
            or "error" in line
            or "exception" in line
            or "failed" in line
            or "assertionerror" in line
            or "runtimeerror" in line
        )

        signal_specs = [
            (
                "log.traceback_detected",
                "error",
                "检测到 Traceback",
                "Traceback detected",
                ("traceback",),
            ),
            (
                "log.oom_detected",
                "error",
                "检测到显存/内存不足信号",
                "Out-of-memory signal detected",
                ("out of memory", "cuda out of memory", "oom"),
            ),
            (
                "log.checkpoint_mismatch",
                "error",
                "检测到梯度检查点重算不一致",
                "Checkpoint recomputation mismatch detected",
                ("checkpointerror", "a different number of tensors was saved"),
            ),
            (
                "log.runtime_error",
                "error",
                "检测到运行时错误",
                "Runtime error detected",
                ("runtimeerror", "assertionerror", "exception"),
            ),
            (
                "log.warning_detected",
                "warning",
                "检测到警告信息",
                "Warning detected",
                ("warning", "futurewarning", "deprecated"),
            ),
        ]

        signals: list[Dict[str, Any]] = []
        seen_codes: set[str] = set()
        for original, lowered in zip(lines, lowered_lines):
            for code, severity, title_zh, title_en, patterns in signal_specs:
                if code in seen_codes:
                    continue
                if any(pattern in lowered for pattern in patterns):
                    signals.append(
                        {
                            "code": code,
                            "severity": severity,
                            "title_zh": title_zh,
                            "title_en": title_en,
                            "matched_line": original,
                        }
                    )
                    seen_codes.add(code)

        last_warning = next(
            (line for line in reversed(lines) if "warning" in line.lower() or "deprecated" in line.lower()),
            None,
        )
        last_error = next(
            (
                line
                for line in reversed(lines)
                if "traceback" in line.lower()
                or "error" in line.lower()
                or "exception" in line.lower()
                or "failed" in line.lower()
            ),
            None,
        )

        return {
            "line_count": len(lines),
            "warning_count": warning_count,
            "error_count": error_count,
            "signal_count": len(signals),
            "last_warning": last_warning,
            "last_error": last_error,
            "signals": signals[:8],
        }

    def _begin_task(
        self,
        task_type: str,
        stage_code: str,
        stage_label_zh: str,
        stage_label_en: str,
        *,
        runtime_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._task_stage_history = []
        self._task_command_history = []
        self._task_log_lines = []
        self._task_log_dirty_count = 0
        self._last_task_log_was_progress = False
        self._task_state = begin_task_state(
            task_type,
            stage_code,
            stage_label_zh,
            stage_label_en,
            runtime_id=runtime_id,
            details=details,
        )
        self._emit_task_snapshot(include_stage_event=True)
        self._persist_active_task_state()

    def _advance_task(
        self,
        stage_code: str,
        stage_label_zh: str,
        stage_label_en: str,
        *,
        state: str = "running",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._task_state = advance_task_state(
            self._task_state,
            stage_code,
            stage_label_zh,
            stage_label_en,
            state=state,
            details=details,
        )
        self._emit_task_snapshot(include_stage_event=True)
        self._persist_active_task_state()

    def _finish_task(
        self,
        *,
        success: bool,
        stage_code: str,
        stage_label_zh: str,
        stage_label_en: str,
        code: Optional[str] = None,
        result_code: Optional[str] = None,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._task_state = finish_task_state(
            self._task_state,
            success=success,
            stage_code=stage_code,
            stage_label_zh=stage_label_zh,
            stage_label_en=stage_label_en,
            code=code,
            result_code=result_code,
            error=error,
            details=details,
        )
        self._state_store.clear()
        self._emit_task_snapshot(include_stage_event=True)
        if self._task_state.get("task_type") != "idle" and self._task_state.get("task_id"):
            result = build_task_result(self._task_state)
            result["stages"] = [dict(item) for item in self._task_stage_history]
            result["commands"] = [dict(item) for item in self._task_command_history]
            result["log_lines"] = list(self._task_log_lines)
            result["log_analysis"] = self._build_log_analysis()
            self._task_history = push_task_history(self._task_history, result)
            self._history_store.save(self._task_history)
            self._emit("task_result", result)

    def _recover_interrupted_task_if_needed(self) -> None:
        snapshot = self._state_store.load()
        if not snapshot:
            return
        if snapshot.get("task_type") == "idle":
            self._state_store.clear()
            return
        if snapshot.get("state") not in {"pending", "running"}:
            self._state_store.clear()
            return
        interrupted = build_interrupted_task_result(snapshot)
        stage_history = snapshot.get("stage_history")
        if isinstance(stage_history, list) and stage_history:
            interrupted["stages"] = [item for item in stage_history if isinstance(item, dict)]
        else:
            interrupted["stages"] = [build_task_stage_event(snapshot)]
        command_records = snapshot.get("command_records")
        if isinstance(command_records, list) and command_records:
            interrupted["commands"] = [item for item in command_records if isinstance(item, dict)]
        log_lines = snapshot.get("log_lines")
        if isinstance(log_lines, list) and log_lines:
            interrupted["log_lines"] = [line for line in log_lines if isinstance(line, str)]
            self._task_log_lines = interrupted["log_lines"]
            interrupted["log_analysis"] = self._build_log_analysis()
            self._task_log_lines = []
        self._task_history = push_task_history(self._task_history, interrupted)
        self._history_store.save(self._task_history)
        self._state_store.clear()

    def _read_output(self) -> None:
        if not self._process or not self._process.stdout:
            return
        try:
            _read_stream_chunks(
                self._process.stdout,
                lambda line, progress: self._append_task_log_line(
                    line,
                    event_name="console_line",
                    progress=progress,
                ),
            )
        except Exception:
            pass
        finally:
            code = self._process.wait() if self._process else -1
            self._process = None
            self._reader_thread = None
            if self._task_state.get("task_type") == "stop":
                self._finish_task(
                    success=code == 0,
                    stage_code="trainer.stop_completed" if code == 0 else "trainer.stop_failed",
                    stage_label_zh="训练进程已停止" if code == 0 else "训练进程停止异常",
                    stage_label_en="Trainer process stopped" if code == 0 else "Trainer process stop failed",
                    code=None if code == 0 else "trainer.stop_failed",
                    result_code="trainer.stop_completed" if code == 0 else None,
                    error=None if code == 0 else f"Process exited with code {code}",
                    details={"exit_code": code},
                )
            elif self._task_state.get("task_type") == "launch":
                self._record_launch_process_exited(code)
                self._finish_task(
                    success=code == 0,
                    stage_code="trainer.process_exited_cleanly" if code == 0 else "trainer.process_exited_with_error",
                    stage_label_zh="训练进程已正常退出" if code == 0 else "训练进程异常退出",
                    stage_label_en="Trainer process exited cleanly" if code == 0 else "Trainer process exited with error",
                    code=None if code == 0 else "trainer.process_exited_with_error",
                    result_code="trainer.process_exited_cleanly" if code == 0 else None,
                    error=None if code == 0 else f"Process exited with code {code}",
                    details={"exit_code": code},
                )
            self._emit(
                "process_exit",
                {
                    "code": code,
                    "success": code == 0,
                    "result_code": "trainer.process_exited_cleanly" if code == 0 else "trainer.process_exited_with_error",
                },
            )

    def _terminate_process(self, timeout: float = 3.0) -> None:
        process = self._process
        if not process:
            return
        pid = process.pid
        try:
            if process.poll() is not None:
                if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        **hidden_subprocess_kwargs(),
                    )
                return
        except Exception:
            pass
        try:
            process.terminate()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        self._force_kill_process(process=process)

    def _force_kill_process(self, process: Optional[subprocess.Popen] = None) -> None:
        process = process or self._process
        if not process:
            return
        pid = process.pid
        try:
            if process.stdout:
                try:
                    process.stdout.close()
                except Exception:
                    pass
            if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    **hidden_subprocess_kwargs(),
                )
            elif process.poll() is None:
                process.kill()
            process.wait(timeout=2.0)
        except Exception:
            pass
