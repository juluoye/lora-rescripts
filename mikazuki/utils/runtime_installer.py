from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from mikazuki.launch_utils import base_dir_path
from mikazuki.log import log
from mikazuki.utils.runtime_dependencies import analyze_training_runtime_dependencies


RuntimePayloadBuilder = Callable[[dict, dict], dict]


@dataclass(frozen=True)
class RuntimeInstallerSpec:
    training_type: str
    display_name: str
    status_file: Path
    requirement_specs: dict[str, str]
    ready_detail: str
    missing_detail: str
    running_detail: str
    completed_detail: str
    already_ready_message: str
    already_running_message: str
    install_started_message: str
    ready_reset_statuses: tuple[str, ...] = ("idle", "ready", "completed")
    restart_requires_new_pid: bool = False
    repo_path: Path | None = None
    repo_missing_message: str | None = None
    payload_builder: RuntimePayloadBuilder | None = None


class RuntimeDependencyInstaller:
    def __init__(self, spec: RuntimeInstallerSpec):
        self.spec = spec
        self._install_lock = threading.Lock()
        self._install_thread: threading.Thread | None = None

    def _default_runtime_status(self) -> dict:
        return {
            "status": "idle",
            "detail": "",
            "logs": [],
            "started_at": "",
            "finished_at": "",
            "restart_required": False,
            "requirements": [],
            "installer_pid": 0,
        }

    def read_status(self) -> dict:
        status = self._default_runtime_status()
        if not self.spec.status_file.exists():
            return status

        try:
            with open(self.spec.status_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return status
            status.update(
                {
                    "status": str(data.get("status", status["status"]) or status["status"]),
                    "detail": str(data.get("detail", status["detail"]) or ""),
                    "logs": list(data.get("logs", status["logs"]) or []),
                    "started_at": str(data.get("started_at", status["started_at"]) or ""),
                    "finished_at": str(data.get("finished_at", status["finished_at"]) or ""),
                    "restart_required": bool(data.get("restart_required", status["restart_required"])),
                    "requirements": list(data.get("requirements", status["requirements"]) or []),
                    "installer_pid": int(data.get("installer_pid", status["installer_pid"]) or 0),
                }
            )
        except Exception as exc:
            log.warning(f"Failed to read {self.spec.display_name} runtime status file: {exc}")
        return status

    def write_status(
        self,
        status: str,
        *,
        detail: str | None = None,
        logs: list[str] | None = None,
        restart_required: bool | None = None,
        requirements: list[str] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        installer_pid: int | None = None,
    ) -> dict:
        payload = self.read_status()
        payload["status"] = str(status or payload["status"]).strip() or payload["status"]
        if detail is not None:
            payload["detail"] = str(detail or "").strip()
        if logs is not None:
            payload["logs"] = [str(item) for item in logs][-200:]
        if restart_required is not None:
            payload["restart_required"] = bool(restart_required)
        if requirements is not None:
            payload["requirements"] = [str(item) for item in requirements]
        if started_at is not None:
            payload["started_at"] = started_at
        if finished_at is not None:
            payload["finished_at"] = finished_at
        if installer_pid is not None:
            payload["installer_pid"] = int(installer_pid or 0)

        self.spec.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.spec.status_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return payload

    def append_log(self, line: str) -> dict:
        payload = self.read_status()
        logs = list(payload.get("logs", []) or [])
        text = str(line or "").rstrip()
        if text:
            logs.append(text)
        return self.write_status(
            payload["status"],
            detail=payload.get("detail", ""),
            logs=logs[-200:],
            restart_required=bool(payload.get("restart_required", False)),
            requirements=list(payload.get("requirements", []) or []),
            started_at=str(payload.get("started_at", "") or ""),
            finished_at=str(payload.get("finished_at", "") or ""),
            installer_pid=int(payload.get("installer_pid", 0) or 0),
        )

    def get_dependency_report(self) -> dict:
        return analyze_training_runtime_dependencies({"model_train_type": self.spec.training_type})

    def get_install_requirements(self, dependency_report: dict | None = None) -> list[str]:
        report = dependency_report if isinstance(dependency_report, dict) else self.get_dependency_report()
        requirements: list[str] = []
        seen: set[str] = set()
        for dependency in report.get("missing", []):
            module_name = str(dependency.get("module_name", "") or "").strip()
            requirement = self.spec.requirement_specs.get(module_name)
            if not requirement or requirement in seen:
                continue
            seen.add(requirement)
            requirements.append(requirement)
        return requirements

    def build_payload(self) -> dict:
        dependency_report = self.get_dependency_report()
        install_status = self.read_status()
        current_pid = os.getpid()

        if dependency_report.get("ready"):
            if install_status.get("restart_required"):
                should_clear_restart = True
                if self.spec.restart_requires_new_pid:
                    installer_pid = int(install_status.get("installer_pid", 0) or 0)
                    should_clear_restart = installer_pid > 0 and installer_pid != current_pid

                if should_clear_restart:
                    install_status = self.write_status(
                        "ready",
                        detail=self.spec.ready_detail,
                        restart_required=False,
                        requirements=[],
                        finished_at=datetime.now().isoformat(timespec="seconds"),
                        installer_pid=0,
                    )
            elif install_status.get("status") in self.spec.ready_reset_statuses:
                install_status = self.write_status(
                    "ready",
                    detail=self.spec.ready_detail,
                    restart_required=False,
                    requirements=[],
                    installer_pid=0,
                )
        elif install_status.get("status") not in {"running", "completed", "failed"}:
            install_status = self.write_status(
                "idle",
                detail=self.spec.missing_detail,
                restart_required=False,
                requirements=self.get_install_requirements(dependency_report),
                installer_pid=0,
            )

        payload = {
            "dependencies": dependency_report,
            "install": install_status,
            "missing_requirements": self.get_install_requirements(dependency_report),
        }
        if self.spec.payload_builder is not None:
            payload.update(self.spec.payload_builder(dependency_report, install_status))
        return payload

    def _install_requirements_worker(self, requirements: list[str]) -> None:
        started_at = datetime.now().isoformat(timespec="seconds")
        installer_pid = os.getpid()
        self.write_status(
            "running",
            detail=self.spec.running_detail,
            logs=[f"[{started_at}] 准备安装: {' '.join(requirements)}"],
            restart_required=False,
            requirements=requirements,
            started_at=started_at,
            finished_at="",
            installer_pid=installer_pid,
        )

        env = os.environ.copy()
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        cmd = [sys.executable, "-m", "pip", "install", "--no-warn-script-location", *requirements]

        try:
            self.append_log(f"$ {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                cwd=str(base_dir_path()),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                self.append_log(line)

            try:
                return_code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                self.write_status(
                    "failed",
                    detail="pip install timed out after 30 minutes.",
                    requirements=requirements,
                    started_at=started_at,
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                    installer_pid=installer_pid,
                )
                return
            finished_at = datetime.now().isoformat(timespec="seconds")
            if return_code == 0:
                self.write_status(
                    "completed",
                    detail=self.spec.completed_detail,
                    restart_required=True,
                    requirements=requirements,
                    started_at=started_at,
                    finished_at=finished_at,
                    installer_pid=installer_pid,
                )
            else:
                self.write_status(
                    "failed",
                    detail=f"{self.spec.display_name}依赖安装失败，pip 退出码: {return_code}",
                    restart_required=False,
                    requirements=requirements,
                    started_at=started_at,
                    finished_at=finished_at,
                    installer_pid=installer_pid,
                )
        except Exception as exc:
            log.exception(f"{self.spec.display_name} dependency installation failed unexpectedly")
            self.append_log(f"[error] {exc}")
            self.write_status(
                "failed",
                detail=f"{self.spec.display_name}依赖安装失败: {exc}",
                restart_required=False,
                requirements=requirements,
                started_at=started_at,
                finished_at=datetime.now().isoformat(timespec="seconds"),
                installer_pid=installer_pid,
            )
        finally:
            with self._install_lock:
                self._install_thread = None

    def start_install(self) -> tuple[bool, str, dict]:
        if self.spec.repo_path is not None and (not self.spec.repo_path.exists() or not self.spec.repo_path.is_dir()):
            payload = self.build_payload()
            message = self.spec.repo_missing_message or f"未找到依赖目录: {self.spec.repo_path}"
            return False, message, payload

        dependency_report = self.get_dependency_report()
        requirements = self.get_install_requirements(dependency_report)
        if not requirements:
            payload = self.build_payload()
            return False, self.spec.already_ready_message, payload

        with self._install_lock:
            if self._install_thread is not None and self._install_thread.is_alive():
                payload = self.build_payload()
                return False, self.spec.already_running_message, payload

            self._install_thread = threading.Thread(
                target=self._install_requirements_worker,
                args=(requirements,),
                name=f"mikazuki-{self.spec.training_type}-dependency-install",
                daemon=True,
            )
            self._install_thread.start()

        payload = self.build_payload()
        return True, self.spec.install_started_message, payload
