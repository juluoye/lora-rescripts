from __future__ import annotations

import csv
import importlib
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mikazuki import launch_utils
from mikazuki.utils.runtime_dependencies import analyze_training_runtime_dependencies


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if not normalized:
        return bool(default)
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(value)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text_list(value: Any, fallback: list[str] | None = None) -> list[str]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", ",")
        items = [item.strip() for item in text.split(",") if item.strip()]
    return items or list(fallback or [])


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_aesthetic_inference_symbols():
    repo_root = str(launch_utils.base_dir_path())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        module = importlib.import_module("scripts.stable.lulynx.aesthetic_fusion.inference")
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Aesthetic inference helper is unavailable because the local scripts package could not be imported. "
            "Please confirm the release package contains ./scripts/stable/lulynx/aesthetic_fusion/inference.py. "
            "/ 美学推理模块当前不可用：无法导入本地 scripts 包，请确认发行包内包含 "
            "./scripts/stable/lulynx/aesthetic_fusion/inference.py。"
        ) from exc

    default_image_extensions = getattr(module, "DEFAULT_IMAGE_EXTENSIONS", None)
    targets = getattr(module, "TARGETS", None)
    score_bucket = getattr(module, "score_bucket", None)
    if default_image_extensions is None or targets is None or not callable(score_bucket):
        raise ValueError(
            "Aesthetic inference symbols are incomplete in the current build. "
            "/ 当前发行包中的美学推理模块缺少必需符号。"
        )

    return tuple(default_image_extensions), tuple(targets), score_bucket


def _get_aesthetic_targets() -> tuple[str, ...]:
    _, targets, _ = _load_aesthetic_inference_symbols()
    return targets


def _get_aesthetic_default_image_extensions() -> tuple[str, ...]:
    default_image_extensions, _, _ = _load_aesthetic_inference_symbols()
    return default_image_extensions


def _score_bucket(score: float | None):
    _, _, score_bucket = _load_aesthetic_inference_symbols()
    return score_bucket(score)


class AestheticInferManager:
    def __init__(self) -> None:
        self.repo_root = launch_utils.base_dir_path()
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.current_task: dict[str, Any] | None = None
        self._starting = False
        self._start_cancel_requested = False
        self.logs: list[dict[str, Any]] = []
        self.next_log_id = 1
        self.records_cache: dict[str, dict[str, Any]] = {}

    def _build_script_env(self) -> tuple[Path, dict[str, str]]:
        script_path = (self.repo_root / "scripts" / "stable" / "aesthetic_scorer_infer.py").resolve()
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        parts = [part for part in existing.split(os.pathsep) if part]
        repo_root_str = str(self.repo_root)
        if repo_root_str not in parts:
            env["PYTHONPATH"] = os.pathsep.join([repo_root_str, *parts]) if parts else repo_root_str
        return script_path, env

    def _append_log(self, message: str) -> None:
        text = str(message or "").rstrip("\r\n")
        if not text:
            return
        with self.lock:
            self.logs.append(
                {
                    "id": self.next_log_id,
                    "time": _now(),
                    "text": text,
                }
            )
            self.next_log_id += 1
            if len(self.logs) > 4000:
                self.logs = self.logs[-4000:]

    def _resolve_path(self, raw: Any) -> Path | None:
        value = _normalize_text(raw)
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (self.repo_root / path).resolve()
        else:
            path = path.resolve()
        return path

    def _resolve_output_dir(self, raw: Any, checkpoint: Path | None = None) -> Path:
        resolved = self._resolve_path(raw)
        if resolved is not None:
            return resolved
        if checkpoint is not None:
            return (checkpoint.parent / "infer_run").resolve()
        return (self.repo_root / "output" / "aesthetic-infer").resolve()

    def _dependency_report(self) -> dict:
        return analyze_training_runtime_dependencies({"model_train_type": "aesthetic-scorer"})

    def get_runtime_payload(self) -> dict:
        report = self._dependency_report()
        with self.lock:
            task = dict(self.current_task) if self.current_task else None
            running = self._starting or (self.proc is not None and self.proc.poll() is None)
        return {
            "dependencies": report,
            "running": running,
            "task": task,
            "support": {
                "batch_inference": True,
                "single_image_api": False,
                "result_gallery": True,
                "filtered_result_table": True,
                "folder_organize": True,
            },
        }

    def _normalize_start_payload(self, payload: dict[str, Any]) -> tuple[bool, str, dict[str, Any] | None]:
        dependency_report = self._dependency_report()
        if not dependency_report.get("ready"):
            missing = ", ".join(item.get("display_name", item.get("module_name", "")) for item in dependency_report.get("missing", []))
            return False, f"当前运行环境缺少美学推理依赖: {missing}", None

        checkpoint = self._resolve_path(payload.get("checkpoint"))
        if checkpoint is None:
            return False, "checkpoint 不能为空。", None
        if not checkpoint.exists() or not checkpoint.is_file():
            return False, f"checkpoint 不存在: {checkpoint}", None

        input_dir = self._resolve_path(payload.get("input_dir"))
        if input_dir is None:
            return False, "input_dir 不能为空。", None
        if not input_dir.exists() or not input_dir.is_dir():
            return False, f"input_dir 不存在或不是目录: {input_dir}", None

        output_dir = self._resolve_output_dir(payload.get("output_dir"), checkpoint=checkpoint)
        batch_size = max(1, int(payload.get("batch_size") or 8))
        special_threshold = float(payload.get("special_threshold") if payload.get("special_threshold") not in (None, "") else 0.5)
        if special_threshold < 0.0 or special_threshold > 1.0:
            return False, "special_threshold 必须在 0 到 1 之间。", None

        recursive = _normalize_bool(payload.get("recursive"), True)
        save_jsonl = _normalize_bool(payload.get("save_jsonl"), True)
        save_csv = _normalize_bool(payload.get("save_csv"), True)
        organize_enabled = _normalize_bool(payload.get("organize_enabled"), False)
        organize_include_special_group = _normalize_bool(payload.get("organize_include_special_group"), True)
        organize_root_dir = self._resolve_path(payload.get("organize_root_dir")) or (output_dir / "organized").resolve()
        organize_mode = _normalize_text(payload.get("organize_mode")) or "copy"
        if organize_mode not in {"copy", "move", "hardlink", "symlink"}:
            return False, f"organize_mode 不支持: {organize_mode}", None

        targets = _get_aesthetic_targets()
        organize_dimensions = [item.lower() for item in _normalize_text_list(payload.get("organize_dimensions"), list(targets))]
        invalid_dimensions = [item for item in organize_dimensions if item not in targets]
        if invalid_dimensions:
            return False, f"organize_dimensions 包含非法项: {invalid_dimensions}", None

        image_extensions = _normalize_text_list(payload.get("image_extensions"), list(_get_aesthetic_default_image_extensions()))
        device = _normalize_text(payload.get("device")) or ""

        return True, "", {
            "checkpoint": str(checkpoint),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "batch_size": batch_size,
            "special_threshold": special_threshold,
            "recursive": recursive,
            "save_jsonl": save_jsonl,
            "save_csv": save_csv,
            "jsonl_name": _normalize_text(payload.get("jsonl_name")) or "predictions.jsonl",
            "csv_name": _normalize_text(payload.get("csv_name")) or "predictions.csv",
            "organize_enabled": organize_enabled,
            "organize_root_dir": str(organize_root_dir),
            "organize_mode": organize_mode,
            "organize_include_special_group": organize_include_special_group,
            "organize_dimensions": organize_dimensions,
            "organize_bucket_strategy": _normalize_text(payload.get("organize_bucket_strategy")) or "nearest_int",
            "image_extensions": image_extensions,
            "device": device,
        }

    def start(self, payload: dict[str, Any]) -> tuple[bool, str, dict[str, Any] | None]:
        try:
            ok, message, normalized = self._normalize_start_payload(payload)
        except (RuntimeError, ValueError) as exc:
            return False, str(exc), None
        if not ok or normalized is None:
            return False, message, None

        task_id = uuid.uuid4().hex[:12]
        task_payload = {
            "task_id": task_id,
            "status": "starting",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "pid": None,
            "params": normalized,
            "summary": None,
            "summary_path": None,
            "error": None,
        }

        with self.lock:
            if self._starting or (self.proc is not None and self.proc.poll() is None):
                return False, "当前已有美学推理任务正在运行或启动中。", None
            self.logs = []
            self.next_log_id = 1
            self.current_task = task_payload
            self._starting = True
            self._start_cancel_requested = False

        try:
            script_path, env = self._build_script_env()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

            command = [
                sys.executable,
                str(script_path),
                "--checkpoint",
                normalized["checkpoint"],
                "--input_dir",
                normalized["input_dir"],
                "--output_dir",
                normalized["output_dir"],
                "--batch_size",
                str(normalized["batch_size"]),
                "--recursive",
                "true" if normalized["recursive"] else "false",
                "--special_threshold",
                str(normalized["special_threshold"]),
                "--save_jsonl",
                "true" if normalized["save_jsonl"] else "false",
                "--save_csv",
                "true" if normalized["save_csv"] else "false",
                "--jsonl_name",
                normalized["jsonl_name"],
                "--csv_name",
                normalized["csv_name"],
                "--organize_enabled",
                "true" if normalized["organize_enabled"] else "false",
                "--organize_root_dir",
                normalized["organize_root_dir"],
                "--organize_mode",
                normalized["organize_mode"],
                "--organize_include_special_group",
                "true" if normalized["organize_include_special_group"] else "false",
                "--organize_dimensions",
                ",".join(normalized["organize_dimensions"]),
                "--organize_bucket_strategy",
                normalized["organize_bucket_strategy"],
                "--image_extensions",
                ",".join(normalized["image_extensions"]),
            ]
            if normalized["device"]:
                command.extend(["--device", normalized["device"]])

            cancelled_before_launch = False
            with self.lock:
                if self.current_task and self.current_task.get("task_id") == task_id and self._start_cancel_requested:
                    self.current_task["status"] = "stopped"
                    self.current_task["finished_at"] = _now()
                    self.proc = None
                    self._starting = False
                    self._start_cancel_requested = False
                    cancelled_before_launch = True
            if cancelled_before_launch:
                self._append_log("[aesthetic_infer] 启动阶段已取消，本次不会启动推理进程。")
                return False, "美学推理任务在启动阶段已取消。", {"task_id": task_id, "params": normalized}

            proc = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            with self.lock:
                if self.current_task and self.current_task.get("task_id") == task_id:
                    self.current_task["status"] = "failed"
                    self.current_task["finished_at"] = _now()
                    self.current_task["error"] = str(exc)
                self.proc = None
                self._starting = False
                self._start_cancel_requested = False
            self._append_log(f"[aesthetic_infer] 启动失败: {exc}")
            return False, f"启动美学推理失败: {exc}", None

        cancel_requested = False
        with self.lock:
            self.proc = proc
            task_payload["pid"] = proc.pid
            task_payload["started_at"] = _now()
            cancel_requested = bool(self._start_cancel_requested)
            task_payload["status"] = "stopping" if cancel_requested else "running"
            self.current_task = task_payload
            self._starting = False
            self._start_cancel_requested = False

        self._append_log(f"[aesthetic_infer] 启动推理: {normalized['checkpoint']}")
        self._append_log(f"[aesthetic_infer] 命令: {' '.join(command)}")

        threading.Thread(target=self._stream_output, args=(proc,), daemon=True).start()
        threading.Thread(target=self._wait_process, args=(proc, task_id), daemon=True).start()

        if cancel_requested:
            self._append_log("[aesthetic_infer] 启动阶段已收到停止请求，正在终止推理进程。")
            try:
                proc.terminate()
            except Exception as exc:
                with self.lock:
                    if self.current_task and self.current_task.get("task_id") == task_id:
                        self.current_task["status"] = "failed"
                        self.current_task["finished_at"] = _now()
                        self.current_task["error"] = str(exc)
                    self.proc = None
                self._append_log(f"[aesthetic_infer] 启动后立即停止失败: {exc}")
                return False, f"美学推理任务在启动后取消失败: {exc}", {"task_id": task_id, "params": normalized}
            return True, "美学推理任务已启动，并已收到停止请求。", {"task_id": task_id, "params": normalized}

        return True, "美学推理任务已启动。", {"task_id": task_id, "params": normalized}

    def _stream_output(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            self._append_log(line)

    def _wait_process(self, proc: subprocess.Popen[str], task_id: str) -> None:
        return_code = proc.wait()
        summary = None
        summary_path = None
        with self.lock:
            task = self.current_task if self.current_task and self.current_task.get("task_id") == task_id else None
            if task is not None:
                output_dir = self._resolve_output_dir(task.get("params", {}).get("output_dir"))
                candidate_summary_path = output_dir / "summary.json"
                if candidate_summary_path.exists():
                    summary_path = str(candidate_summary_path)
                    try:
                        summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
                    except Exception:
                        summary = None
                if task.get("status") == "stopping":
                    task["status"] = "stopped"
                else:
                    task["status"] = "done" if return_code == 0 else "failed"
                task["return_code"] = int(return_code)
                task["finished_at"] = _now()
                task["summary"] = summary
                task["summary_path"] = summary_path
            self.proc = None
            self._starting = False

    def stop(self) -> tuple[bool, str, dict[str, Any] | None]:
        proc = None
        task_id = None
        cancelling_startup = False
        with self.lock:
            if self._starting:
                cancelling_startup = True
                self._start_cancel_requested = True
                if self.current_task is not None:
                    self.current_task["status"] = "stopping"
                    task_id = self.current_task.get("task_id")
                proc = self.proc if self.proc is not None and self.proc.poll() is None else None
            elif self.proc is None or self.proc.poll() is not None:
                return False, "当前没有运行中的美学推理任务。", None
            else:
                proc = self.proc
                if self.current_task is not None:
                    self.current_task["status"] = "stopping"
                    task_id = self.current_task.get("task_id")
        if cancelling_startup and proc is None:
            self._append_log("[aesthetic_infer] 收到停止请求，正在取消启动中的推理任务。")
            return True, "已发送停止请求，正在取消启动。", {"task_id": task_id}
        if proc is None:
            return False, "当前没有运行中的美学推理任务。", None
        proc.terminate()
        if cancelling_startup:
            self._append_log("[aesthetic_infer] 收到停止请求，正在终止刚启动的推理进程。")
        else:
            self._append_log("[aesthetic_infer] 收到停止请求，正在终止推理进程。")
        return True, "已发送停止请求。", {"task_id": task_id}

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            task = dict(self.current_task) if self.current_task else None
            running = self._starting or (self.proc is not None and self.proc.poll() is None)
            last_log_id = self.logs[-1]["id"] if self.logs else 0
            log_count = len(self.logs)
        return {
            "task": task,
            "running": running,
            "log_count": log_count,
            "last_log_id": last_log_id,
        }

    def get_logs(self, since_id: int = 0, limit: int = 300) -> dict[str, Any]:
        with self.lock:
            items = [item for item in self.logs if int(item["id"]) > int(since_id)][: max(1, min(int(limit), 2000))]
            last_id = self.logs[-1]["id"] if self.logs else 0
        return {
            "items": items,
            "last_id": last_id,
        }

    def _read_summary(self, output_dir: Path) -> dict[str, Any] | None:
        summary_path = output_dir / "summary.json"
        if not summary_path.exists():
            return None
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _pick_prediction_file(self, output_dir: Path, summary: dict[str, Any] | None) -> Path | None:
        if summary and isinstance(summary.get("output_files"), dict):
            output_files = summary["output_files"]
            for key in ("jsonl", "csv"):
                candidate = self._resolve_path(output_files.get(key))
                if candidate and candidate.exists():
                    return candidate
        for name in ("predictions.jsonl", "predictions.csv"):
            candidate = output_dir / name
            if candidate.exists():
                return candidate
        return None

    def _load_records_cached(self, prediction_file: Path) -> list[dict[str, Any]]:
        cache_key = str(prediction_file.resolve())
        stat = prediction_file.stat()
        with self.lock:
            cached = self.records_cache.get(cache_key)
            if cached and cached.get("mtime_ns") == stat.st_mtime_ns and cached.get("size") == stat.st_size:
                return list(cached["records"])

        rows: list[dict[str, Any]] = []
        if prediction_file.suffix.lower() == ".jsonl":
            with prediction_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(self._format_record(json.loads(line)))
                    except Exception:
                        continue
        else:
            with prediction_file.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows.append(self._format_record(row))

        with self.lock:
            self.records_cache[cache_key] = {
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
                "records": list(rows),
            }
        return rows

    def _format_record(self, row: dict[str, Any]) -> dict[str, Any]:
        formatted = dict(row)
        for key in _get_aesthetic_targets():
            formatted[key] = _safe_float(formatted.get(key))
        formatted["in_domain_prob"] = _safe_float(formatted.get("in_domain_prob"))
        formatted["in_domain_pred"] = _safe_int(formatted.get("in_domain_pred"))
        formatted["special_tag"] = _safe_int(formatted.get("special_tag"))
        formatted["special_reason"] = str(formatted.get("special_reason") or "")
        formatted["error"] = str(formatted.get("error") or "")
        return formatted

    def _decompose_record(self, row: dict[str, Any]) -> dict[str, Any]:
        formatted = self._format_record(row)
        score_heads = []
        names = {
            "aesthetic": "美学",
            "composition": "构图",
            "color": "色彩",
            "sexual": "色情",
        }
        for key in _get_aesthetic_targets():
            score = formatted.get(key)
            score_heads.append(
                {
                    "key": key,
                    "name": names.get(key, key),
                    "score": score,
                "bucket": _score_bucket(score) if score is not None else None,
                }
            )
        formatted["score_heads"] = score_heads
        return formatted

    def get_results(
        self,
        *,
        output_dir_raw: Any,
        page: int = 1,
        page_size: int = 24,
        keyword: str = "",
        special_filter: str = "all",
        sort_by: str = "",
        sort_order: str = "desc",
    ) -> tuple[bool, str, dict[str, Any] | None]:
        output_dir = self._resolve_output_dir(output_dir_raw)
        summary = self._read_summary(output_dir)
        prediction_file = self._pick_prediction_file(output_dir, summary)
        if prediction_file is None or not prediction_file.exists():
            return False, "未找到推理结果文件。", None

        records = self._load_records_cached(prediction_file)
        items = list(records)
        normalized_keyword = str(keyword or "").strip().lower()
        if normalized_keyword:
            items = [
                item
                for item in items
                if normalized_keyword in str(item.get("relative_path") or "").lower()
                or normalized_keyword in str(item.get("image_path") or "").lower()
            ]

        normalized_special_filter = str(special_filter or "all").strip().lower()
        if normalized_special_filter == "special":
            items = [item for item in items if int(item.get("special_tag") or 0) == 1]
        elif normalized_special_filter == "in_domain":
            items = [item for item in items if int(item.get("special_tag") or 0) == 0]

        allowed_sort = {"aesthetic", "composition", "color", "sexual", "in_domain_prob", "special_tag"}
        if sort_by in allowed_sort:
            reverse = str(sort_order or "desc").lower() != "asc"
            items = sorted(items, key=lambda item: (item.get(sort_by) is None, item.get(sort_by)), reverse=reverse)

        normalized_page_size = max(1, min(int(page_size), 200))
        normalized_page = max(1, int(page))
        total = len(items)
        pages = (total + normalized_page_size - 1) // normalized_page_size if total else 0
        if pages > 0 and normalized_page > pages:
            normalized_page = pages
        start = (normalized_page - 1) * normalized_page_size
        end = start + normalized_page_size
        paged = items[start:end]

        result_items = []
        for row in paged:
            item = self._decompose_record(row)
            item["image_url"] = f"/api/aesthetic_infer/image?path={quote(str(item.get('image_path', '') or ''))}"
            result_items.append(item)

        special_count = sum(1 for row in records if int(row.get("special_tag") or 0) == 1)
        return True, "", {
            "output_dir": str(output_dir),
            "summary": summary,
            "prediction_file": str(prediction_file),
            "total": total,
            "page": normalized_page,
            "pages": pages,
            "page_size": normalized_page_size,
            "special_count": special_count,
            "records_count": len(records),
            "items": result_items,
        }

    def resolve_output_file(self, output_dir_raw: Any, kind: str) -> Path | None:
        output_dir = self._resolve_output_dir(output_dir_raw)
        summary = self._read_summary(output_dir)
        normalized_kind = str(kind or "").strip().lower()
        if normalized_kind == "summary":
            candidate = output_dir / "summary.json"
            return candidate if candidate.exists() else None
        if summary and isinstance(summary.get("output_files"), dict):
            output_files = summary["output_files"]
            if normalized_kind in {"jsonl", "csv"}:
                candidate = self._resolve_path(output_files.get(normalized_kind))
                if candidate and candidate.exists():
                    return candidate
        fallback_name = {
            "jsonl": "predictions.jsonl",
            "csv": "predictions.csv",
        }.get(normalized_kind)
        if fallback_name:
            candidate = output_dir / fallback_name
            if candidate.exists():
                return candidate
        return None

    def resolve_image_path(self, raw_path: Any) -> Path | None:
        path = self._resolve_path(raw_path)
        if path is None or not path.exists() or not path.is_file():
            return None
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
            return None
        return path


aesthetic_infer_manager = AestheticInferManager()
