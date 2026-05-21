from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from mikazuki.utils.resume_guard import CKPT_EXTENSIONS, resolve_local_path

TB_EVENT_FILE_GLOB = "events.out.tfevents.*"


def is_tensorboard_logging_enabled(config: dict) -> bool:
    log_with = str(config.get("log_with", "") or "").strip().lower()
    if log_with in {"wandb"}:
        return False
    return True


def resolve_tensorboard_logging_root(config: dict, repo_root: Path) -> Path:
    logging_dir_raw = str(config.get("logging_dir", "./logs") or "./logs").strip() or "./logs"
    return resolve_local_path(logging_dir_raw, repo_root)


def serialize_tensorboard_config_path(path: Path) -> str:
    """Write Windows paths with forward slashes so TOML never sees backslash escapes."""
    return path.resolve().as_posix()


def sanitize_tensorboard_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value or "").strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_.")
    return cleaned or "model"


def resolve_tensorboard_model_name(config: dict) -> str:
    output_name = str(config.get("output_name", "") or "").strip()
    if output_name:
        return sanitize_tensorboard_component(output_name)
    train_type = str(config.get("model_train_type", "") or "").strip()
    if train_type:
        return sanitize_tensorboard_component(train_type)
    return "model"


def read_resume_tensorboard_run_dir(config: dict, repo_root: Path) -> Optional[Path]:
    resume_path = str(config.get("resume", "") or "").strip()
    if not resume_path:
        return None

    local_resume_dir = resolve_local_path(resume_path, repo_root)
    if not local_resume_dir.exists() or not local_resume_dir.is_dir():
        return None

    train_state_file = local_resume_dir / "train_state.json"
    if not train_state_file.exists():
        return None

    try:
        data = json.loads(train_state_file.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    for key in ("logging_dir", "logging_run_dir", "tensorboard_run_dir"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        run_dir = resolve_local_path(value.strip(), repo_root)
        if run_dir.exists() and run_dir.is_dir():
            return run_dir

    return None


def find_latest_tensorboard_run(logging_root: Path, model_name: str) -> Optional[Path]:
    if not logging_root.exists() or not logging_root.is_dir():
        return None

    pattern = re.compile(rf"^{re.escape(model_name)}__run(\d{{3,}})__\d{{14}}$")
    candidates = []
    for entry in logging_root.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.fullmatch(entry.name)
        if not match:
            continue
        try:
            candidates.append((int(match.group(1)), entry.stat().st_mtime, entry))
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def build_next_tensorboard_run(logging_root: Path, model_name: str) -> Path:
    logging_root.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(model_name)}__run(\d{{3,}})__\d{{14}}$")
    max_index = 0
    for entry in logging_root.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.fullmatch(entry.name)
        if not match:
            continue
        try:
            max_index = max(max_index, int(match.group(1)))
        except Exception:
            continue

    run_index = max_index + 1
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return logging_root / f"{model_name}__run{run_index:03d}__{timestamp}"


def apply_tensorboard_runtime_config(config: dict, repo_root: Path) -> dict:
    if not isinstance(config, dict) or not is_tensorboard_logging_enabled(config):
        return {"enabled": False}

    logging_root = resolve_tensorboard_logging_root(config, repo_root)
    model_name = resolve_tensorboard_model_name(config)
    changed = False

    run_dir = read_resume_tensorboard_run_dir(config, repo_root)
    reused_from_state = run_dir is not None
    if run_dir is None:
        resume_path = str(config.get("resume", "") or "").strip()
        if resume_path:
            run_dir = find_latest_tensorboard_run(logging_root, model_name)
    reused_existing = run_dir is not None
    if run_dir is None:
        run_dir = build_next_tensorboard_run(logging_root, model_name)

    if config.get("log_with") is None:
        config["log_with"] = "tensorboard"
        changed = True

    logging_root_str = serialize_tensorboard_config_path(logging_root)
    if str(config.get("logging_dir", "") or "").strip() != logging_root_str:
        config["logging_dir"] = logging_root_str
        changed = True

    run_dir_str = serialize_tensorboard_config_path(run_dir)
    if str(config.get("logging_run_dir", "") or "").strip() != run_dir_str:
        config["logging_run_dir"] = run_dir_str
        changed = True

    return {
        "enabled": True,
        "changed": changed,
        "logging_root": logging_root,
        "run_dir": run_dir,
        "model_name": model_name,
        "resume_merge": reused_existing,
        "reused_from_state": reused_from_state,
    }


def snapshot_tensorboard_event_files(run_dir: Optional[Path]) -> dict:
    snapshot = {}
    if run_dir is None or not run_dir.exists():
        return snapshot

    for event_file in run_dir.rglob(TB_EVENT_FILE_GLOB):
        if not event_file.is_file():
            continue
        try:
            stat = event_file.stat()
        except Exception:
            continue
        snapshot[str(event_file.resolve())] = (stat.st_size, stat.st_mtime)

    return snapshot


def list_checkpoint_files_for_run(config: dict, repo_root: Path) -> list[Path]:
    output_dir_raw = str(config.get("output_dir", "./output") or "./output").strip() or "./output"
    output_dir = resolve_local_path(output_dir_raw, repo_root)
    if not output_dir.exists() or not output_dir.is_dir():
        return []

    output_name = str(config.get("output_name", "") or "").strip()
    files: dict[str, Path] = {}
    for ext in CKPT_EXTENSIONS:
        pattern = f"{output_name}*{ext}" if output_name else f"*{ext}"
        for ckpt_file in output_dir.glob(pattern):
            if ckpt_file.is_file():
                files[str(ckpt_file.resolve())] = ckpt_file

    return list(files.values())


def has_new_checkpoint_since(config: dict, repo_root: Path, started_at: float) -> bool:
    output_dir_raw = str(config.get("output_dir", "./output") or "./output").strip() or "./output"
    output_dir = resolve_local_path(output_dir_raw, repo_root)
    output_name = str(config.get("output_name", "") or "").strip()

    for ckpt_file in list_checkpoint_files_for_run(config, repo_root):
        try:
            if ckpt_file.stat().st_mtime >= started_at:
                return True
        except Exception:
            continue

    if output_dir.exists() and output_dir.is_dir():
        for child in output_dir.iterdir():
            if not child.is_dir():
                continue
            if output_name and not child.name.startswith(output_name):
                continue
            if child.name.endswith("-state"):
                continue
            marker_file = child / "model_index.json"
            if not marker_file.is_file():
                continue
            try:
                if max(child.stat().st_mtime, marker_file.stat().st_mtime) >= started_at:
                    return True
            except Exception:
                continue

    return False


def cleanup_tensorboard_records_without_checkpoint(run_dir: Optional[Path], existed_before: bool, event_snapshot: dict) -> None:
    if run_dir is None or not run_dir.exists():
        return

    if not existed_before:
        shutil.rmtree(run_dir, ignore_errors=True)
        return

    existing_keys = set(event_snapshot.keys())
    for event_file in run_dir.rglob(TB_EVENT_FILE_GLOB):
        if not event_file.is_file():
            continue
        if str(event_file.resolve()) in existing_keys:
            continue
        try:
            event_file.unlink()
        except Exception:
            continue

    for dir_path in sorted([p for p in run_dir.rglob("*") if p.is_dir()], reverse=True):
        try:
            dir_path.rmdir()
        except OSError:
            continue
