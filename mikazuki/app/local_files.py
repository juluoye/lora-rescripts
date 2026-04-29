from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mikazuki import launch_utils


REPO_ROOT = launch_utils.base_dir_path()
LOGS_ROOT = REPO_ROOT / "logs"
TRAIN_ROOT = REPO_ROOT / "train"
OUTPUT_ROOT = REPO_ROOT / "output"
SAMPLE_OUTPUT_DIR = OUTPUT_ROOT / "sample"
SD_MODELS_ROOT = REPO_ROOT / "sd-models"
PREVIEW_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_FILE_EXTENSIONS = {".safetensors", ".ckpt", ".pt"}
BUILTIN_PICKER_ROOTS = {
    "folder": TRAIN_ROOT,
    "output-folder": OUTPUT_ROOT,
    "model-file": SD_MODELS_ROOT,
    "file": SD_MODELS_ROOT,
    "model-saved-file": OUTPUT_ROOT,
}


def require_safe_child_name(raw_name: str, *, label: str) -> str:
    name = str(raw_name or "").strip()
    if not name:
        raise ValueError(f"{label} is required")
    if any(part in {"..", ""} for part in Path(name).parts) or "/" in name or "\\" in name:
        raise ValueError(f"Invalid {label}")
    return name


def open_directory_in_shell(target_dir: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(target_dir))  # type: ignore[attr-defined]
        return
    command = ["open", str(target_dir)] if sys.platform == "darwin" else ["xdg-open", str(target_dir)]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
