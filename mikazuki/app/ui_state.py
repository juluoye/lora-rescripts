from __future__ import annotations

import re
from pathlib import Path

from mikazuki import launch_utils


REPO_ROOT = launch_utils.base_dir_path()
ASSETS_ROOT = REPO_ROOT / "assets"
UI_STATE_ROOT = ASSETS_ROOT / "ui_state"
SAVED_CONFIGS_DIR = UI_STATE_ROOT / "saved_configs"
TASK_HISTORY_FILE = UI_STATE_ROOT / "task_history.json"


def ensure_ui_state_root() -> None:
    UI_STATE_ROOT.mkdir(parents=True, exist_ok=True)


def sanitize_saved_config_name(raw_name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", str(raw_name or "")).strip()


def get_saved_config_path(raw_name: str) -> Path:
    safe_name = sanitize_saved_config_name(raw_name)
    if not safe_name:
        raise ValueError("参数名称不能为空。")
    ensure_ui_state_root()
    SAVED_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    return SAVED_CONFIGS_DIR / f"{safe_name}.json"
