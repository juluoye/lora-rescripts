from __future__ import annotations

import re
from pathlib import Path

from mikazuki import launch_utils


REPO_ROOT = launch_utils.base_dir_path()
ASSETS_ROOT = REPO_ROOT / "assets"
UI_STATE_ROOT = ASSETS_ROOT / "ui_state"
SAVED_CONFIGS_DIR = UI_STATE_ROOT / "saved_configs"
TASK_HISTORY_FILE = UI_STATE_ROOT / "task_history.json"
LEGACY_LORA_CONFIGS_DIR = UI_STATE_ROOT / "legacy_lora_configs"


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


def sanitize_legacy_lora_page_id(raw_page: str) -> str:
    safe_page = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(raw_page or "")).strip("._ ")
    if not safe_page:
        raise ValueError("页面标识不能为空。")
    return safe_page


def get_legacy_lora_page_dir(raw_page: str) -> Path:
    safe_page = sanitize_legacy_lora_page_id(raw_page)
    ensure_ui_state_root()
    page_dir = LEGACY_LORA_CONFIGS_DIR / safe_page
    page_dir.mkdir(parents=True, exist_ok=True)
    return page_dir
