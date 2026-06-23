from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

try:
    import toml
except ModuleNotFoundError:
    toml = None

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from mikazuki.app.config import app_config
from mikazuki.app.models import APIResponse, APIResponseSuccess
from mikazuki.app.ui_state import (
    LEGACY_LORA_CONFIGS_DIR,
    SAVED_CONFIGS_DIR,
    TASK_HISTORY_FILE,
    ensure_ui_state_root,
    get_legacy_lora_page_dir,
    get_saved_config_path,
    sanitize_saved_config_name,
)
from mikazuki.utils.frontend_profiles import resolve_frontend_profile_id


router = APIRouter()
REPO_ROOT = TASK_HISTORY_FILE.parents[2]
AUTOSAVE_DIR = REPO_ROOT / "config" / "autosave"


def _json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "error", "message": message})


def _load_local_task_history() -> list[dict]:
    ensure_ui_state_root()
    try:
        tasks = json.loads(TASK_HISTORY_FILE.read_text(encoding="utf-8")) if TASK_HISTORY_FILE.exists() else []
    except Exception:
        tasks = []
    if not isinstance(tasks, list):
        return []
    return [item for item in tasks if isinstance(item, dict)]


def _save_local_task_history(tasks: list[dict]) -> None:
    ensure_ui_state_root()
    TASK_HISTORY_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_legacy_lora_page_files(page: str):
    page_dir = get_legacy_lora_page_dir(page)
    return sorted(
        [file for file in page_dir.glob("*.toml") if file.is_file()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _legacy_config_item(file: Path, source: str) -> dict:
    return {
        "name": file.stem,
        "time": int(file.stat().st_mtime * 1000),
        "source": source,
    }


def _load_toml_config(file: Path) -> dict:
    raw_text = file.read_text(encoding="utf-8")
    if tomllib is not None:
        payload = tomllib.loads(raw_text)
    elif toml is not None:
        payload = toml.loads(raw_text)
    else:
        raise RuntimeError("当前运行环境缺少 TOML 解析能力。")
    if not isinstance(payload, dict):
        raise ValueError("参数文件格式无效。")
    return payload


def _toml_escape_string(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _toml_format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return f'"{_toml_escape_string(value)}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_format_value(item) for item in value) + "]"
    raise TypeError(f"暂不支持写入 TOML 的值类型：{type(value).__name__}")


def _dump_toml_lines(data: dict, prefix: str = "") -> list[str]:
    lines: list[str] = []
    child_tables: list[tuple[str, dict]] = []
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, dict):
            child_tables.append((key, value))
            continue
        lines.append(f"{key} = {_toml_format_value(value)}")

    for key, value in child_tables:
        table_name = f"{prefix}.{key}" if prefix else key
        if lines:
            lines.append("")
        lines.append(f"[{table_name}]")
        lines.extend(_dump_toml_lines(value, table_name))
    return lines


def _write_toml_config(file: Path, payload: dict) -> None:
    if toml is not None:
        file.write_text(toml.dumps(payload), encoding="utf-8")
        return
    file.write_text("\n".join(_dump_toml_lines(payload)).strip() + "\n", encoding="utf-8")


@router.get("/config/saved_params")
async def get_saved_params() -> APIResponse:
    saved_params = app_config["saved_params"]
    return APIResponseSuccess(data=saved_params)


@router.get("/config/summary")
async def get_config_summary() -> APIResponse:
    return APIResponseSuccess(data={
        "last_path": app_config["last_path"] or "",
        "saved_param_keys": sorted((app_config["saved_params"] or {}).keys()),
        "saved_param_count": len(app_config["saved_params"] or {}),
        "active_ui_profile": resolve_frontend_profile_id(app_config["active_ui_profile"]),
        "config_path": str(app_config.path),
        "plugin_developer_mode": bool(app_config["plugin_developer_mode"]),
    })


@router.get("/legacy_lora_configs/list")
async def list_legacy_lora_configs(page: str) -> APIResponse:
    local_configs = [_legacy_config_item(file, "legacy-local") for file in _iter_legacy_lora_page_files(page)]

    autosave_configs: list[dict] = []
    if AUTOSAVE_DIR.exists():
        autosave_configs = [
            _legacy_config_item(file, "autosave")
            for file in sorted(
                [item for item in AUTOSAVE_DIR.glob("*.toml") if item.is_file()],
                key=lambda entry: entry.stat().st_mtime,
                reverse=True,
            )[:50]
        ]

    return APIResponseSuccess(data={"configs": local_configs, "autosave": autosave_configs})


@router.get("/legacy_lora_configs/load")
async def load_legacy_lora_config(page: str, name: str, source: str = "legacy-local"):
    if source == "autosave":
        safe_name = sanitize_saved_config_name(name)
        file_path = (AUTOSAVE_DIR / f"{safe_name}.toml").resolve()
        try:
            file_path.relative_to(AUTOSAVE_DIR.resolve())
        except ValueError:
            return _json_error("参数文件名无效。")
    else:
        safe_name = sanitize_saved_config_name(name)
        page_dir = get_legacy_lora_page_dir(page)
        file_path = (page_dir / f"{safe_name}.toml").resolve()
        try:
            file_path.relative_to(page_dir.resolve())
        except ValueError:
            return _json_error("参数文件名无效。")

    if not file_path.exists():
        return _json_error("参数文件不存在。", status_code=404)

    try:
        payload = _load_toml_config(file_path)
    except Exception as exc:
        return _json_error(f"参数文件读取失败：{exc}", status_code=500)

    return APIResponseSuccess(data={"name": name, "source": source, "config": payload})


@router.post("/legacy_lora_configs/save")
async def save_legacy_lora_config(request: Request):
    try:
        payload = json.loads((await request.body()).decode("utf-8"))
    except json.JSONDecodeError:
        return _json_error("请求体不是合法 JSON。")
    if not isinstance(payload, dict):
        return _json_error("请求体必须是 JSON 对象。")

    page = str(payload.get("page", "") or "")
    name = str(payload.get("name", "") or "").strip()
    config = payload.get("config")
    if not isinstance(config, dict):
        return _json_error("缺少配置内容。")

    target_name = name or "latest"
    try:
        page_dir = get_legacy_lora_page_dir(page)
    except ValueError as exc:
        return _json_error(str(exc))

    safe_name = sanitize_saved_config_name(target_name)
    file_path = (page_dir / f"{safe_name}.toml").resolve()
    try:
        file_path.relative_to(page_dir.resolve())
    except ValueError:
        return _json_error("参数文件名无效。")

    _write_toml_config(file_path, config)
    return APIResponseSuccess(data={"name": file_path.stem})


@router.post("/saved_configs/save")
async def save_named_config(request: Request):
    try:
        payload = json.loads((await request.body()).decode("utf-8"))
    except json.JSONDecodeError:
        return _json_error("请求体不是合法 JSON。")
    if not isinstance(payload, dict):
        return _json_error("请求体必须是 JSON 对象。")

    name = payload.get("name")
    config = payload.get("config")
    if not isinstance(config, dict):
        return _json_error("缺少参数名称或配置内容。")

    try:
        file_path = get_saved_config_path(str(name or ""))
    except ValueError as exc:
        return _json_error(str(exc))

    safe_name = file_path.stem
    file_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return APIResponseSuccess(data={"name": safe_name})


@router.get("/saved_configs/list")
async def list_saved_configs() -> APIResponse:
    if not SAVED_CONFIGS_DIR.exists():
        return APIResponseSuccess(data={"configs": []})

    configs = sorted(
        (
            {
                "name": file.stem,
                "time": int(file.stat().st_mtime * 1000),
            }
            for file in SAVED_CONFIGS_DIR.glob("*.json")
            if file.is_file()
        ),
        key=lambda item: item["time"],
        reverse=True,
    )
    return APIResponseSuccess(data={"configs": configs})


@router.get("/saved_configs/load")
async def load_named_config(name: str):
    try:
        file_path = get_saved_config_path(name)
    except ValueError as exc:
        return _json_error(str(exc))

    if not file_path.exists():
        return _json_error("参数文件不存在。", status_code=404)

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _json_error("参数文件损坏，无法读取。", status_code=500)

    if not isinstance(payload, dict):
        return _json_error("参数文件格式无效。", status_code=500)

    return APIResponseSuccess(data=payload)


@router.get("/saved_configs/delete")
async def delete_named_config(name: str):
    try:
        file_path = get_saved_config_path(name)
    except ValueError as exc:
        return _json_error(str(exc))

    if not file_path.exists():
        return _json_error("参数文件不存在。", status_code=404)

    file_path.unlink()
    return APIResponseSuccess()


@router.post("/saved_configs/rename")
async def rename_saved_config(request: Request):
    try:
        payload = json.loads((await request.body()).decode("utf-8"))
    except json.JSONDecodeError:
        return _json_error("请求体不是合法 JSON。")
    if not isinstance(payload, dict):
        return _json_error("请求体必须是 JSON 对象。")

    old_name = str(payload.get("oldName", "") or "")
    new_name = str(payload.get("newName", "") or "")

    try:
        old_path = get_saved_config_path(old_name)
        new_path = get_saved_config_path(new_name)
    except ValueError as exc:
        return _json_error(str(exc))

    if not old_path.exists():
        return _json_error("原参数文件不存在。", status_code=404)

    if old_path == new_path:
        return APIResponseSuccess(data={"name": new_path.stem})

    if new_path.exists():
        return _json_error("新名称已存在，请换一个名称。", status_code=409)

    old_path.rename(new_path)
    return APIResponseSuccess(data={"name": new_path.stem})


@router.api_route("/local/task_history", methods=["GET", "POST", "DELETE"])
async def manage_local_task_history(request: Request):
    if request.method == "GET":
        return APIResponseSuccess(data={"tasks": _load_local_task_history()})

    if request.method == "DELETE":
        if TASK_HISTORY_FILE.exists():
            TASK_HISTORY_FILE.unlink()
        return APIResponseSuccess()

    try:
        payload = json.loads((await request.body()).decode("utf-8"))
    except json.JSONDecodeError:
        return _json_error("请求体不是合法 JSON。")
    if not isinstance(payload, dict):
        return _json_error("请求体必须是 JSON 对象。")

    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        return _json_error("tasks 必须是数组。")

    _save_local_task_history([item for item in tasks if isinstance(item, dict)])
    return APIResponseSuccess()


@router.delete("/local/task_history/{task_id}")
async def delete_local_task_history_item(task_id: str):
    task_id = str(task_id or "").strip()
    if not task_id:
        return _json_error("task_id 不能为空。")

    tasks = _load_local_task_history()
    remaining = [item for item in tasks if str(item.get("id", "") or "") != task_id]
    deleted = len(tasks) - len(remaining)
    if deleted <= 0:
        return _json_error("任务历史不存在。", status_code=404)

    _save_local_task_history(remaining)
    return APIResponseSuccess(data={"deleted": deleted, "task_id": task_id})
