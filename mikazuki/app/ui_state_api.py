from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from mikazuki.app.config import app_config
from mikazuki.app.models import APIResponse, APIResponseSuccess
from mikazuki.app.ui_state import (
    SAVED_CONFIGS_DIR,
    TASK_HISTORY_FILE,
    ensure_ui_state_root,
    get_saved_config_path,
)
from mikazuki.utils.frontend_profiles import resolve_frontend_profile_id


router = APIRouter()


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
