from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from mikazuki.app.local_files import (
    BUILTIN_PICKER_ROOTS,
    LOGS_ROOT,
    MODEL_FILE_EXTENSIONS,
    PREVIEW_IMAGE_EXTENSIONS,
    REPO_ROOT,
    SAMPLE_OUTPUT_DIR,
    open_directory_in_shell,
    require_safe_child_name,
)
from mikazuki.app.models import APIResponse, APIResponseFail, APIResponseSuccess
from mikazuki.utils.tk_window import open_directory_selector, open_file_selector


router = APIRouter()


def _json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "error", "message": message})


@router.get("/pick_file")
async def pick_file(picker_type: str):
    if picker_type in {"folder", "output-folder"}:
        coro = asyncio.to_thread(open_directory_selector, "")
    elif picker_type in {"model-file", "output-model-file"}:
        file_types = [("checkpoints", "*.safetensors;*.ckpt;*.pt"), ("all files", "*.*")]
        coro = asyncio.to_thread(open_file_selector, "", "Select file", file_types)
    elif picker_type == "text-file":
        file_types = [("text files", "*.txt;*.text;*.prompt"), ("all files", "*.*")]
        coro = asyncio.to_thread(open_file_selector, "", "Select prompt file", file_types)
    else:
        return APIResponseFail(message="Invalid picker type")

    result = await coro
    if result == "":
        return APIResponseFail(message="用户取消选择")

    return APIResponseSuccess(data={
        "path": result
    })


@router.get("/get_files")
async def get_files(pick_type) -> APIResponse:
    pick_preset = {
        "model-file": {
            "type": "file",
            "path": "./sd-models",
            "filter": "(.safetensors|.ckpt|.pt)"
        },
        "model-saved-file": {
            "type": "file",
            "path": "./output",
            "filter": "(.safetensors|.ckpt|.pt)"
        },
        "train-dir": {
            "type": "folder",
            "path": "./train",
            "filter": None
        },
    }

    folder_blacklist = [".ipynb_checkpoints", ".DS_Store"]

    def list_path_or_files(preset_info):
        path = Path(preset_info["path"])
        file_type = preset_info["type"]
        regex_filter = preset_info["filter"]
        result_list = []

        if not path.exists():
            return result_list

        if file_type == "file":
            if regex_filter:
                pattern = re.compile(regex_filter)
                files = [f for f in path.glob("**/*") if f.is_file() and pattern.search(f.name)]
            else:
                files = [f for f in path.glob("**/*") if f.is_file()]
            for file in files:
                result_list.append({
                    "path": str(file.resolve().absolute()).replace("\\", "/"),
                    "name": file.name,
                    "size": f"{round(file.stat().st_size / (1024**3),2)} GB"
                })
        elif file_type == "folder":
            folders = [f for f in path.iterdir() if f.is_dir()]
            for folder in folders:
                if folder.name in folder_blacklist:
                    continue
                result_list.append({
                    "path": str(folder.resolve().absolute()).replace("\\", "/"),
                    "name": folder.name,
                    "size": 0
                })

        return result_list

    if pick_type not in pick_preset:
        return APIResponseFail(message="Invalid request")

    dirs = list_path_or_files(pick_preset[pick_type])
    return APIResponseSuccess(data={
        "files": dirs
    })


@router.get("/builtin_picker")
async def get_builtin_picker(picker_type: str = "file") -> APIResponse:
    root_path = BUILTIN_PICKER_ROOTS.get(picker_type, BUILTIN_PICKER_ROOTS["file"])
    try:
        root_label = str(root_path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        root_label = str(root_path)

    items: list[str] = []
    if root_path.exists():
        entries = list(root_path.iterdir())
        if picker_type in {"folder", "output-folder"}:
            items = sorted(
                [entry.name for entry in entries if entry.is_dir() and not entry.name.startswith(".")],
                key=str.lower,
            )
        else:
            items = sorted(
                [
                    entry.name
                    for entry in entries
                    if entry.is_file()
                    and not entry.name.startswith(".")
                    and entry.suffix.lower() in MODEL_FILE_EXTENSIONS
                ],
                key=str.lower,
            )

    return APIResponseSuccess(data={"rootLabel": root_label, "items": items})


@router.get("/log_dirs")
async def get_log_dirs() -> APIResponse:
    if not LOGS_ROOT.exists():
        return APIResponseSuccess(data={"dirs": []})

    dirs = sorted(
        (
            {
                "name": directory.name,
                "time": int(directory.stat().st_mtime * 1000),
                "hasEvents": any(child.name.startswith("events.out") for child in directory.iterdir()),
            }
            for directory in LOGS_ROOT.iterdir()
            if directory.is_dir()
        ),
        key=lambda item: item["time"],
        reverse=True,
    )
    return APIResponseSuccess(data={"dirs": dirs})


@router.get("/log_detail")
async def get_log_detail(dir: str):
    if not dir:
        return _json_error("缺少目录名。")

    target_dir = (LOGS_ROOT / dir).resolve()
    try:
        target_dir.relative_to(LOGS_ROOT.resolve())
    except ValueError:
        return _json_error("目录名无效。")

    if not target_dir.exists() or not target_dir.is_dir():
        return _json_error("日志目录不存在。", status_code=404)

    files = [
        {
            "name": item.name,
            "size": item.stat().st_size,
            "time": int(item.stat().st_mtime * 1000),
        }
        for item in sorted(target_dir.iterdir(), key=lambda child: child.name.lower())
        if item.is_file()
    ]
    return APIResponseSuccess(data={"dir": dir, "files": files})


@router.get("/local/sample_images")
async def get_sample_images() -> APIResponse:
    if not SAMPLE_OUTPUT_DIR.exists() or not SAMPLE_OUTPUT_DIR.is_dir():
        return APIResponseSuccess(data={"images": [], "total": 0})

    images = sorted(
        (
            {
                "name": file.name,
                "path": str(file.resolve()).replace("\\", "/"),
                "mtime": int(file.stat().st_mtime * 1000),
            }
            for file in SAMPLE_OUTPUT_DIR.iterdir()
            if file.is_file() and file.suffix.lower() in PREVIEW_IMAGE_EXTENSIONS
        ),
        key=lambda item: item["mtime"],
        reverse=True,
    )
    return APIResponseSuccess(data={"images": images, "total": len(images)})


@router.get("/local/sample_file")
async def get_sample_file(name: str):
    try:
        safe_name = require_safe_child_name(name, label="file name")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target_file = SAMPLE_OUTPUT_DIR / safe_name
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    if target_file.suffix.lower() not in PREVIEW_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    return FileResponse(target_file)


@router.post("/local/open_folder")
async def open_local_folder(request: Request):
    try:
        payload = json.loads((await request.body()).decode("utf-8"))
    except json.JSONDecodeError:
        return _json_error("请求体不是合法 JSON。")
    if not isinstance(payload, dict):
        return _json_error("请求体必须是 JSON 对象。")

    raw_folder = str(payload.get("folder", "") or "").strip() or "output"
    target_dir = Path(raw_folder).expanduser()
    if not target_dir.is_absolute():
        target_dir = (REPO_ROOT / target_dir).resolve()
    else:
        target_dir = target_dir.resolve()

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(open_directory_in_shell, target_dir)
    except Exception as exc:
        return _json_error(f"打开目录失败：{exc}", status_code=500)
    return APIResponseSuccess()
