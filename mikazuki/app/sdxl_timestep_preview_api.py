from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from mikazuki import launch_utils
from mikazuki.app.models import APIResponse, APIResponseFail, APIResponseSuccess


router = APIRouter()

REPO_ROOT = launch_utils.base_dir_path()
PREVIEW_SCRIPT = REPO_ROOT / "tools" / "preview_sdxl_timestep_sampling.py"
PREVIEW_OUTPUTS = {
    "sdxl": REPO_ROOT / "tmp" / "sdxl_timestep_sampling_preview_live.png",
    "anima": REPO_ROOT / "tmp" / "anima_timestep_sampling_preview_live.png",
}
PREVIEW_STATUS_FILE = REPO_ROOT / "tmp" / "timestep_preview_status.json"
_REQUIRED_MODULES = ("matplotlib", "tkinter")


def _write_preview_status(status: str, detail: str = "", *, preview_mode: str = "sdxl") -> None:
    PREVIEW_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": str(status or "unknown"),
        "detail": str(detail or ""),
        "preview_mode": str(preview_mode or "sdxl"),
        "updated_at": int(time.time() * 1000),
    }
    with open(PREVIEW_STATUS_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_preview_status() -> dict:
    if not PREVIEW_STATUS_FILE.exists():
        return {
            "status": "idle",
            "detail": "",
            "preview_mode": "sdxl",
            "updated_at": 0,
        }

    try:
        with open(PREVIEW_STATUS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("status file payload must be a dict")
        return {
            "status": str(data.get("status", "idle") or "idle"),
            "detail": str(data.get("detail", "") or ""),
            "preview_mode": str(data.get("preview_mode", "sdxl") or "sdxl"),
            "updated_at": int(data.get("updated_at", 0) or 0),
        }
    except Exception:
        return {
            "status": "idle",
            "detail": "",
            "preview_mode": "sdxl",
            "updated_at": 0,
        }


def _resolve_launch_status_for_mode(preview_mode: str, preview_output: Path) -> dict:
    preview_status = _read_preview_status()
    if preview_status.get("preview_mode") == preview_mode:
        return preview_status

    if preview_output.exists() and preview_output.is_file():
        return {
            "status": "ready",
            "detail": "当前预览图可用。",
            "preview_mode": preview_mode,
            "updated_at": int(preview_output.stat().st_mtime * 1000),
        }

    return {
        "status": "idle",
        "detail": "",
        "preview_mode": preview_mode,
        "updated_at": 0,
    }


def _probe_python_modules(python_exe: str | Path) -> bool:
    probe = (
        "import importlib.util, json; "
        f"mods={list(_REQUIRED_MODULES)!r}; "
        "print(json.dumps({m: importlib.util.find_spec(m) is not None for m in mods}))"
    )
    try:
        result = subprocess.run(
            [str(python_exe), "-c", probe],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return False

    if result.returncode != 0:
        return False

    try:
        payload = json.loads((result.stdout or "").strip() or "{}")
    except json.JSONDecodeError:
        return False

    return all(bool(payload.get(module_name, False)) for module_name in _REQUIRED_MODULES)


def _iter_preview_python_candidates():
    seen: set[str] = set()

    def register(candidate: str | Path | None):
        if candidate is None:
            return
        candidate_path = Path(candidate)
        try:
            resolved = candidate_path.resolve()
        except Exception:
            resolved = candidate_path
        if not resolved.exists():
            return
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        yield resolved

    current_python = Path(sys.executable)
    current_pythonw = current_python.with_name("pythonw.exe")
    for path in (
        current_python,
        current_pythonw,
        shutil.which("python"),
        shutil.which("pythonw"),
        REPO_ROOT / "env" / "python" / "python.exe",
        REPO_ROOT / "env" / "python" / "pythonw.exe",
        REPO_ROOT / "python" / "python.exe",
        REPO_ROOT / "python" / "pythonw.exe",
    ):
        yield from register(path)


def _resolve_preview_python() -> tuple[Path | None, Path | None]:
    for candidate in _iter_preview_python_candidates():
        probe_candidate = candidate.with_name("python.exe") if candidate.name.lower() == "pythonw.exe" else candidate

        if not probe_candidate.exists():
            continue
        if not _probe_python_modules(probe_candidate):
            continue

        launcher_candidate = probe_candidate.with_name("pythonw.exe")
        if launcher_candidate.exists():
            return probe_candidate, launcher_candidate
        return probe_candidate, probe_candidate

    return None, None


def _normalize_preview_mode(payload: dict) -> str:
    preview_mode = str(payload.get("preview_mode", "sdxl") or "sdxl").strip().lower()
    if preview_mode not in PREVIEW_OUTPUTS:
        preview_mode = "sdxl"
    return preview_mode


def _parse_common_payload(payload: dict) -> dict:
    preview_mode = _normalize_preview_mode(payload)

    def parse_int(key: str, default: int) -> int:
        try:
            return int(payload.get(key, default))
        except (TypeError, ValueError):
            return default

    def parse_float(key: str, default: float) -> float:
        try:
            return float(payload.get(key, default))
        except (TypeError, ValueError):
            return default

    min_timestep = max(0, parse_int("min_timestep", 0))
    max_timestep = max(1, parse_int("max_timestep", 1000))
    if max_timestep <= min_timestep:
        max_timestep = min_timestep + 1

    return {
        "preview_mode": preview_mode,
        "min_timestep": min_timestep,
        "max_timestep": max_timestep,
        "scale": parse_float("scale", 1.0),
        "shift": max(0.01, parse_float("shift", 1.0)),
        "weight_scale": parse_float("weight_scale", 1.0),
        "weight_shift": max(0.01, parse_float("weight_shift", 1.0)),
        "logit_mean": parse_float("logit_mean", 0.0),
        "logit_std": max(0.1, parse_float("logit_std", 1.0)),
        "mode_scale": parse_float("mode_scale", 1.29),
        "width": max(256, parse_int("width", 1024)),
        "height": max(256, parse_int("height", 1024)),
    }


def _parse_preview_payload(payload: dict) -> dict:
    options = _parse_common_payload(payload)

    if options["preview_mode"] == "anima":
        mode = str(payload.get("mode", "shift") or "shift").strip().lower()
        if mode not in {"sigma", "uniform", "sigmoid", "shift", "flux_shift"}:
            mode = "shift"

        weight_mode = str(payload.get("weight_mode", "uniform") or "uniform").strip().lower()
        if weight_mode not in {"uniform", "sigma_sqrt", "cosmap", "none", "logit_normal", "mode"}:
            weight_mode = "uniform"
    else:
        mode = str(payload.get("mode", "uniform") or "uniform").strip().lower()
        if mode not in {"uniform", "sigmoid", "shift"}:
            mode = "uniform"

        weight_mode = str(payload.get("weight_mode", "none") or "none").strip().lower()
        if weight_mode not in {"none", "linear", "cosine", "sigmoid", "shift"}:
            weight_mode = "none"

    options.update({
        "mode": mode,
        "weight_mode": weight_mode,
    })
    return options


def _build_output_image_url(preview_mode: str) -> str:
    return f"/api/timestep_preview/{preview_mode}/image"


def _get_preview_output(preview_mode: str) -> Path:
    return PREVIEW_OUTPUTS.get(preview_mode, PREVIEW_OUTPUTS["sdxl"])


def _build_preview_command(python_exe: Path, preview_output: Path, options: dict, *, save_only: bool) -> list[str]:
    command = [
        str(python_exe),
        str(PREVIEW_SCRIPT),
        "--output",
        str(preview_output),
        "--preview_mode",
        options["preview_mode"],
        "--mode",
        options["mode"],
        "--scale",
        str(options["scale"]),
        "--shift",
        str(options["shift"]),
        "--weight_mode",
        options["weight_mode"],
        "--weight_scale",
        str(options["weight_scale"]),
        "--weight_shift",
        str(options["weight_shift"]),
        "--logit_mean",
        str(options["logit_mean"]),
        "--logit_std",
        str(options["logit_std"]),
        "--mode_scale",
        str(options["mode_scale"]),
        "--min_timestep",
        str(options["min_timestep"]),
        "--max_timestep",
        str(options["max_timestep"]),
        "--width",
        str(options["width"]),
        "--height",
        str(options["height"]),
    ]
    if save_only:
        command.append("--save-only")
    return command


@router.post("/timestep_preview/open")
async def open_timestep_preview(request: Request) -> APIResponse:
    try:
        payload = json.loads((await request.body()).decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return APIResponseFail(message="请求体不是合法 JSON。")

    if not isinstance(payload, dict):
        return APIResponseFail(message="请求体必须是 JSON 对象。")

    if not PREVIEW_SCRIPT.exists():
        return APIResponseFail(message=f"预览脚本不存在：{PREVIEW_SCRIPT}")

    probe_python, launch_python = _resolve_preview_python()
    if probe_python is None or launch_python is None:
        return APIResponseFail(message="未找到同时支持 matplotlib 与 tkinter 的 Python 运行时，无法打开交互式预览窗口。")

    options = _parse_preview_payload(payload)
    preview_output = _get_preview_output(options["preview_mode"])
    preview_output.parent.mkdir(parents=True, exist_ok=True)
    previous_mtime = preview_output.stat().st_mtime if preview_output.exists() and preview_output.is_file() else 0.0

    save_command = _build_preview_command(probe_python, preview_output, options, save_only=True)
    launch_command = _build_preview_command(launch_python, preview_output, options, save_only=False)

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        _write_preview_status("rendering", "正在先生成预览快照。", preview_mode=options["preview_mode"])
        render_result = subprocess.run(
            save_command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if render_result.returncode != 0:
            detail = (render_result.stderr or render_result.stdout or "").strip()
            if not detail:
                detail = "预览脚本返回非零退出码。"
            _write_preview_status("failed", f"生成预览快照失败：{detail}", preview_mode=options["preview_mode"])
            return APIResponseFail(message=f"生成预览快照失败：{detail}")

        _write_preview_status("launching", "预览快照已生成，正在启动交互式预览窗口。", preview_mode=options["preview_mode"])
        subprocess.Popen(
            launch_command,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired:
        _write_preview_status("failed", "生成预览快照超时，请检查 matplotlib 运行环境。", preview_mode=options["preview_mode"])
        return APIResponseFail(message="生成预览快照超时，请检查 matplotlib 运行环境。")
    except Exception as exc:
        _write_preview_status("failed", f"打开预览工具失败：{exc}", preview_mode=options["preview_mode"])
        return APIResponseFail(message=f"打开预览工具失败：{exc}")

    for _ in range(20):
        if preview_output.exists() and preview_output.is_file() and preview_output.stat().st_mtime > previous_mtime:
            _write_preview_status("ready", "预览图已更新。", preview_mode=options["preview_mode"])
            break
        time.sleep(0.2)
    else:
        _write_preview_status(
            "pending",
            "预览窗口已启动，但快照暂未更新。请检查是否有新的 matplotlib 预览窗口被系统隐藏或阻挡。",
            preview_mode=options["preview_mode"],
        )

    return APIResponseSuccess(
        data={
            "output_path": str(preview_output),
            "image_url": _build_output_image_url(options["preview_mode"]),
            "python": str(probe_python),
            "status_file": str(PREVIEW_STATUS_FILE),
            **options,
        }
    )


@router.get("/timestep_preview/status")
async def get_timestep_preview_status(preview_mode: str = "sdxl") -> APIResponse:
    preview_mode = preview_mode if preview_mode in PREVIEW_OUTPUTS else "sdxl"
    preview_output = _get_preview_output(preview_mode)
    exists = preview_output.exists() and preview_output.is_file()
    return APIResponseSuccess(
        data={
            "preview_mode": preview_mode,
            "exists": exists,
            "output_path": str(preview_output),
            "image_url": _build_output_image_url(preview_mode) if exists else None,
            "mtime": int(preview_output.stat().st_mtime * 1000) if exists else None,
            "size": preview_output.stat().st_size if exists else 0,
            "launch_status": _resolve_launch_status_for_mode(preview_mode, preview_output),
        }
    )


@router.get("/timestep_preview/{preview_mode}/image")
async def get_timestep_preview_image(preview_mode: str):
    if preview_mode not in PREVIEW_OUTPUTS:
        raise HTTPException(status_code=404, detail="Preview mode not found")

    preview_output = _get_preview_output(preview_mode)
    if not preview_output.exists() or not preview_output.is_file():
        raise HTTPException(status_code=404, detail="Preview image not found")
    return FileResponse(preview_output)


@router.post("/sdxl/timestep_preview/open")
async def open_sdxl_timestep_preview_legacy(request: Request) -> APIResponse:
    return await open_timestep_preview(request)


@router.get("/sdxl/timestep_preview/status")
async def get_sdxl_timestep_preview_status_legacy() -> APIResponse:
    return await get_timestep_preview_status("sdxl")


@router.get("/sdxl/timestep_preview/image")
async def get_sdxl_timestep_preview_image_legacy():
    return await get_timestep_preview_image("sdxl")
