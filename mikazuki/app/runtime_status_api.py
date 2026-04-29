from __future__ import annotations

import sys

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from mikazuki.app.models import APIResponse, APIResponseFail, APIResponseSuccess
from mikazuki.log import log
from mikazuki.tasks import tm
from mikazuki.utils.aesthetic_infer_runtime import aesthetic_infer_manager
from mikazuki.utils.aesthetic_runtime import (
    build_aesthetic_runtime_payload,
    start_aesthetic_dependency_install,
)
from mikazuki.utils.backend_status import read_backend_status, request_backend_restart
from mikazuki.utils.devices import get_xformers_status, printable_devices
from mikazuki.utils.direct_trainers import build_newbie_runtime_payload
from mikazuki.utils.runtime_dependencies import build_runtime_status_payload
from mikazuki.utils.runtime_mode import infer_runtime_environment_name
from mikazuki.utils.yolo_runtime import (
    build_yolo_runtime_payload,
    start_yolo_dependency_install,
)


router = APIRouter()


def _fallback_runtime_status_payload(error_message: str) -> dict:
    return {
        "environment": infer_runtime_environment_name(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "required_ready": False,
        "packages": {},
        "inspection_errors": [str(error_message)],
        "error": str(error_message),
    }


def _safe_build_runtime_status_payload() -> dict:
    try:
        payload = build_runtime_status_payload()
    except Exception as exc:
        log.exception("Failed to build runtime status payload for /graphic_cards")
        return _fallback_runtime_status_payload(f"Runtime inspection failed: {exc}")

    if not isinstance(payload, dict):
        return _fallback_runtime_status_payload("Runtime inspection returned an invalid payload.")
    return payload


def _safe_get_xformers_status() -> dict:
    fallback = {
        "version": None,
        "installed": False,
        "supported": False,
        "reason": "xformers status is unavailable.",
    }
    try:
        status = get_xformers_status()
    except Exception as exc:
        log.exception("Failed to probe xformers status for /graphic_cards")
        fallback["reason"] = f"xformers probe failed: {exc}"
        return fallback

    if not isinstance(status, dict):
        fallback["reason"] = "xformers probe returned an invalid payload."
        return fallback

    return {
        "version": status.get("version"),
        "installed": bool(status.get("installed", False)),
        "supported": bool(status.get("supported", False)),
        "reason": str(status.get("reason", "") or ""),
    }


@router.get("/tasks", response_model_exclude_none=True)
async def get_tasks() -> APIResponse:
    return APIResponseSuccess(data={
        "tasks": tm.dump()
    })


@router.get("/backend/status", response_model_exclude_none=True)
async def get_backend_status() -> APIResponse:
    return APIResponseSuccess(data=read_backend_status())


@router.post("/backend/restart", response_model_exclude_none=True)
async def restart_backend() -> APIResponse:
    ok, message = request_backend_restart()
    if not ok:
        return APIResponseFail(message=message)
    return APIResponseSuccess(message=message, data={"status": read_backend_status()})


@router.get("/yolo/runtime_status", response_model_exclude_none=True)
async def get_yolo_runtime_status() -> APIResponse:
    return APIResponseSuccess(data=build_yolo_runtime_payload())


@router.get("/newbie/runtime_status", response_model_exclude_none=True)
async def get_newbie_runtime_status() -> APIResponse:
    return APIResponseSuccess(data=build_newbie_runtime_payload())


@router.post("/yolo/install_dependencies", response_model_exclude_none=True)
async def install_yolo_dependencies() -> APIResponse:
    ok, message, payload = start_yolo_dependency_install()
    if not ok:
        return APIResponseFail(message=message, data=payload)
    return APIResponseSuccess(message=message, data=payload)


@router.get("/aesthetic/runtime_status", response_model_exclude_none=True)
async def get_aesthetic_runtime_status() -> APIResponse:
    return APIResponseSuccess(data=build_aesthetic_runtime_payload())


@router.post("/aesthetic/install_dependencies", response_model_exclude_none=True)
async def install_aesthetic_dependencies() -> APIResponse:
    ok, message, payload = start_aesthetic_dependency_install()
    if not ok:
        return APIResponseFail(message=message, data=payload)
    return APIResponseSuccess(message=message, data=payload)


@router.get("/aesthetic_infer/runtime_status", response_model_exclude_none=True)
async def get_aesthetic_infer_runtime_status() -> APIResponse:
    return APIResponseSuccess(data=aesthetic_infer_manager.get_runtime_payload())


@router.get("/aesthetic_infer/status", response_model_exclude_none=True)
async def get_aesthetic_infer_status() -> APIResponse:
    return APIResponseSuccess(data=aesthetic_infer_manager.get_status())


@router.get("/aesthetic_infer/logs", response_model_exclude_none=True)
async def get_aesthetic_infer_logs(since_id: int = 0, limit: int = 300) -> APIResponse:
    return APIResponseSuccess(data=aesthetic_infer_manager.get_logs(since_id=since_id, limit=limit))


@router.post("/aesthetic_infer/start", response_model_exclude_none=True)
async def start_aesthetic_infer(request: Request) -> APIResponse:
    try:
        payload = await request.json()
    except Exception:
        return APIResponseFail(message="请求体不是合法的 JSON。")
    if not isinstance(payload, dict):
        return APIResponseFail(message="请求体必须是 JSON 对象。")
    ok, message, data = aesthetic_infer_manager.start(payload)
    if not ok:
        return APIResponseFail(message=message, data=data)
    return APIResponseSuccess(message=message, data=data)


@router.post("/aesthetic_infer/stop", response_model_exclude_none=True)
async def stop_aesthetic_infer() -> APIResponse:
    ok, message, data = aesthetic_infer_manager.stop()
    if not ok:
        return APIResponseFail(message=message, data=data)
    return APIResponseSuccess(message=message, data=data)


@router.get("/aesthetic_infer/results", response_model_exclude_none=True)
async def get_aesthetic_infer_results(
    output_dir: str = "",
    page: int = 1,
    page_size: int = 24,
    q: str = "",
    special_filter: str = "all",
    sort_by: str = "",
    sort_order: str = "desc",
) -> APIResponse:
    ok, message, data = aesthetic_infer_manager.get_results(
        output_dir_raw=output_dir,
        page=page,
        page_size=page_size,
        keyword=q,
        special_filter=special_filter,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    if not ok:
        return APIResponseFail(message=message)
    return APIResponseSuccess(data=data)


@router.get("/aesthetic_infer/file")
async def get_aesthetic_infer_file(output_dir: str, kind: str):
    path = aesthetic_infer_manager.resolve_output_file(output_dir, kind)
    if path is None:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(path))


@router.get("/aesthetic_infer/image")
async def get_aesthetic_infer_image(path: str):
    resolved_path = aesthetic_infer_manager.resolve_image_path(path)
    if resolved_path is None:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(resolved_path))


@router.get("/tasks/terminate/{task_id}", response_model_exclude_none=True)
async def terminate_task(task_id: str):
    result = tm.request_terminate_task(task_id)
    if result == "not-found":
        return APIResponseFail(message="Task not found")
    if result == "already-requested":
        return APIResponseSuccess(message="Stop request is already in progress / 停止请求已在处理中。")
    if result == "already-stopped":
        return APIResponseSuccess(message="Task is already stopped / 任务已经停止。")
    return APIResponseSuccess(message="Stop request accepted / 已接受停止请求。")


@router.get("/task_output/{task_id}", response_model_exclude_none=True)
async def get_task_output(task_id: str, tail: int = 50) -> APIResponse:
    task = tm.tasks.get(task_id)
    if task is None:
        return APIResponseFail(message="Task not found")

    safe_tail = max(1, min(int(tail or 50), 1000))
    lines, total = task.get_output_snapshot(tail=safe_tail)
    return APIResponseSuccess(data={
        "lines": lines,
        "total": total,
    })


@router.get("/system_monitor")
async def get_system_monitor() -> APIResponse:
    result: dict = {"gpu": {"available": False}, "cpu": {}, "ram": {}}
    try:
        from mikazuki.utils.nvidia_smi import query_gpu_memory, query_gpu_metrics

        mem_info = query_gpu_memory()
        if mem_info.get("ok") and mem_info.get("gpus"):
            metrics_info = query_gpu_metrics()
            metrics_map = {}
            if metrics_info.get("ok"):
                for gm in metrics_info.get("gpus", []):
                    metrics_map[str(gm.get("index", ""))] = gm
            gpus = []
            for g in mem_info["gpus"]:
                idx = str(g.get("index", ""))
                total_mb = g.get("memory_total_mb") or 0
                used_mb = g.get("memory_used_mb") or 0
                free_mb = g.get("memory_free_mb") or 0
                pct = round(used_mb / total_mb * 100, 1) if total_mb > 0 else 0
                gm = metrics_map.get(idx, {})
                gpu_name = f"GPU {idx}"
                try:
                    import torch

                    if torch.cuda.is_available() and int(idx) < torch.cuda.device_count():
                        gpu_name = torch.cuda.get_device_properties(int(idx)).name
                except Exception:
                    pass
                gpus.append({
                    "index": idx,
                    "name": gpu_name,
                    "total_mb": round(total_mb),
                    "used_mb": round(used_mb),
                    "free_mb": round(free_mb),
                    "utilization_pct": pct,
                    "temperature_c": gm.get("temperature_c"),
                    "power_draw_w": gm.get("power_draw_w"),
                })
            result["gpu"] = {"available": True, "gpus": gpus}
    except Exception:
        pass

    try:
        import psutil

        result["cpu"] = {"percent": psutil.cpu_percent(interval=0), "count": psutil.cpu_count()}
        vm = psutil.virtual_memory()
        result["ram"] = {"total_mb": round(vm.total / 1048576), "used_mb": round(vm.used / 1048576), "percent": vm.percent}
    except Exception:
        pass
    return APIResponseSuccess(data=result)


@router.get("/gpu_status")
async def get_gpu_status() -> APIResponse:
    try:
        import torch

        if not torch.cuda.is_available():
            return APIResponseSuccess(data={"available": False})
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_mb": round(total / 1048576),
                    "allocated_mb": round(allocated / 1048576),
                    "reserved_mb": round(reserved / 1048576),
                    "utilization_pct": round(allocated / total * 100, 1) if total > 0 else 0,
                }
            )
        return APIResponseSuccess(data={"available": True, "gpus": gpus})
    except Exception as exc:
        return APIResponseSuccess(data={"available": False, "error": str(exc)})


@router.get("/graphic_cards")
async def list_avaliable_cards() -> APIResponse:
    runtime_info = _safe_build_runtime_status_payload()
    xformers_info = _safe_get_xformers_status()
    cards = list(printable_devices)
    if not cards:
        return APIResponse(
            status="pending",
            message="GPU detection is still in progress.",
            data={
                "cards": [],
                "xformers": xformers_info,
                "runtime": runtime_info,
            },
        )

    return APIResponseSuccess(data={
        "cards": cards,
        "xformers": xformers_info,
        "runtime": runtime_info,
    })
