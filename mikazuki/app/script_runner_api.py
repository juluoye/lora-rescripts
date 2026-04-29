from __future__ import annotations

import asyncio
import json

import mikazuki.process as process
from mikazuki import launch_utils
from mikazuki.app.models import APIResponseFail, APIResponseSuccess
from mikazuki.app.tooling_registry import (
    AVAILABLE_SCRIPTS,
    SCRIPT_POSITIONAL_ARGS,
    resolve_script_path,
)
from mikazuki.log import log
from mikazuki.tasks import tm
from fastapi import APIRouter, Request


router = APIRouter()


@router.post("/run_script")
async def run_script(request: Request):
    paras = await request.body()
    payload = json.loads(paras.decode("utf-8"))
    script_name = payload["script_name"]
    if script_name not in AVAILABLE_SCRIPTS:
        return APIResponseFail(message="Script not found")
    del payload["script_name"]

    repo_root = launch_utils.base_dir_path()
    script_path = resolve_script_path(script_name)
    if script_path is None:
        return APIResponseFail(message=f"Script path not found: {script_name}")

    script_path, script_env = process.prepare_python_script(script_path)
    cmd = [str(launch_utils.python_bin), str(process.get_script_runner_path()), str(script_path)]

    positional_args = SCRIPT_POSITIONAL_ARGS.get(script_name, [])
    for arg_name in positional_args:
        value = payload.pop(arg_name, None)
        if value is not None and value != "":
            cmd.append(str(value))

    for key, value in payload.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            if len(value) > 0:
                cmd.append(f"--{key}")
                cmd.extend(str(item) for item in value)
        else:
            if value is None or value == "":
                continue
            cmd.append(f"--{key}")
            cmd.append(str(value))

    task = tm.create_task(cmd, script_env, cwd=str(repo_root))
    if not task:
        return APIResponseFail(message="Cannot create script task / 无法创建脚本任务")

    def _run_script_task():
        try:
            if not task.execute():
                log.info(f"Script {script_name} start was cancelled before process launch")
                return
            result = task.communicate()
            if result.returncode != 0:
                log.warning(f"Script {script_name} exited with code {result.returncode}")
            else:
                log.info(f"Script {script_name} finished successfully")
        except Exception as exc:
            log.error(f"Script {script_name} failed: {exc}")

    asyncio.create_task(asyncio.to_thread(_run_script_task))
    return APIResponseSuccess(data={"task_id": task.task_id})
