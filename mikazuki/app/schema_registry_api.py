from __future__ import annotations

import os

from fastapi import APIRouter

from mikazuki.app.models import APIResponse, APIResponseSuccess
from mikazuki.app.tooling_registry import (
    AVAILABLE_PRESETS,
    AVAILABLE_SCHEMAS,
    AVAILABLE_SCRIPTS,
    SCRIPT_POSITIONAL_ARGS,
    load_presets,
    load_schemas,
)
from mikazuki.log import log


router = APIRouter()


@router.get("/schemas/hashes")
async def list_schema_hashes() -> APIResponse:
    await load_schemas()
    return APIResponseSuccess(data={
        "schemas": [
            {
                "name": schema["name"],
                "hash": schema["hash"],
            }
            for schema in AVAILABLE_SCHEMAS
        ]
    })


@router.get("/schemas/all")
async def get_all_schemas() -> APIResponse:
    await load_schemas()
    return APIResponseSuccess(data={"schemas": AVAILABLE_SCHEMAS})


@router.get("/presets")
async def get_presets() -> APIResponse:
    if os.environ.get("MIKAZUKI_SCHEMA_HOT_RELOAD", "0") == "1":
        log.info("Hot reloading presets")
        await load_presets()
    return APIResponseSuccess(data={"presets": AVAILABLE_PRESETS})


@router.get("/scripts")
async def get_available_scripts() -> APIResponse:
    return APIResponseSuccess(data={
        "scripts": [
            {
                "name": script_name,
                "positional_args": SCRIPT_POSITIONAL_ARGS.get(script_name, []),
                "category": script_name.split("/", 1)[0] if "/" in script_name else "misc",
            }
            for script_name in AVAILABLE_SCRIPTS
        ]
    })
