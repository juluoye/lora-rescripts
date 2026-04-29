from __future__ import annotations

from fastapi import APIRouter, Request

from mikazuki.app.training_preview_api import build_router as build_training_preview_router
from mikazuki.app.training_prompt_utils import parse_boolish
from mikazuki.app.training_run_flow import handle_training_run_request
from mikazuki.app.training_ui_overrides import apply_training_ui_overrides


router = APIRouter()
router.include_router(
    build_training_preview_router(
        apply_training_ui_overrides=apply_training_ui_overrides,
        parse_boolish=parse_boolish,
    )
)


@router.post("/run")
async def create_toml_file(request: Request):
    return await handle_training_run_request(request)
