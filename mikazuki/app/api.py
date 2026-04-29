from __future__ import annotations

from fastapi import APIRouter

from mikazuki.app.local_files_api import router as local_files_router
from mikazuki.app.plugins_api import router as plugins_router
from mikazuki.app.runtime_status_api import router as runtime_status_router
from mikazuki.app.schema_registry_api import router as schema_registry_router
from mikazuki.app.script_runner_api import router as script_runner_router
from mikazuki.app.training_helpers_api import router as training_helpers_router
from mikazuki.app.training_run_api import router as training_run_router
from mikazuki.app.ui_state_api import router as ui_state_router


router = APIRouter()
router.include_router(local_files_router)
router.include_router(plugins_router)
router.include_router(runtime_status_router)
router.include_router(schema_registry_router)
router.include_router(script_runner_router)
router.include_router(training_helpers_router)
router.include_router(training_run_router)
router.include_router(ui_state_router)
