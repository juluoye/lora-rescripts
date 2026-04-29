from __future__ import annotations

from fastapi import Request

from mikazuki.app.training_run_context import create_training_run_context
from mikazuki.app.training_run_execution import launch_training, prepare_training_run


async def handle_training_run_request(request: Request):
    context, context_error = await create_training_run_context(request)
    if context_error:
        return context_error

    preparation_error = prepare_training_run(context)
    if preparation_error:
        return preparation_error

    return launch_training(context)
