from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request

from mikazuki.app.config import app_config
from mikazuki.app.models import APIResponse, APIResponseFail, APIResponseSuccess
from mikazuki.log import log
from mikazuki.plugins.runtime import plugin_runtime
from mikazuki.utils.train_utils import parse_boolish
from mikazuki.utils.frontend_profiles import (
    PLUGIN_ROOT,
    install_github_frontend_plugin,
    list_frontend_profiles,
    parse_github_repo_url,
    resolve_frontend_profile,
    resolve_frontend_profile_id,
    uninstall_frontend_plugin,
)


router = APIRouter()


@router.get("/plugins/runtime")
async def get_plugin_runtime_status() -> APIResponse:
    plugin_runtime.ensure_runtime_ready()
    return APIResponseSuccess(data=plugin_runtime.get_status())


@router.post("/plugins/reload")
async def reload_plugin_runtime() -> APIResponse:
    summary = plugin_runtime.reload()
    return APIResponseSuccess(data=summary)


@router.get("/plugins/capabilities")
async def get_plugin_capabilities() -> APIResponse:
    return APIResponseSuccess(data={"capabilities": plugin_runtime.list_capability_catalog()})


@router.get("/plugins/hooks")
async def get_plugin_hooks() -> APIResponse:
    return APIResponseSuccess(data={"hooks": plugin_runtime.list_hook_catalog()})


@router.get("/plugins/training_protocol")
async def get_plugin_training_protocol() -> APIResponse:
    return APIResponseSuccess(data=plugin_runtime.get_training_event_protocol_status())


@router.post("/plugins/developer_mode")
async def set_plugin_developer_mode(request: Request) -> APIResponse:
    payload = json.loads((await request.body()).decode("utf-8"))
    enabled = parse_boolish(payload.get("enabled", False))
    plugin_runtime.set_developer_mode(enabled)
    app_config["plugin_developer_mode"] = enabled
    app_config.save_config()
    return APIResponseSuccess(
        message=f"Plugin developer mode {'enabled' if enabled else 'disabled'}",
        data={"plugin_developer_mode": enabled},
    )


@router.post("/plugins/set_enabled")
async def set_plugin_enabled_state(request: Request) -> APIResponse:
    payload = json.loads((await request.body()).decode("utf-8"))
    plugin_id = str(payload.get("plugin_id", "")).strip()
    if not plugin_id:
        return APIResponseFail(message="plugin_id is required")
    enabled = parse_boolish(payload.get("enabled", False))
    updated_by = str(payload.get("updated_by", "")).strip() or "local-user"
    try:
        record = plugin_runtime.set_plugin_enabled(plugin_id, enabled=enabled, updated_by=updated_by)
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Failed to change plugin enabled state")
        return APIResponseFail(message="Failed to change plugin enabled state.")
    return APIResponseSuccess(
        message=f"Plugin {'enabled' if enabled else 'disabled'}",
        data={"enabled_record": record},
    )


@router.post("/plugins/reset_enabled")
async def reset_plugin_enabled_state(request: Request) -> APIResponse:
    payload = json.loads((await request.body()).decode("utf-8"))
    plugin_id = str(payload.get("plugin_id", "")).strip()
    if not plugin_id:
        return APIResponseFail(message="plugin_id is required")
    updated_by = str(payload.get("updated_by", "")).strip() or "local-user"
    try:
        record = plugin_runtime.reset_plugin_enabled(plugin_id, updated_by=updated_by)
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Failed to reset plugin enabled state")
        return APIResponseFail(message="Failed to reset plugin enabled state.")
    return APIResponseSuccess(
        message="Plugin enabled state reset to manifest default",
        data={"reset_record": record},
    )


@router.post("/plugins/approve")
async def approve_plugin(request: Request) -> APIResponse:
    payload = json.loads((await request.body()).decode("utf-8"))
    plugin_id = str(payload.get("plugin_id", "")).strip()
    approved_by = str(payload.get("approved_by", "")).strip() or "local-user"
    if not plugin_id:
        return APIResponseFail(message="plugin_id is required")
    try:
        record = plugin_runtime.approve_plugin(plugin_id, approved_by=approved_by)
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Failed to approve plugin")
        return APIResponseFail(message="Failed to approve plugin.")
    return APIResponseSuccess(data={"approval": record})


@router.post("/plugins/revoke_approval")
async def revoke_plugin_approval(request: Request) -> APIResponse:
    payload = json.loads((await request.body()).decode("utf-8"))
    plugin_id = str(payload.get("plugin_id", "")).strip()
    if not plugin_id:
        return APIResponseFail(message="plugin_id is required")
    removed = plugin_runtime.revoke_plugin_approval(plugin_id)
    return APIResponseSuccess(data={"plugin_id": plugin_id, "removed": removed})


@router.get("/plugins/audit")
async def get_plugin_audit(limit: int = 200) -> APIResponse:
    normalized_limit = max(1, min(int(limit or 200), 2000))
    events = plugin_runtime.list_recent_audit(limit=normalized_limit)
    return APIResponseSuccess(data={"events": events, "count": len(events)})


@router.get("/ui_profiles")
async def get_ui_profiles() -> APIResponse:
    requested_profile_id = app_config["active_ui_profile"]
    active_profile = resolve_frontend_profile(requested_profile_id)
    return APIResponseSuccess(data={
        "profiles": [
            {
                "id": profile["id"],
                "kind": profile["kind"],
                "name": profile["name"],
                "version": profile["version"],
                "source_path": profile["source_path"],
                "plugin_path": profile["plugin_path"],
                "source_url": profile["source_url"],
                "available": profile["available"],
                "removable": profile["removable"],
                "remove_block_reason": profile["remove_block_reason"],
            }
            for profile in list_frontend_profiles()
        ],
        "active_profile_id": active_profile["id"],
        "plugin_root": str(PLUGIN_ROOT),
        "config_path": str(app_config.path),
    })


@router.post("/ui_profiles/activate")
async def activate_ui_profile(request: Request) -> APIResponse:
    payload = json.loads((await request.body()).decode("utf-8"))
    profile_id = str(payload.get("profile_id", "")).strip()
    if not profile_id:
        return APIResponseFail(message="profile_id is required")

    profile = resolve_frontend_profile(profile_id)
    if profile["id"] != profile_id:
        return APIResponseFail(message=f"UI not found: {profile_id}")
    if not profile.get("available", False):
        return APIResponseFail(message=f"UI is not ready yet: {profile_id}")

    app_config["active_ui_profile"] = profile["id"]
    app_config.save_config()

    return APIResponseSuccess(
        message=f"Switched active UI to {profile['name']}",
        data={
            "active_profile_id": profile["id"],
            "reload_required": True,
        },
    )


@router.post("/ui_profiles/install")
async def install_ui_profile(request: Request) -> APIResponse:
    payload = json.loads((await request.body()).decode("utf-8"))
    repo_url = str(payload.get("repo_url", "")).strip()
    replace_existing = bool(payload.get("replace_existing", False))

    if not repo_url:
        return APIResponseFail(message="repo_url is required")

    parsed_repo = parse_github_repo_url(repo_url)
    if parsed_repo is None:
        return APIResponseFail(message="Only standard GitHub repository URLs are supported right now.")

    try:
        profile = await asyncio.to_thread(
            install_github_frontend_plugin,
            repo_url,
            replace_existing=replace_existing,
        )
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Failed to install community UI from GitHub")
        return APIResponseFail(message="Failed to download or install the GitHub community UI. Check the logs for details.")

    return APIResponseSuccess(
        message=f"Installed community UI {profile['name']}",
        data={
            "installed_profile": {
                "id": profile["id"],
                "name": profile["name"],
                "kind": profile["kind"],
                "version": profile["version"],
                "plugin_path": profile["plugin_path"],
                "source_path": profile["source_path"],
            },
            "plugin_root": str(PLUGIN_ROOT),
        },
    )


@router.post("/ui_profiles/uninstall")
async def uninstall_ui_profile(request: Request) -> APIResponse:
    payload = json.loads((await request.body()).decode("utf-8"))
    profile_id = str(payload.get("profile_id", "")).strip()
    if not profile_id:
        return APIResponseFail(message="profile_id is required")

    try:
        removed_profile = await asyncio.to_thread(uninstall_frontend_plugin, profile_id)
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Failed to uninstall community UI")
        return APIResponseFail(message="Failed to uninstall the selected community UI. Check the logs for details.")

    if app_config["active_ui_profile"] == removed_profile["id"]:
        app_config["active_ui_profile"] = resolve_frontend_profile_id(None)
        app_config.save_config()

    return APIResponseSuccess(
        message=f"Removed community UI {removed_profile['name']}",
        data={
            "removed_profile_id": removed_profile["id"],
            "active_profile_id": resolve_frontend_profile_id(app_config["active_ui_profile"]),
            "reload_required": True,
        },
    )
