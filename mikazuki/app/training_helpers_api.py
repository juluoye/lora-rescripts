from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from mikazuki.app.models import (
    APIResponse,
    APIResponseFail,
    APIResponseSuccess,
    CaptionBackupListRequest,
    CaptionBackupRequest,
    CaptionBackupRestoreRequest,
    CaptionCleanupRequest,
    DatasetAnalysisRequest,
    ImageResizeRequest,
    MaskedLossAuditRequest,
    TaggerInterrogateRequest,
)
from mikazuki.log import log
from mikazuki.tagger.interrogator import available_interrogators, on_interrogate
from mikazuki.tagger.llm import (
    LLM_INTERROGATORS,
    LLM_TEMPLATE_PRESETS,
    get_llm_interrogator_meta,
    is_llm_interrogator,
    run_llm_interrogate,
)
from mikazuki.utils.caption_backup import (
    NO_CAPTIONS_TO_BACKUP_MESSAGE,
    create_caption_backup,
    list_caption_backups,
    restore_caption_backup,
)
from mikazuki.utils.caption_cleanup import apply_caption_cleanup, preview_caption_cleanup
from mikazuki.utils.dataset_analysis import analyze_dataset
from mikazuki.utils.image_resize_runtime import (
    build_image_resize_preview_manifest,
    resolve_image_resize_file,
    resolve_image_resize_path,
    run_image_resize_job,
)
from mikazuki.utils.masked_loss_audit import analyze_masked_loss_dataset


router = APIRouter()


def maybe_create_caption_backup(
    *,
    path: str,
    caption_extension: str,
    recursive: bool,
    snapshot_name: str,
    allow_missing_captions: bool = False,
) -> tuple[Optional[dict], list[str]]:
    warnings: list[str] = []
    try:
        backup = create_caption_backup(
            path=path,
            caption_extension=caption_extension,
            recursive=recursive,
            snapshot_name=snapshot_name,
        )
    except ValueError as exc:
        if allow_missing_captions and str(exc) == NO_CAPTIONS_TO_BACKUP_MESSAGE:
            warnings.append("No existing caption files were found, so no backup snapshot was created.")
            return None, warnings
        raise

    return backup, warnings


@router.get("/image_resize/preview")
async def get_image_resize_preview(input_dir: str, recursive: bool = False, limit: int = 8) -> APIResponse:
    try:
        manifest = build_image_resize_preview_manifest(
            input_dir,
            recursive=recursive,
            limit=limit,
        )
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Image resize preview manifest failed")
        return APIResponseFail(message="Failed to load image preprocess preview / 图像预处理预览加载失败，请查看日志。")

    return APIResponseSuccess(data=manifest)


@router.get("/image_resize/file")
async def get_image_resize_file(path: str):
    try:
        file_path = resolve_image_resize_file(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Image not found: {exc.args[0]}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(file_path)


@router.post("/image_resize")
async def run_image_resize(req: ImageResizeRequest, background_tasks: BackgroundTasks):
    try:
        input_dir = resolve_image_resize_path(req.input_dir)
    except ValueError:
        return APIResponseFail(message="Input folder is empty / 输入目录不能为空。")

    if not input_dir.exists():
        return APIResponseFail(message=f"Input folder does not exist / 输入目录不存在：{input_dir}")
    if not input_dir.is_dir():
        return APIResponseFail(message=f"Input path is not a folder / 输入路径不是文件夹：{input_dir}")

    output_dir_raw = req.output_dir.strip()
    if output_dir_raw:
        try:
            output_dir = resolve_image_resize_path(output_dir_raw)
        except ValueError:
            return APIResponseFail(message="Output folder is invalid / 输出目录无效。")
        if output_dir.exists() and not output_dir.is_dir():
            return APIResponseFail(message=f"Output path is not a folder / 输出路径不是文件夹：{output_dir}")

    payload = req.model_dump()
    if "resize_mode" not in req.model_fields_set and req.exact_size:
        payload["resize_mode"] = "crop"
    background_tasks.add_task(run_image_resize_job, payload)
    return APIResponseSuccess(message="Image preprocessing task submitted / 图像预处理任务已提交。")


@router.post("/interrogate")
async def run_interrogate(req: TaggerInterrogateRequest, background_tasks: BackgroundTasks):
    batch_path = req.path.strip()
    if not batch_path:
        return APIResponseFail(message="Input folder is empty / 输入的图片文件夹为空。")
    if not os.path.isdir(batch_path):
        return APIResponseFail(message="Input path is not a valid folder / 输入路径不是有效文件夹。")

    use_llm_interrogator = is_llm_interrogator(req.interrogator_model)
    if not use_llm_interrogator:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return APIResponseFail(message="onnxruntime is not installed, please reinstall dependencies and try again.")

    response_warnings: list[str] = []
    backup_result = None
    if req.create_backup_before_write and req.batch_output_action_on_conflict != "ignore":
        backup_result, backup_warnings = maybe_create_caption_backup(
            path=batch_path,
            caption_extension=".txt",
            recursive=req.batch_input_recursive,
            snapshot_name=req.backup_snapshot_name.strip() or f"pre-batch-tagger-{req.interrogator_model}",
            allow_missing_captions=True,
        )
        response_warnings.extend(backup_warnings)

    if use_llm_interrogator:
        llm_meta = get_llm_interrogator_meta(req.interrogator_model)
        llm_api_key = req.llm_api_key.strip()
        llm_model = req.llm_model.strip()
        if not llm_api_key:
            return APIResponseFail(message="LLM API Key is empty / LLM API Key 不能为空。")
        if not llm_model:
            return APIResponseFail(message="LLM model is empty / LLM 模型名称不能为空。")
        if req.interrogator_model == "llm-custom" and not req.llm_api_base.strip():
            return APIResponseFail(message="Custom API Base URL is empty / 自定义 API 地址不能为空。")

        background_tasks.add_task(run_llm_interrogate, req.model_dump())
        llm_api_style = req.llm_api_style if req.interrogator_model == "llm-custom" else llm_meta.get("api_style", req.interrogator_model)
        llm_template_preset = (req.llm_template_preset or "anime-tags").strip() or "anime-tags"
        message = (
            f"LLM batch interrogate started for {batch_path} "
            f"({llm_api_style} / {llm_model} / preset={llm_template_preset})"
        )
        if req.llm_output_mode == "raw_text":
            response_warnings.append(
                "LLM raw_text mode writes the model response directly and does not apply tag post-processing."
            )
    else:
        interrogator = available_interrogators.get(req.interrogator_model, available_interrogators["wd14-convnextv2-v2"])
        background_tasks.add_task(
            on_interrogate,
            image=None,
            batch_input_glob=batch_path,
            batch_input_recursive=req.batch_input_recursive,
            batch_output_dir="",
            batch_output_filename_format="[name].[output_extension]",
            batch_output_action_on_conflict=req.batch_output_action_on_conflict,
            batch_remove_duplicated_tag=True,
            batch_output_save_json=False,
            interrogator=interrogator,
            threshold=req.threshold,
            character_threshold=req.character_threshold,
            add_rating_tag=req.add_rating_tag,
            add_model_tag=req.add_model_tag,
            additional_tags=req.additional_tags,
            exclude_tags=req.exclude_tags,
            sort_by_alphabetical_order=False,
            add_confident_as_weight=False,
            replace_underscore=req.replace_underscore,
            replace_underscore_excludes=req.replace_underscore_excludes,
            escape_tag=req.escape_tag,
            unload_model_after_running=True,
        )
        message = f"Batch interrogate started for {batch_path}"

    if backup_result is not None:
        message = f"{message} Created backup {backup_result['archive_name']} first."
    elif response_warnings:
        message = f"{message} {response_warnings[0]}"

    data = {}
    if backup_result is not None:
        data["backup"] = backup_result
    if response_warnings:
        data["warnings"] = response_warnings

    return APIResponseSuccess(message=message, data=data or None)


@router.post("/dataset/analyze")
async def dataset_analyze(req: DatasetAnalysisRequest) -> APIResponse:
    try:
        result = analyze_dataset(
            path=req.path,
            caption_extension=req.caption_extension,
            top_tags=req.top_tags,
            sample_limit=req.sample_limit,
        )
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Dataset analysis failed")
        return APIResponseFail(message="Dataset analysis failed / 数据集分析失败，请查看日志。")

    return APIResponseSuccess(data=result)


@router.post("/dataset/masked_loss_audit")
async def dataset_masked_loss_audit(req: MaskedLossAuditRequest) -> APIResponse:
    try:
        result = analyze_masked_loss_dataset(
            path=req.path,
            recursive=req.recursive,
            sample_limit=req.sample_limit,
        )
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Masked-loss dataset audit failed")
        return APIResponseFail(message="Masked-loss dataset audit failed / 蒙版损失数据集检查失败，请查看日志。")

    return APIResponseSuccess(data=result)


@router.get("/interrogators")
async def get_interrogators() -> APIResponse:
    default_interrogator = TaggerInterrogateRequest.model_fields["interrogator_model"].default
    return APIResponseSuccess(data={
        "default": default_interrogator,
        "interrogators": (
            [
                {
                    "name": name,
                    "kind": "cl" if name.startswith("cl_") else "wd",
                    "repo_id": getattr(interrogator, "repo_id", None),
                    "is_default": name == default_interrogator,
                }
                for name, interrogator in available_interrogators.items()
            ]
            + [
                {
                    "name": name,
                    "kind": meta["kind"],
                    "repo_id": None,
                    "api_style": meta["api_style"],
                    "default_api_base": meta["default_api_base"],
                    "is_default": False,
                }
                for name, meta in LLM_INTERROGATORS.items()
            ]
        ),
        "llm_template_presets": [
            {
                "id": preset["id"],
                "label": preset["label"],
                "output_mode": preset["output_mode"],
            }
            for preset in LLM_TEMPLATE_PRESETS.values()
        ],
    })


@router.post("/captions/cleanup/preview")
async def captions_cleanup_preview(req: CaptionCleanupRequest) -> APIResponse:
    try:
        preview_payload = req.model_dump(exclude={"create_backup_before_apply", "backup_snapshot_name"})
        result = preview_caption_cleanup(**preview_payload)
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Caption cleanup preview failed")
        return APIResponseFail(message="Caption cleanup preview failed / Caption 清洗预览失败，请查看日志。")

    return APIResponseSuccess(data=result)


@router.post("/captions/cleanup/apply")
async def captions_cleanup_apply(req: CaptionCleanupRequest) -> APIResponse:
    try:
        backup_result = None
        backup_warnings: list[str] = []
        if req.create_backup_before_apply:
            backup_result, backup_warnings = maybe_create_caption_backup(
                path=req.path,
                caption_extension=req.caption_extension,
                recursive=req.recursive,
                snapshot_name=req.backup_snapshot_name.strip() or "pre-caption-cleanup",
            )

        cleanup_payload = req.model_dump(exclude={"create_backup_before_apply", "backup_snapshot_name"})
        result = apply_caption_cleanup(**cleanup_payload)
        if backup_result is not None:
            result["backup"] = backup_result
        if backup_warnings:
            result["warnings"] = [*result.get("warnings", []), *backup_warnings]
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Caption cleanup apply failed")
        return APIResponseFail(message="Caption cleanup apply failed / Caption 清洗应用失败，请查看日志。")

    message = f"Updated {result['summary']['changed_file_count']} caption files."
    if backup_result is not None:
        message = f"{message} Created backup {backup_result['archive_name']} first."
    return APIResponseSuccess(message=message, data=result)


@router.post("/captions/backups/create")
async def captions_backup_create(req: CaptionBackupRequest) -> APIResponse:
    try:
        result = create_caption_backup(**req.model_dump())
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Caption backup creation failed")
        return APIResponseFail(message="Caption backup creation failed / Caption 备份创建失败，请查看日志。")

    return APIResponseSuccess(message=f"Created caption backup {result['archive_name']}", data=result)


@router.post("/captions/backups/list")
async def captions_backup_list(req: CaptionBackupListRequest) -> APIResponse:
    try:
        result = list_caption_backups(path=req.path.strip() or None)
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Caption backup listing failed")
        return APIResponseFail(message="Caption backup listing failed / Caption 备份列表读取失败，请查看日志。")

    return APIResponseSuccess(data={"backups": result})


@router.post("/captions/backups/restore")
async def captions_backup_restore(req: CaptionBackupRestoreRequest) -> APIResponse:
    try:
        result = restore_caption_backup(**req.model_dump())
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Caption backup restore failed")
        return APIResponseFail(message="Caption backup restore failed / Caption 备份恢复失败，请查看日志。")

    return APIResponseSuccess(message=f"Restored {result['restored_file_count']} caption files.", data=result)


@router.get("/dataset/list_images")
async def list_dataset_images(folder: str, limit: int = 8) -> APIResponse:
    try:
        folder_path = Path(folder).expanduser().resolve()
        if not folder_path.is_dir():
            return APIResponseFail(message="Folder not found")
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        all_images = sorted(
            [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in exts],
            key=lambda p: p.name.lower(),
        )
        images = [str(p.resolve()) for p in all_images[:limit]]
        first_tag = ""
        for img_path in all_images[:1]:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.is_file():
                try:
                    content = txt_path.read_text(encoding="utf-8-sig").strip()
                    if content:
                        first_tag = content.split(",")[0].strip()
                except Exception:
                    pass
                break
        return APIResponseSuccess(data={"images": images, "total": len(all_images), "first_tag": first_tag})
    except Exception as exc:
        return APIResponseFail(message=str(exc))
