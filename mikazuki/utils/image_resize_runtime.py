from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from PIL import Image, UnidentifiedImageError

from mikazuki.launch_utils import base_dir_path
from mikazuki.log import log
from mikazuki.utils.resume_guard import resolve_local_path


IMAGE_PREVIEW_DEFAULT_LIMIT = 8
IMAGE_PREVIEW_MAX_LIMIT = 24


@lru_cache(maxsize=1)
def _load_image_preprocessor_symbols():
    repo_root = str(base_dir_path())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        module = importlib.import_module("scripts.stable.lulynx.image_preprocessor")
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Lulynx image preprocessor is unavailable because the local scripts package could not be imported. "
            "Please confirm the release package contains ./scripts/stable/lulynx/image_preprocessor.py. "
            "/ Lulynx 图像预处理模块当前不可用：无法导入本地 scripts 包，请确认发行包内包含 "
            "./scripts/stable/lulynx/image_preprocessor.py。"
        ) from exc

    supported_image_extensions = getattr(module, "SUPPORTED_IMAGE_EXTENSIONS", None)
    collect_images = getattr(module, "collect_images", None)
    run_image_preprocessor = getattr(module, "run_image_preprocessor", None)
    if supported_image_extensions is None or not callable(collect_images) or not callable(run_image_preprocessor):
        raise ValueError(
            "Lulynx image preprocessor symbols are incomplete in the current build. "
            "/ 当前发行包中的 Lulynx 图像预处理模块缺少必需符号。"
        )

    return supported_image_extensions, collect_images, run_image_preprocessor


def resolve_image_resize_path(raw_path: str) -> Path:
    cleaned = str(raw_path or "").strip()
    if not cleaned:
        raise ValueError("Path is empty")
    return resolve_local_path(cleaned, base_dir_path())


def resolve_image_resize_file(raw_path: str) -> Path:
    path = resolve_image_resize_path(raw_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    supported_image_extensions, _, _ = _load_image_preprocessor_symbols()
    if path.suffix.lower() not in supported_image_extensions:
        raise ValueError(f"Unsupported image file: {path.name}")
    return path


def build_image_resize_preview_manifest(raw_input_dir: str, recursive: bool = False, limit: int = IMAGE_PREVIEW_DEFAULT_LIMIT) -> dict[str, object]:
    input_dir = resolve_image_resize_path(raw_input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a folder: {input_dir}")

    _, collect_images, _ = _load_image_preprocessor_symbols()
    safe_limit = max(1, min(int(limit or IMAGE_PREVIEW_DEFAULT_LIMIT), IMAGE_PREVIEW_MAX_LIMIT))
    images = collect_images(input_dir, recursive=recursive)
    items: list[dict[str, object]] = []
    skipped = 0

    for image_path in images:
        if len(items) >= safe_limit:
            break
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            skipped += 1
            continue

        try:
            relative_path = image_path.relative_to(input_dir)
        except ValueError:
            relative_path = Path(image_path.name)

        items.append(
            {
                "path": str(image_path),
                "name": image_path.name,
                "relative_path": str(relative_path).replace("\\", "/"),
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 6) if height else 1,
            }
        )

    return {
        "input_dir": str(input_dir),
        "recursive": bool(recursive),
        "total_count": len(images),
        "limit": safe_limit,
        "preview_count": len(items),
        "skipped_count": skipped,
        "items": items,
        "samples": items,
    }


def run_image_resize_job(raw_config: Mapping[str, object]) -> dict[str, object]:
    _, _, run_image_preprocessor = _load_image_preprocessor_symbols()
    payload = dict(raw_config)
    input_dir = resolve_image_resize_path(str(payload.get("input_dir", "") or ""))

    output_dir_raw = str(payload.get("output_dir", "") or "").strip()
    output_dir: Path | None = resolve_image_resize_path(output_dir_raw) if output_dir_raw else None

    payload["input_dir"] = str(input_dir)
    payload["output_dir"] = str(output_dir) if output_dir is not None else ""
    payload["log_callback"] = log.info

    log.info(
        "Starting lulynx image preprocessor: "
        f"input={input_dir} output={output_dir or '[in-place]'} "
        f"format={payload.get('format', 'ORIGINAL')} resize={payload.get('enable_resize', True)} "
        f"mode={payload.get('resize_mode', 'fit')} rename_mode={payload.get('rename_mode', 'legacy_suffix')} "
        f"recursive={payload.get('recursive', False)}"
    )
    log.info(
        "开始执行 lulynx 图像预处理："
        f"输入目录={input_dir}，输出目录={output_dir or '原地处理'}，"
        f"输出格式={payload.get('format', 'ORIGINAL')}，"
        f"启用缩放={payload.get('enable_resize', True)}，"
        f"模式={payload.get('resize_mode', 'fit')}，"
        f"命名模式={payload.get('rename_mode', 'legacy_suffix')}，"
        f"递归处理={payload.get('recursive', False)}"
    )

    try:
        summary = run_image_preprocessor(payload)
    except Exception:
        log.exception("lulynx image preprocessor failed")
        raise

    log.info(f"lulynx image preprocessor finished: {summary}")
    log.info(f"lulynx 图像预处理完成：{summary}")
    return summary
