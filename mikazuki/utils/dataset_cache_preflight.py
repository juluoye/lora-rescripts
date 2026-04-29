from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from mikazuki.utils.dataset_analysis import discover_dataset_folders, iter_image_files, read_image_size
from mikazuki.utils.train_utils import parse_boolish


@dataclass(frozen=True)
class LatentCacheProfile:
    suffix: str
    stride: int
    multi_resolution: bool
    supports_legacy_npz: bool = False


@dataclass(frozen=True)
class BucketPlan:
    resolution: tuple[int, int]
    enable_bucket: bool
    bucket_no_upscale: bool
    min_bucket_reso: int
    max_bucket_reso: int
    bucket_reso_steps: int
    predefined_resos: tuple[tuple[int, int], ...]
    predefined_aspect_ratios: tuple[float, ...]

    @property
    def max_area(self) -> int:
        return self.resolution[0] * self.resolution[1]


LATENT_CACHE_PROFILES: dict[str, LatentCacheProfile] = {
    "sd": LatentCacheProfile("_sd.npz", 8, False, supports_legacy_npz=True),
    "sdxl": LatentCacheProfile("_sdxl.npz", 8, False, supports_legacy_npz=True),
    "sd3": LatentCacheProfile("_sd3.npz", 8, True),
    "flux": LatentCacheProfile("_flux.npz", 8, True),
    "lumina": LatentCacheProfile("_lumina.npz", 8, True),
    "hunyuan-image": LatentCacheProfile("_hi.npz", 32, True),
    "anima": LatentCacheProfile("_anima.npz", 8, True),
}

MAX_SAMPLE_ITEMS = 8


def analyze_dataset_cache_preflight(config: dict, *, training_type: str) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    train_data_dir = str(config.get("train_data_dir", "")).strip()
    if not train_data_dir:
        return {
            "ready": True,
            "errors": [],
            "warnings": [],
            "notes": [],
            "summary": {},
        }

    root = Path(train_data_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return {
            "ready": True,
            "errors": [],
            "warnings": [],
            "notes": [],
            "summary": {},
        }

    scan_dirs, _, discover_warnings = discover_dataset_folders(root)
    warnings.extend(discover_warnings)

    image_size_cache: dict[Path, tuple[int, int]] = {}

    cache_report = analyze_latent_cache_state(
        root=root,
        scan_dirs=scan_dirs,
        image_size_cache=image_size_cache,
        config=config,
        training_type=training_type,
    )
    errors.extend(cache_report["errors"])
    warnings.extend(cache_report["warnings"])
    notes.extend(cache_report["notes"])

    metadata_report = analyze_metadata_cache_state(
        scan_dirs=scan_dirs,
        image_size_cache=image_size_cache,
        cache_info_enabled=parse_boolish(config.get("cache_info", False)),
        clear_before_train=parse_boolish(config.get("clear_dataset_npz_before_train", False)),
    )
    errors.extend(metadata_report["errors"])
    warnings.extend(metadata_report["warnings"])
    notes.extend(metadata_report["notes"])

    summary = {
        "latent_cache_file_count": cache_report["summary"].get("selected_cache_file_count", 0),
        "stale_latent_cache_count": cache_report["summary"].get("stale_cache_count", 0),
        "legacy_latent_cache_count": cache_report["summary"].get("legacy_cache_count", 0),
        "legacy_shadowing_count": cache_report["summary"].get("legacy_shadowing_count", 0),
        "metadata_cache_file_count": metadata_report["summary"].get("metadata_cache_file_count", 0),
        "metadata_mismatch_count": metadata_report["summary"].get("metadata_mismatch_count", 0),
    }

    return {
        "ready": len(errors) == 0,
        "errors": dedupe_strings(errors),
        "warnings": dedupe_strings(warnings),
        "notes": dedupe_strings(notes),
        "summary": summary,
    }


def analyze_latent_cache_state(
    *,
    root: Path,
    scan_dirs: list[Path],
    image_size_cache: dict[Path, tuple[int, int]],
    config: dict,
    training_type: str,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    cache_to_disk = parse_boolish(config.get("cache_latents_to_disk", False))
    clear_before_train = parse_boolish(config.get("clear_dataset_npz_before_train", False))
    skip_cache_check = parse_boolish(config.get("skip_cache_check", False))
    if not cache_to_disk:
        return {
            "errors": [],
            "warnings": [],
            "notes": [],
            "summary": {},
        }

    if clear_before_train:
        return {
            "errors": [],
            "warnings": [],
            "notes": ["clear_dataset_npz_before_train is enabled, so stale latent caches will be cleared before launch."],
            "summary": {},
        }

    if skip_cache_check:
        warnings.append(
            "skip_cache_check is enabled while cache_latents_to_disk is on. This can reuse stale latent caches after "
            "changing resolution / bucket settings or replacing images. / 当前启用了 skip_cache_check 且开启了磁盘 latent 缓存，"
            "若修改过分辨率、bucket 设置或替换过图片，可能会复用过期缓存。"
        )

    profile = resolve_latent_cache_profile(training_type)
    if profile is None:
        notes.append(f"Dataset latent cache audit is not implemented for trainer type {training_type}.")
        return {
            "errors": [],
            "warnings": warnings,
            "notes": notes,
            "summary": {},
        }

    bucket_plan = build_bucket_plan(config)
    if bucket_plan is None:
        warnings.append("Dataset latent cache audit skipped because resolution / bucket settings could not be parsed.")
        return {
            "errors": [],
            "warnings": warnings,
            "notes": notes,
            "summary": {},
        }

    selected_cache_file_count = 0
    stale_cache_count = 0
    legacy_cache_count = 0
    legacy_shadowing_count = 0
    stale_samples: list[str] = []
    legacy_shadow_samples: list[str] = []

    for folder in scan_dirs:
        for image_path in iter_image_files(folder):
            width, height = get_image_size_cached(image_path, image_size_cache)
            if not width or not height:
                continue

            bucket_reso = select_bucket_resolution(bucket_plan, width, height)
            cache_path, cache_kind, shadowed_new_cache = resolve_latent_cache_path(profile, image_path, width, height)

            if cache_kind == "legacy":
                legacy_cache_count += 1
                if shadowed_new_cache is not None and shadowed_new_cache.exists():
                    legacy_shadowing_count += 1
                    append_sample(
                        legacy_shadow_samples,
                        f"{cache_path} shadows {shadowed_new_cache}",
                    )

            if cache_path is None or not cache_path.exists():
                continue

            selected_cache_file_count += 1
            valid, reason = validate_latent_cache_file(
                cache_path,
                profile=profile,
                bucket_reso=bucket_reso,
                flip_aug=parse_boolish(config.get("flip_aug", False)),
                alpha_mask=parse_boolish(config.get("alpha_mask", False)),
            )
            if valid:
                continue

            stale_cache_count += 1
            append_sample(
                stale_samples,
                f"{cache_path} -> expected bucket {bucket_reso[0]}x{bucket_reso[1]} ({reason})",
            )

    if stale_cache_count > 0:
        sample_text = "\n".join(stale_samples)
        if skip_cache_check:
            errors.append(
                "Detected stale dataset latent cache files that do not match the current training bucket / resolution settings, "
                "and skip_cache_check is enabled, so the runtime may incorrectly reuse them. Delete the dataset latent cache files "
                "(*.npz such as *_sd.npz / *_sdxl.npz / *_flux.npz / legacy .npz), disable skip_cache_check, or enable "
                "clear_dataset_npz_before_train, then retry.\n"
                "检测到数据集 latent 缓存与当前训练的分桶 / 分辨率设置不匹配，且当前启用了 skip_cache_check，"
                "运行时可能会错误复用这些过期缓存。请删除数据集目录中的 latent 缓存文件"
                "（*.npz，例如 *_sd.npz / *_sdxl.npz / *_flux.npz / 旧版 .npz），"
                "或关闭 skip_cache_check，或启用 clear_dataset_npz_before_train 后重试。\n"
                f"Examples:\n{sample_text}"
            )
        else:
            warnings.append(
                "Detected stale dataset latent cache files that do not match the current training bucket / resolution settings. "
                "Valid caches can still be reused, and stale caches should be regenerated during latent caching before training starts. "
                "If you prefer a full cleanup first, delete the dataset latent cache files (*.npz such as *_sd.npz / *_sdxl.npz / "
                "*_flux.npz / legacy .npz) or enable clear_dataset_npz_before_train.\n"
                "检测到数据集 latent 缓存与当前训练的分桶 / 分辨率设置不匹配。有效缓存仍可继续复用，"
                "过期缓存会在训练开始前的 latent 缓存阶段重新生成。若你希望先完整清理，"
                "可删除数据集目录中的 latent 缓存文件（*.npz，例如 *_sd.npz / *_sdxl.npz / *_flux.npz / 旧版 .npz），"
                "或启用 clear_dataset_npz_before_train。\n"
                f"Examples:\n{sample_text}"
            )

    if legacy_shadowing_count > 0:
        sample_text = "\n".join(legacy_shadow_samples)
        warnings.append(
            "Legacy generic .npz latent caches were found and they shadow newer cache files with resolution suffixes. "
            "Even when valid, these old caches make debugging harder after dataset changes. Consider deleting them.\n"
            "检测到旧版通用 .npz latent 缓存，它们会遮蔽带分辨率后缀的新缓存文件。即使当前还能用，"
            "数据集变动后也更容易引发问题，建议清理。\n"
            f"Examples:\n{sample_text}"
        )
    elif legacy_cache_count > 0:
        warnings.append(
            f"Detected {legacy_cache_count} legacy generic .npz latent cache files. "
            "They are older cache format and can be confusing after dataset or bucket changes."
        )

    if selected_cache_file_count > 0 and stale_cache_count == 0:
        notes.append(
            f"Checked {selected_cache_file_count} dataset latent cache files against the current bucket settings."
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "summary": {
            "selected_cache_file_count": selected_cache_file_count,
            "stale_cache_count": stale_cache_count,
            "legacy_cache_count": legacy_cache_count,
            "legacy_shadowing_count": legacy_shadowing_count,
        },
    }


def analyze_metadata_cache_state(
    *,
    scan_dirs: list[Path],
    image_size_cache: dict[Path, tuple[int, int]],
    cache_info_enabled: bool,
    clear_before_train: bool,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    metadata_cache_file_count = 0
    metadata_mismatch_count = 0
    metadata_samples: list[str] = []

    for folder in scan_dirs:
        metadata_path = folder / "metadata_cache.json"
        if not metadata_path.exists():
            continue

        metadata_cache_file_count += 1
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"Could not read metadata cache file {metadata_path}: {exc}")
            continue

        for raw_path, item in payload.items():
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = (folder / raw_path).resolve()
            else:
                image_path = image_path.resolve()

            if not image_path.exists():
                continue

            recorded = item.get("resolution")
            if not isinstance(recorded, list) or len(recorded) != 2:
                continue

            width, height = get_image_size_cached(image_path, image_size_cache)
            if not width or not height:
                continue

            actual = (int(width), int(height))
            cached = (int(recorded[0]), int(recorded[1]))
            if actual == cached:
                continue

            metadata_mismatch_count += 1
            append_sample(
                metadata_samples,
                f"{image_path} -> metadata {cached[0]}x{cached[1]}, actual {actual[0]}x{actual[1]}",
            )

    if metadata_mismatch_count > 0:
        sample_text = "\n".join(metadata_samples)
        if clear_before_train:
            notes.append(
                "Detected stale metadata_cache.json entries whose recorded image sizes do not match the current files. "
                "clear_dataset_npz_before_train is enabled, so metadata_cache.json will be cleared before training starts.\n"
                "检测到 metadata_cache.json 中记录的图像尺寸与当前文件不一致。由于已启用 "
                "clear_dataset_npz_before_train，训练开始前会自动清理 metadata_cache.json。\n"
                f"Examples:\n{sample_text}"
            )
        else:
            message = (
                "Detected stale metadata_cache.json entries whose recorded image sizes do not match the current files. "
                "Delete metadata_cache.json in the affected dataset folders before training, "
                "or enable clear_dataset_npz_before_train to auto-clear it before launch.\n"
                "检测到 metadata_cache.json 中记录的图像尺寸与当前文件不一致。请删除对应数据集文件夹中的 "
                "metadata_cache.json，或启用 clear_dataset_npz_before_train 让训练前自动清理后再训练。\n"
                f"Examples:\n{sample_text}"
            )
            if cache_info_enabled:
                errors.append(message)
            else:
                notes.append(message)

    if metadata_cache_file_count > 0 and metadata_mismatch_count == 0:
        notes.append(f"Checked {metadata_cache_file_count} metadata_cache.json files.")

    return {
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "summary": {
            "metadata_cache_file_count": metadata_cache_file_count,
            "metadata_mismatch_count": metadata_mismatch_count,
        },
    }


def resolve_latent_cache_profile(training_type: str) -> Optional[LatentCacheProfile]:
    normalized = str(training_type or "").strip().lower()
    if normalized.startswith("sdxl"):
        return LATENT_CACHE_PROFILES["sdxl"]
    if normalized.startswith("sd3"):
        return LATENT_CACHE_PROFILES["sd3"]
    if normalized.startswith("flux"):
        return LATENT_CACHE_PROFILES["flux"]
    if normalized.startswith("lumina"):
        return LATENT_CACHE_PROFILES["lumina"]
    if normalized.startswith("hunyuan-image"):
        return LATENT_CACHE_PROFILES["hunyuan-image"]
    if normalized.startswith("anima"):
        return LATENT_CACHE_PROFILES["anima"]
    if normalized.startswith("sd"):
        return LATENT_CACHE_PROFILES["sd"]
    return None


def build_bucket_plan(config: dict) -> Optional[BucketPlan]:
    resolution = parse_resolution(config.get("resolution"))
    if resolution is None:
        return None

    enable_bucket = parse_boolish(config.get("enable_bucket", False))
    bucket_no_upscale = parse_boolish(config.get("bucket_no_upscale", False))
    min_bucket_reso = int(config.get("min_bucket_reso", 256) or 256)
    max_bucket_reso = int(config.get("max_bucket_reso", max(resolution)) or max(resolution))
    bucket_reso_steps = int(config.get("bucket_reso_steps", 64) or 64)

    predefined_resos: tuple[tuple[int, int], ...] = ()
    predefined_aspect_ratios: tuple[float, ...] = ()

    if enable_bucket:
        min_bucket_reso, max_bucket_reso = adjust_min_max_bucket_reso_by_steps(
            resolution,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
        )
        if not bucket_no_upscale:
            predefined_resos = tuple(make_bucket_resolutions(resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps))
            predefined_aspect_ratios = tuple(w / h for w, h in predefined_resos)

    return BucketPlan(
        resolution=resolution,
        enable_bucket=enable_bucket,
        bucket_no_upscale=bucket_no_upscale,
        min_bucket_reso=min_bucket_reso,
        max_bucket_reso=max_bucket_reso,
        bucket_reso_steps=bucket_reso_steps,
        predefined_resos=predefined_resos,
        predefined_aspect_ratios=predefined_aspect_ratios,
    )


def adjust_min_max_bucket_reso_by_steps(
    resolution: tuple[int, int],
    min_bucket_reso: int,
    max_bucket_reso: int,
    bucket_reso_steps: int,
) -> tuple[int, int]:
    if min_bucket_reso % bucket_reso_steps != 0:
        min_bucket_reso = min_bucket_reso - min_bucket_reso % bucket_reso_steps
    if max_bucket_reso % bucket_reso_steps != 0:
        max_bucket_reso = max_bucket_reso + bucket_reso_steps - max_bucket_reso % bucket_reso_steps

    min_bucket_reso = min(min_bucket_reso, min(resolution))
    max_bucket_reso = max(max_bucket_reso, max(resolution))
    return min_bucket_reso, max_bucket_reso


def make_bucket_resolutions(
    max_reso: tuple[int, int],
    min_size: int = 256,
    max_size: int = 1024,
    divisible: int = 64,
) -> list[tuple[int, int]]:
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos: set[tuple[int, int]] = set()
    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))
        width += divisible

    return sorted(resos)


def select_bucket_resolution(plan: BucketPlan, image_width: int, image_height: int) -> tuple[int, int]:
    if not plan.enable_bucket:
        return plan.resolution

    aspect_ratio = image_width / image_height
    if not plan.bucket_no_upscale:
        reso = (image_width, image_height)
        if reso not in set(plan.predefined_resos):
            ar_errors = [abs(candidate - aspect_ratio) for candidate in plan.predefined_aspect_ratios]
            candidate_index = ar_errors.index(min(ar_errors))
            reso = plan.predefined_resos[candidate_index]
        return reso

    if image_width * image_height > plan.max_area:
        resized_width = math.sqrt(plan.max_area * aspect_ratio)
        resized_height = plan.max_area / resized_width

        rounded_width = round_to_steps(resized_width, plan.bucket_reso_steps)
        height_from_width = round_to_steps(rounded_width / aspect_ratio, plan.bucket_reso_steps)
        ar_width = rounded_width / height_from_width

        rounded_height = round_to_steps(resized_height, plan.bucket_reso_steps)
        width_from_height = round_to_steps(rounded_height * aspect_ratio, plan.bucket_reso_steps)
        ar_height = width_from_height / rounded_height

        if abs(ar_width - aspect_ratio) < abs(ar_height - aspect_ratio):
            resized_size = (rounded_width, int(rounded_width / aspect_ratio + 0.5))
        else:
            resized_size = (int(rounded_height * aspect_ratio + 0.5), rounded_height)
    else:
        resized_size = (image_width, image_height)

    bucket_width = max(plan.bucket_reso_steps, resized_size[0] - resized_size[0] % plan.bucket_reso_steps)
    bucket_height = max(plan.bucket_reso_steps, resized_size[1] - resized_size[1] % plan.bucket_reso_steps)
    return bucket_width, bucket_height


def round_to_steps(value: float, steps: int) -> int:
    rounded = int(value + 0.5)
    return max(steps, rounded - rounded % steps)


def resolve_latent_cache_path(
    profile: LatentCacheProfile,
    image_path: Path,
    image_width: int,
    image_height: int,
) -> tuple[Optional[Path], Optional[str], Optional[Path]]:
    legacy_path = image_path.with_suffix(".npz")
    new_cache_path = image_path.with_name(f"{image_path.stem}_{image_width:04d}x{image_height:04d}{profile.suffix}")

    if profile.supports_legacy_npz and legacy_path.exists():
        return legacy_path, "legacy", new_cache_path
    return new_cache_path, "current", None


def validate_latent_cache_file(
    cache_path: Path,
    *,
    profile: LatentCacheProfile,
    bucket_reso: tuple[int, int],
    flip_aug: bool,
    alpha_mask: bool,
) -> tuple[bool, str]:
    expected_latents_size = (bucket_reso[1] // profile.stride, bucket_reso[0] // profile.stride)
    key_suffix = f"_{expected_latents_size[0]}x{expected_latents_size[1]}" if profile.multi_resolution else ""

    try:
        with np.load(cache_path) as payload:
            latents_key = "latents" + key_suffix
            if latents_key not in payload:
                return False, f"missing {latents_key}"
            latents = payload[latents_key]
            if tuple(latents.shape[1:3]) != expected_latents_size:
                return False, f"latent shape is {tuple(latents.shape[1:3])}, expected {expected_latents_size}"

            original_size_key = "original_size" + key_suffix
            crop_key = "crop_ltrb" + key_suffix
            if original_size_key not in payload or crop_key not in payload:
                return False, f"missing {original_size_key} or {crop_key}"

            if flip_aug:
                flipped_key = "latents_flipped" + key_suffix
                if flipped_key not in payload:
                    return False, f"missing {flipped_key}"
                flipped = payload[flipped_key]
                if tuple(flipped.shape[1:3]) != expected_latents_size:
                    return False, f"flipped latent shape is {tuple(flipped.shape[1:3])}, expected {expected_latents_size}"

            alpha_key = "alpha_mask" + key_suffix
            if alpha_mask:
                if alpha_key not in payload:
                    return False, f"missing {alpha_key}"
                alpha = payload[alpha_key]
                if tuple(alpha.shape[0:2]) != (bucket_reso[1], bucket_reso[0]):
                    return False, f"alpha mask shape is {tuple(alpha.shape[0:2])}, expected {(bucket_reso[1], bucket_reso[0])}"
    except Exception as exc:
        return False, f"cache read failed: {exc}"

    return True, "ok"


def get_image_size_cached(image_path: Path, cache: dict[Path, tuple[int, int]]) -> tuple[int, int]:
    resolved = image_path.resolve()
    cached = cache.get(resolved)
    if cached is not None:
        return cached

    width, height, _ = read_image_size(resolved)
    if width is None or height is None:
        return 0, 0

    size = (int(width), int(height))
    cache[resolved] = size
    return size


def parse_resolution(value) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, str):
        pieces = [item.strip() for item in value.replace("x", ",").split(",") if item.strip()]
        if not pieces:
            return None
        if len(pieces) == 1:
            size = int(pieces[0])
            return size, size
        if len(pieces) == 2:
            return int(pieces[0]), int(pieces[1])
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            size = int(value[0])
            return size, size
        if len(value) >= 2:
            return int(value[0]), int(value[1])
    if isinstance(value, (int, float)):
        size = int(value)
        return size, size
    return None


def append_sample(target: list[str], value: str) -> None:
    if len(target) >= MAX_SAMPLE_ITEMS or value in target:
        return
    target.append(value)


def dedupe_strings(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
