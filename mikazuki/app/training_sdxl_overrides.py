from __future__ import annotations

from typing import Optional, Tuple

from mikazuki.app.training_prompt_utils import parse_boolish


def _parse_resolution_pair(value) -> Optional[Tuple[int, int]]:
    if value is None:
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    normalized = text.lower().replace("x", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) < 2:
        return None

    try:
        return int(float(parts[0])), int(float(parts[1]))
    except (TypeError, ValueError):
        return None


def _normalize_sdxl_low_vram_resolution_mode(value) -> str:
    normalized = str(value or "").strip().lower()
    if "short" in normalized or "短" in normalized:
        return "short_edge"
    return "long_edge"


def _normalize_sdxl_low_vram_preview_policy(value) -> str:
    normalized = str(value or "").strip().lower()
    if "disable" in normalized or "off" in normalized or "关闭" in normalized:
        return "disable"
    if "2" in normalized:
        return "every_2_epochs"
    return "every_4_epochs"


def _normalize_sdxl_low_vram_swap_threshold_ratio(value) -> float:
    try:
        percent = float(value)
    except (TypeError, ValueError):
        percent = 0.0
    percent = min(99.0, max(0.0, percent))
    return percent / 100.0


def _format_sdxl_fixed_block_swap_scope(swap_input_blocks: bool, swap_middle_block: bool, swap_output_blocks: bool) -> str:
    scopes: list[str] = []
    if swap_input_blocks:
        scopes.append("input")
    if swap_middle_block:
        scopes.append("middle")
    if swap_output_blocks:
        scopes.append("output")
    return "/".join(scopes) if scopes else "none"


def apply_sdxl_block_swap_ui_overrides(config: dict) -> list[str]:
    model_train_type = str(config.get("model_train_type", "") or "").strip().lower()
    if model_train_type != "sdxl-lora":
        return []

    warnings: list[str] = []
    block_swap_enabled = parse_boolish(config.get("sdxl_block_swap_enabled", False))
    low_vram_enabled = parse_boolish(config.get("sdxl_low_vram_optimization", False))

    child_requests = (
        parse_boolish(config.get("sdxl_block_swap_output_blocks", False))
        or parse_boolish(config.get("sdxl_block_swap_middle_block", False))
        or parse_boolish(config.get("sdxl_block_swap_offload_after_backward", False))
        or parse_boolish(config.get("sdxl_block_swap_input_blocks", False))
        or config.get("sdxl_block_swap_vram_threshold", None) not in (None, "")
    )

    if not block_swap_enabled:
        if child_requests:
            warnings.append("SDXL Block Swap 总开关已关闭，已忽略其子选项。")
        if low_vram_enabled:
            return warnings

        config["sdxl_fixed_block_swap"] = False
        config["sdxl_fixed_block_swap_input_blocks"] = False
        config["sdxl_fixed_block_swap_middle_block"] = False
        config["sdxl_fixed_block_swap_output_blocks"] = False
        config["sdxl_fixed_block_swap_offload_after_backward"] = True
        config["sdxl_fixed_block_swap_vram_threshold_ratio"] = 0.0
        return warnings

    swap_output_blocks = parse_boolish(config.get("sdxl_block_swap_output_blocks", True))
    swap_middle_block = parse_boolish(config.get("sdxl_block_swap_middle_block", True))
    swap_offload_after_backward = parse_boolish(config.get("sdxl_block_swap_offload_after_backward", True))
    swap_input_blocks = parse_boolish(config.get("sdxl_block_swap_input_blocks", False))
    swap_vram_threshold_ratio = _normalize_sdxl_low_vram_swap_threshold_ratio(config.get("sdxl_block_swap_vram_threshold", 70))
    fixed_block_swap = bool(swap_input_blocks or swap_middle_block or swap_output_blocks)

    config["sdxl_fixed_block_swap"] = fixed_block_swap
    config["sdxl_fixed_block_swap_input_blocks"] = swap_input_blocks
    config["sdxl_fixed_block_swap_middle_block"] = swap_middle_block
    config["sdxl_fixed_block_swap_output_blocks"] = swap_output_blocks
    config["sdxl_fixed_block_swap_offload_after_backward"] = swap_offload_after_backward
    config["sdxl_fixed_block_swap_vram_threshold_ratio"] = swap_vram_threshold_ratio

    if low_vram_enabled:
        warnings.append("已启用独立 SDXL Block Swap；本次会覆盖 ≤6GB 低显存优化里的 block swap 预设。")

    if fixed_block_swap:
        threshold_label = "始终尽快卸载" if swap_vram_threshold_ratio <= 0.0 else f"{swap_vram_threshold_ratio * 100:.0f}%"
        warnings.append(
            "已启用 SDXL Block Swap：推荐尝试顺序为 output -> middle -> offload_after_backward -> input。"
            f" 当前交换范围={_format_sdxl_fixed_block_swap_scope(swap_input_blocks, swap_middle_block, swap_output_blocks)}，"
            f"反向后卸载={'开启' if swap_offload_after_backward else '关闭'}，目标显存水线={threshold_label}。"
        )
    else:
        warnings.append("SDXL Block Swap 总开关已开启，但当前未勾选任何交换范围，因此本次不会启用 block swap。")

    return warnings


def apply_sdxl_low_vram_ui_overrides(config: dict) -> list[str]:
    model_train_type = str(config.get("model_train_type", "") or "").strip().lower()
    if model_train_type != "sdxl-lora":
        return []

    if not parse_boolish(config.get("sdxl_low_vram_optimization")):
        return []

    warnings: list[str] = []

    resolution_mode = _normalize_sdxl_low_vram_resolution_mode(config.get("sdxl_low_vram_resolution_mode"))
    preview_policy = _normalize_sdxl_low_vram_preview_policy(config.get("sdxl_low_vram_preview_policy"))
    two_phase_cache = parse_boolish(config.get("sdxl_low_vram_two_phase_cache", True))
    component_cpu_residency = parse_boolish(config.get("sdxl_low_vram_component_cpu_residency", True))
    fixed_block_swap = parse_boolish(config.get("sdxl_low_vram_fixed_block_swap", True))
    swap_input_blocks = parse_boolish(config.get("sdxl_low_vram_swap_input_blocks", False))
    swap_middle_block = parse_boolish(config.get("sdxl_low_vram_swap_middle_block", True))
    swap_output_blocks = parse_boolish(config.get("sdxl_low_vram_swap_output_blocks", True))
    fixed_block_swap = bool(fixed_block_swap and (swap_input_blocks or swap_middle_block or swap_output_blocks))
    swap_offload_after_backward = parse_boolish(config.get("sdxl_low_vram_swap_offload_after_backward", True))
    swap_vram_threshold_ratio = _normalize_sdxl_low_vram_swap_threshold_ratio(
        config.get("sdxl_low_vram_swap_vram_threshold", 0)
    )
    auto_protection = parse_boolish(config.get("sdxl_low_vram_auto_protection", True))
    auto_resolution_probe = parse_boolish(config.get("sdxl_low_vram_auto_resolution_probe", True))

    try:
        bucket_steps = int(config.get("sdxl_low_vram_bucket_reso_steps", 32) or 32)
    except (TypeError, ValueError):
        bucket_steps = 32
    if bucket_steps not in {32, 64}:
        bucket_steps = 32

    resolution_pair = _parse_resolution_pair(config.get("resolution"))
    if resolution_pair is None:
        resolution_pair = (1024, 1024)
    width, height = resolution_pair
    target_edge = max(width, height) if resolution_mode == "long_edge" else min(width, height)
    target_edge = max(64, int(target_edge))

    config["enable_bucket"] = True
    config["bucket_no_upscale"] = True
    config["bucket_reso_steps"] = bucket_steps
    config["gradient_checkpointing"] = True
    config["cache_latents"] = True
    config["cache_text_encoder_outputs"] = True
    config["network_train_unet_only"] = True
    config["network_train_text_encoder_only"] = False
    config["sample_at_first"] = False
    config["sdxl_bucket_resolution_mode"] = resolution_mode
    config["sdxl_bucket_target_edge"] = target_edge
    config["sdxl_component_cpu_residency"] = component_cpu_residency
    config["sdxl_fixed_block_swap"] = fixed_block_swap
    config["sdxl_fixed_block_swap_input_blocks"] = swap_input_blocks
    config["sdxl_fixed_block_swap_middle_block"] = swap_middle_block
    config["sdxl_fixed_block_swap_output_blocks"] = swap_output_blocks
    config["sdxl_fixed_block_swap_offload_after_backward"] = swap_offload_after_backward
    config["sdxl_fixed_block_swap_vram_threshold_ratio"] = swap_vram_threshold_ratio
    config["sdxl_low_vram_two_phase_cache"] = two_phase_cache
    config["sdxl_low_vram_auto_protection"] = auto_protection
    config["sdxl_low_vram_auto_resolution_probe"] = auto_resolution_probe
    config["sdxl_low_vram_probe_dedicated_limit_ratio"] = 0.95
    config["_runtime_safe_preview_enabled"] = preview_policy != "disable"
    config["_runtime_preview_backend"] = "sdpa-safe"
    config["_runtime_safe_preview_max_width"] = 768
    config["_runtime_safe_preview_max_height"] = 768
    config["_runtime_safe_preview_max_steps"] = 12
    config["_runtime_safe_preview_cfg_cap"] = 5.5

    if parse_boolish(config.get("shuffle_caption")):
        config["shuffle_caption"] = False
        warnings.append("低显存优化已自动关闭 shuffle_caption，以兼容 SDXL 文本编码器输出缓存。")

    for numeric_key in (
        "caption_dropout_rate",
        "caption_dropout_every_n_epochs",
        "caption_tag_dropout_rate",
        "token_warmup_step",
    ):
        raw_value = config.get(numeric_key)
        if raw_value in (None, "", 0, 0.0):
            continue
        config[numeric_key] = 0
        warnings.append(f"低显存优化已自动将 `{numeric_key}` 设为 0，以保证文本编码器缓存稳定可复用。")

    for bool_key in ("random_crop", "color_aug"):
        if parse_boolish(config.get(bool_key)):
            config[bool_key] = False
            warnings.append(f"低显存优化已自动关闭 `{bool_key}`，以兼容 latent 缓存。")

    config.pop("sample_every_n_steps", None)
    if preview_policy == "disable":
        config["enable_preview"] = False
        config["sample_every_n_epochs"] = 0
        config["_runtime_safe_preview_enabled"] = False
    else:
        config["enable_preview"] = True
        config["sample_every_n_epochs"] = 2 if preview_policy == "every_2_epochs" else 4

    warnings.append(
        "已启用 SDXL 低显存优化（≤6GB）：强制开启 gradient_checkpointing、cache_latents、cache_text_encoder_outputs，"
        f"并切换为 `{'long_edge' if resolution_mode == 'long_edge' else 'short_edge'}` 边长规划、bucket_step={bucket_steps}。"
    )
    if fixed_block_swap:
        threshold_label = "始终尽快卸载" if swap_vram_threshold_ratio <= 0.0 else f"{swap_vram_threshold_ratio * 100:.0f}%"
        warnings.append(
            "低显存优化已启用 U-Net block swap："
            f"当前交换范围={_format_sdxl_fixed_block_swap_scope(swap_input_blocks, swap_middle_block, swap_output_blocks)}，"
            f"反向后卸载={'开启' if swap_offload_after_backward else '关闭'}，目标显存水线={threshold_label}。"
        )
    elif parse_boolish(config.get("sdxl_low_vram_fixed_block_swap", True)):
        warnings.append("低显存优化中的 U-Net block swap 总开关已开启，但当前未勾选任何交换范围，因此本次不会启用 block swap。")
    if auto_resolution_probe:
        warnings.append(
            "低显存优化已启用启动前自动分辨率探测：会先做 3 步预跑；若检测到共享显存或专用显存峰值超过 95%，会按 64 为单位自动下调目标边长。"
        )
    if preview_policy != "disable":
        warnings.append("低显存优化已对预览启用前置硬钳制：单张预览最长边会在生成前被压到不高于 768，步数不高于 12。")

    return warnings


def build_sdxl_clip_skip_warning(config: dict) -> Optional[str]:
    training_type = str(config.get("model_train_type", "") or "").strip().lower()
    if not training_type.startswith("sdxl"):
        return None

    raw_clip_skip = config.get("clip_skip")
    try:
        clip_skip = int(raw_clip_skip)
    except (TypeError, ValueError):
        return None

    if clip_skip <= 1:
        return None

    return (
        f"当前配置启用了 SDXL clip_skip={clip_skip}。该组合仍属实验性设置，"
        "可能导致训练结果与推理表现不一致，也可能让预览图提前出现异常；若无明确需要，建议改回 1。"
    )
