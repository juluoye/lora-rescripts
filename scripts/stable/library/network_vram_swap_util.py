from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.device_utils import clean_memory_on_device


SUPPORTED_NETWORK_MODULES = {
    "networks.lora",
    "networks.lora_fa",
    "networks.tlora",
    "networks.vera",
    "networks.lora_flux",
    "networks.tlora_flux",
    "networks.lora_sd3",
    "networks.lora_lumina",
    "networks.lora_hunyuan_image",
    "networks.lora_anima",
    "networks.tlora_anima",
}

UNSUPPORTED_OPTIMIZER_KEYWORDS = (
    "bitsandbytes",
    "8bit",
    "paged",
    "ademamix",
)


def parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def network_module_supports_vram_swap(network_module: str) -> bool:
    return str(network_module or "").strip().lower() in SUPPORTED_NETWORK_MODULES


def optimizer_supports_vram_swap(optimizer_name: str, optimizer_type: str | None = None) -> bool:
    text = f"{optimizer_name or ''} {optimizer_type or ''}".strip().lower()
    if not text:
        return True
    return not any(keyword in text for keyword in UNSUPPORTED_OPTIMIZER_KEYWORDS)


def module_uses_vram_swap(module: nn.Module) -> bool:
    return bool(getattr(module, "_vram_swap_to_ram_enabled", False))


def enable_vram_swap_for_module(module: nn.Module) -> None:
    setattr(module, "_vram_swap_to_ram_enabled", True)
    setattr(module, "_vram_swap_runtime_device", None)


def set_vram_swap_runtime_device(module: nn.Module, device: torch.device) -> None:
    if module_uses_vram_swap(module):
        setattr(module, "_vram_swap_runtime_device", device)


def get_vram_swap_runtime_device(module: nn.Module) -> torch.device | None:
    return getattr(module, "_vram_swap_runtime_device", None)


def forward_linear_module(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    weight = module.weight.to(device=x.device, dtype=x.dtype)
    bias = module.bias.to(device=x.device, dtype=x.dtype) if module.bias is not None else None
    return F.linear(x, weight, bias)


def forward_conv2d_module(module: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    weight = module.weight.to(device=x.device, dtype=x.dtype)
    bias = module.bias.to(device=x.device, dtype=x.dtype) if module.bias is not None else None
    return F.conv2d(
        x,
        weight,
        bias,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
    )


def forward_supported_module(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return forward_linear_module(module, x)
    if isinstance(module, nn.Conv2d):
        return forward_conv2d_module(module, x)
    return module(x)


def maybe_activate_network_vram_swap(
    args,
    accelerator,
    network,
    optimizer_name: str,
    logger: logging.Logger,
    *,
    route_label: str,
) -> bool:
    if not parse_boolish(getattr(args, "vram_swap_to_ram", False)):
        return False

    network_module = str(getattr(args, "network_module", "") or "").strip().lower()
    if not network_module_supports_vram_swap(network_module):
        logger.warning(
            f"{route_label}: vram_swap_to_ram is currently supported only on native LoRA routes "
            f"(current network_module={network_module or '(unknown)'}). Ignoring this option."
        )
        logger.warning(
            f"{route_label}：vram_swap_to_ram 当前只支持原生 LoRA 路线，"
            f"当前 network_module={network_module or '(unknown)'}，已自动忽略。"
        )
        return False

    if getattr(args, "deepspeed", False):
        logger.warning(f"{route_label}: vram_swap_to_ram is not supported together with DeepSpeed. Ignoring this option.")
        logger.warning(f"{route_label}：vram_swap_to_ram 暂不支持与 DeepSpeed 同时使用，已自动忽略。")
        return False

    if getattr(accelerator, "num_processes", 1) != 1:
        logger.warning(
            f"{route_label}: vram_swap_to_ram is currently limited to single-process training. "
            f"Detected num_processes={accelerator.num_processes}, so the option will be ignored."
        )
        logger.warning(
            f"{route_label}：vram_swap_to_ram 当前仅支持单进程训练。"
            f"检测到 num_processes={accelerator.num_processes}，已自动忽略。"
        )
        return False

    if getattr(args, "full_fp16", False) or getattr(args, "full_bf16", False):
        logger.warning(
            f"{route_label}: vram_swap_to_ram is not enabled together with full_fp16/full_bf16 yet. Ignoring this option."
        )
        logger.warning(f"{route_label}：vram_swap_to_ram 暂不支持与 full_fp16/full_bf16 同时使用，已自动忽略。")
        return False

    if not optimizer_supports_vram_swap(optimizer_name, getattr(args, "optimizer_type", None)):
        logger.warning(
            f"{route_label}: optimizer {optimizer_name} is not compatible with vram_swap_to_ram yet. Ignoring this option."
        )
        logger.warning(f"{route_label}：当前优化器 {optimizer_name} 暂不兼容 vram_swap_to_ram，已自动忽略。")
        return False

    if accelerator.device.type == "cpu":
        logger.warning(f"{route_label}: vram_swap_to_ram requires a GPU/XPU runtime. Ignoring this option.")
        logger.warning(f"{route_label}：vram_swap_to_ram 需要可用的 GPU/XPU 训练设备，已自动忽略。")
        return False

    unwrapped_network = accelerator.unwrap_model(network)
    enable_fn = getattr(unwrapped_network, "enable_vram_swap_to_ram", None)
    if callable(enable_fn):
        enabled = bool(enable_fn())
    else:
        enabled = bool(_enable_vram_swap_generic(unwrapped_network))
    if not enabled:
        logger.warning(f"{route_label}: enable_vram_swap_to_ram() returned false. Ignoring this option.")
        logger.warning(f"{route_label}：enable_vram_swap_to_ram() 返回 false，已自动忽略。")
        return False

    clean_memory_on_device(accelerator.device)
    logger.info(
        f"{route_label}: enabled VRAM Swap to RAM. Native adapter weights now stay on CPU RAM and are moved to the runtime device on demand."
    )
    logger.info(f"{route_label}：已启用 VRAM Swap to RAM。原生适配器权重会常驻 CPU RAM，并在前向时按需拉回训练设备。")
    return True


def _move_registered_tensors_to_cpu(module: nn.Module) -> None:
    for name, param in list(module._parameters.items()):
        if param is None or not isinstance(param, torch.nn.Parameter):
            continue
        if param.device.type != "cpu":
            param.data = param.data.to("cpu")
        if param.grad is not None and param.grad.device.type != "cpu":
            param.grad = param.grad.to("cpu")

    for name, buffer in list(module._buffers.items()):
        if buffer is None or not torch.is_tensor(buffer):
            continue
        if buffer.device.type != "cpu":
            module._buffers[name] = buffer.to("cpu")


def _move_module_tree_to_cpu(module: nn.Module) -> None:
    for child in module.modules():
        _move_registered_tensors_to_cpu(child)


def _enable_vram_swap_generic(network: nn.Module) -> bool:
    lora_modules = list(getattr(network, "text_encoder_loras", []) or []) + list(getattr(network, "unet_loras", []) or [])
    if len(lora_modules) == 0:
        return False

    for lora in lora_modules:
        enable_vram_swap_for_module(lora)
        _move_module_tree_to_cpu(lora)

    for buffer_name, buffer in list(getattr(network, "_buffers", {}).items()):
        if not buffer_name.startswith("vera_shared_"):
            continue
        if buffer is None or not torch.is_tensor(buffer):
            continue
        if buffer.device.type != "cpu":
            network._buffers[buffer_name] = buffer.to("cpu")

    setattr(network, "_vram_swap_to_ram_active", True)
    return True


__all__ = [
    "enable_vram_swap_for_module",
    "forward_conv2d_module",
    "forward_linear_module",
    "forward_supported_module",
    "get_vram_swap_runtime_device",
    "maybe_activate_network_vram_swap",
    "module_uses_vram_swap",
    "network_module_supports_vram_swap",
    "optimizer_supports_vram_swap",
    "parse_boolish",
    "set_vram_swap_runtime_device",
]
