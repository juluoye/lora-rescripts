from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import logging
import math
from typing import Any

import torch
import transformers

from library import lulynx_optimizer_compat
from mikazuki.utils.runtime_mode import infer_attention_runtime_mode, is_amd_rocm_runtime, is_intel_xpu_runtime


def _requires_experimental_runtime_fallback(raw_optimizer_type: str) -> bool:
    lowered = str(raw_optimizer_type or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith("pytorch_optimizer."):
        return True
    return (
        lowered.startswith("bitsandbytes.")
        or "8bit" in lowered
        or "paged" in lowered
        or "bitsandbytes" in lowered
        or "ademamix" in lowered
    )


def resolve_optimizer_type(args: argparse.Namespace, logger: logging.Logger) -> str:
    runtime_mode = infer_attention_runtime_mode()
    experimental_runtime = is_amd_rocm_runtime(runtime_mode) or is_intel_xpu_runtime(runtime_mode)
    runtime_label = "AMD ROCm" if is_amd_rocm_runtime(runtime_mode) else ("Intel XPU" if is_intel_xpu_runtime(runtime_mode) else "")

    optimizer_type = args.optimizer_type
    if experimental_runtime and args.use_8bit_adam:
        logger.warning(f"{runtime_label} experimental runtime does not support use_8bit_adam. Falling back to AdamW.")
        logger.warning(f"{runtime_label} 实验运行时不支持 use_8bit_adam，已自动回退为 AdamW。")
        args.use_8bit_adam = False

    if args.use_8bit_adam:
        assert (
            not args.use_lion_optimizer
        ), "both option use_8bit_adam and use_lion_optimizer are specified / use_8bit_adamとuse_lion_optimizerの両方のオプションが指定されています / use_8bit_adam 与 use_lion_optimizer 不能同时指定"
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_8bit_adam and optimizer_type are specified / use_8bit_adamとoptimizer_typeの両方のオプションが指定されています / use_8bit_adam 与 optimizer_type 不能同时指定"
        optimizer_type = "AdamW8bit"
    elif args.use_lion_optimizer:
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_lion_optimizer and optimizer_type are specified / use_lion_optimizerとoptimizer_typeの両方のオプションが指定されています / use_lion_optimizer 与 optimizer_type 不能同时指定"
        optimizer_type = "Lion"

    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"

    if experimental_runtime and _requires_experimental_runtime_fallback(optimizer_type):
        logger.warning(
            f"{runtime_label} experimental runtime does not support optimizer_type={optimizer_type}. Falling back to AdamW."
        )
        logger.warning(
            f"{runtime_label} 实验运行时不支持 optimizer_type={optimizer_type}，已自动回退为 AdamW。"
        )
        optimizer_type = "AdamW"
        args.optimizer_type = "AdamW"

    return optimizer_type.lower()


def validate_optimizer_choice(args: argparse.Namespace, optimizer_type: str) -> None:
    if args.fused_backward_pass:
        assert (
            optimizer_type == "Adafactor".lower()
        ), "fused_backward_pass currently only works with optimizer_type Adafactor / fused_backward_passは現在optimizer_type Adafactorでのみ機能します / fused_backward_pass 当前仅支持 optimizer_type=Adafactor"
        assert (
            args.gradient_accumulation_steps == 1
        ), "fused_backward_pass does not work with gradient_accumulation_steps > 1 / fused_backward_passはgradient_accumulation_steps>1では機能しません / fused_backward_pass 不支持 gradient_accumulation_steps > 1"


_EPSILON_KEYS = frozenset({"eps", "eps2", "eps_floor", "epsilon"})
_WEIGHT_DECAY_KEYS = frozenset({"weight_decay", "stable_weight_decay"})
_BETA_KEYS = frozenset({"betas"})
_SAFE_MIN_EPS = float(torch.finfo(torch.float32).tiny)
_AGGRESSIVE_CLIP_KEYS = frozenset({"adaptive_clip", "clip_threshold"})
_DEFAULT_EXPERIMENTAL_PYTORCH_OPTIMIZER_ARGS: dict[str, dict[str, Any]] = {
    "compass": {
        "amp_fac": 2.0,
        "eps": _SAFE_MIN_EPS,
    },
    "fcompass": {
        "amp_fac": 1.5,
        "eps": _SAFE_MIN_EPS,
        "weight_decouple": True,
    },
    "fishmonger": {
        "eps": _SAFE_MIN_EPS,
        "weight_decouple": True,
    },
    "farmscrop": {
        "eps": _SAFE_MIN_EPS,
        "theta": 0.999,
    },
    "compassplus": {
        "amp_fac": 2.0,
        "eps": _SAFE_MIN_EPS,
        "centralize_gradients": True,
    },
    "ranger21": {
        "eps": _SAFE_MIN_EPS,
    },
    "ademamix": {
        "eps": _SAFE_MIN_EPS,
    },
}


def _normalize_optimizer_base_name(raw_optimizer_type: str) -> str:
    value = str(raw_optimizer_type or "").strip()
    if not value:
        return ""
    return value.rsplit(".", 1)[-1].lower()


def _coerce_finite_float(
    value: Any,
    *,
    logger: logging.Logger,
    key: str,
    optimizer_label: str,
) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(
            f"Ignoring non-numeric optimizer arg {key}={value!r} for {optimizer_label}. "
            "Please pass a finite numeric value."
        )
        return None

    if not math.isfinite(parsed):
        logger.warning(
            f"Ignoring non-finite optimizer arg {key}={value!r} for {optimizer_label}. "
            "Please pass a finite numeric value."
        )
        return None

    return parsed


def _sanitize_optimizer_kwargs(
    optimizer_kwargs: dict[str, Any],
    *,
    optimizer_type: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    sanitized = dict(optimizer_kwargs)
    optimizer_label = optimizer_type or "optimizer"
    optimizer_base = _normalize_optimizer_base_name(optimizer_type)

    defaults = _DEFAULT_EXPERIMENTAL_PYTORCH_OPTIMIZER_ARGS.get(optimizer_base)
    if defaults:
        for key, value in defaults.items():
            if key not in sanitized:
                sanitized[key] = value
                logger.info(f"{optimizer_label}: auto-injected {key}={value} for safer default behavior.")

    for key in list(sanitized.keys()):
        value = sanitized[key]

        if key in _EPSILON_KEYS:
            parsed = _coerce_finite_float(value, logger=logger, key=key, optimizer_label=optimizer_label)
            if parsed is None or parsed <= 0:
                if parsed is not None:
                    logger.warning(
                        f"{optimizer_label}: {key}={value!r} is not positive. "
                        f"Automatically overriding to float32 tiny ({_SAFE_MIN_EPS})."
                    )
                sanitized[key] = _SAFE_MIN_EPS
            else:
                sanitized[key] = max(parsed, _SAFE_MIN_EPS)
            continue

        if key in _WEIGHT_DECAY_KEYS:
            parsed = _coerce_finite_float(value, logger=logger, key=key, optimizer_label=optimizer_label)
            if parsed is None:
                sanitized.pop(key, None)
                continue
            if parsed < 0:
                logger.warning(
                    f"{optimizer_label}: {key}={value!r} is negative. Automatically clamping it to 0.0."
                )
                parsed = 0.0
            sanitized[key] = parsed
            continue

        if key in _AGGRESSIVE_CLIP_KEYS:
            parsed = _coerce_finite_float(value, logger=logger, key=key, optimizer_label=optimizer_label)
            if parsed is None:
                sanitized.pop(key, None)
                continue
            if parsed <= 0:
                logger.warning(
                    f"{optimizer_label}: {key}={value!r} must be positive. Removing this optimizer arg."
                )
                sanitized.pop(key, None)
                continue
            sanitized[key] = parsed
            continue

        if key in _BETA_KEYS:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                logger.warning(
                    f"{optimizer_label}: betas={value!r} is invalid. Expected a pair like (0.9, 0.999). "
                    "Removing this optimizer arg."
                )
                sanitized.pop(key, None)
                continue

            beta_values: list[float] = []
            valid = True
            for index, beta_value in enumerate(value):
                parsed = _coerce_finite_float(
                    beta_value,
                    logger=logger,
                    key=f"{key}[{index}]",
                    optimizer_label=optimizer_label,
                )
                if parsed is None:
                    valid = False
                    break
                if parsed < 0 or parsed >= 1:
                    logger.warning(
                        f"{optimizer_label}: betas[{index}]={beta_value!r} is outside the valid range [0, 1). "
                        "Removing this optimizer arg."
                    )
                    valid = False
                    break
                beta_values.append(parsed)

            if not valid:
                sanitized.pop(key, None)
                continue

            sanitized[key] = tuple(beta_values)
            continue

        if isinstance(value, (int, float)):
            if isinstance(value, bool):
                continue
            if not math.isfinite(float(value)):
                logger.warning(
                    f"{optimizer_label}: removing non-finite optimizer arg {key}={value!r}."
                )
                sanitized.pop(key, None)

    return sanitized


def _filter_kwargs_for_optimizer_signature(
    optimizer_class,
    optimizer_kwargs: dict[str, Any],
    *,
    optimizer_label: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    try:
        signature = inspect.signature(optimizer_class.__init__)
    except (TypeError, ValueError):
        return dict(optimizer_kwargs)

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(optimizer_kwargs)

    supported = {name for name in parameters.keys() if name != "self"}
    filtered: dict[str, Any] = {}
    dropped: list[str] = []

    for key, value in optimizer_kwargs.items():
        if key in supported:
            filtered[key] = value
        else:
            dropped.append(key)

    if dropped:
        logger.warning(
            f"{optimizer_label}: dropping unsupported optimizer kwargs: {', '.join(sorted(dropped))}"
        )

    return filtered


def parse_optimizer_kwargs(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
    optimizer_kwargs: dict[str, Any] = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            key, value = arg.split("=")
            optimizer_kwargs[key] = ast.literal_eval(value)

    configured_weight_decay = getattr(args, "weight_decay", None)
    if configured_weight_decay is not None and "weight_decay" not in optimizer_kwargs:
        try:
            optimizer_kwargs["weight_decay"] = float(configured_weight_decay)
        except (TypeError, ValueError):
            logger.warning(
                f"Ignoring invalid weight_decay value: {configured_weight_decay}. "
                "Please pass a numeric value (for example 0.01)."
            )
    return _sanitize_optimizer_kwargs(optimizer_kwargs, optimizer_type=getattr(args, "optimizer_type", ""), logger=logger)


def _build_lion_optimizer(trainable_params, optimizer_kwargs: dict[str, Any], lr, logger: logging.Logger):
    try:
        import lion_pytorch
    except ImportError:
        raise ImportError("No lion_pytorch / lion_pytorch がインストールされていないようです / 未安装 lion_pytorch")

    logger.info(f"use Lion optimizer | {optimizer_kwargs}")
    optimizer_class = lion_pytorch.Lion
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def _build_8bit_optimizer(
    optimizer_type: str,
    trainable_params,
    optimizer_kwargs: dict[str, Any],
    lr,
    logger: logging.Logger,
):
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです / 未安装 bitsandbytes")

    optimizer_class = None
    optimizer = None

    if optimizer_type == "AdamW8bit".lower():
        logger.info(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = bnb.optim.AdamW8bit
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    elif optimizer_type == "AdamW8bitKahan".lower():
        try:
            from library.adamw_8bit_kahan import AdamW8bitKahan
        except ImportError:
            raise ImportError(
                "AdamW8bitKahan requires bitsandbytes / AdamW8bitKahan には bitsandbytes が必要です / "
                "AdamW8bitKahan 需要 bitsandbytes"
            )
        logger.info(f"use 8-bit AdamW Kahan optimizer | {optimizer_kwargs}")
        optimizer_class = AdamW8bitKahan
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    elif optimizer_type == "SGDNesterov8bit".lower():
        logger.info(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            logger.warning(
                "8-bit SGD with Nesterov must be with momentum, set momentum to 0.9 / "
                "8-bit SGD with Nesterovはmomentum指定が必須のため0.9に設定します / "
                "8-bit SGD with Nesterov 必须设置 momentum，已自动设为 0.9"
            )
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = bnb.optim.SGD8bit
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)
    elif optimizer_type == "Lion8bit".lower():
        logger.info(f"use 8-bit Lion optimizer | {optimizer_kwargs}")
        try:
            optimizer_class = bnb.optim.Lion8bit
        except AttributeError:
            raise AttributeError(
                "No Lion8bit. The version of bitsandbytes installed seems to be old. "
                "Please install 0.38.0 or later. / "
                "Lion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.38.0以上をインストールしてください / "
                "未找到 Lion8bit，bitsandbytes 版本可能过旧，请安装 0.38.0 或更高版本"
            )
    elif optimizer_type == "PagedAdamW8bit".lower():
        logger.info(f"use 8-bit PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            optimizer_class = bnb.optim.PagedAdamW8bit
        except AttributeError:
            raise AttributeError(
                "No PagedAdamW8bit. The version of bitsandbytes installed seems to be old. "
                "Please install 0.39.0 or later. / "
                "PagedAdamW8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください / "
                "未找到 PagedAdamW8bit，bitsandbytes 版本可能过旧，请安装 0.39.0 或更高版本"
            )
    elif optimizer_type == "PagedLion8bit".lower():
        logger.info(f"use 8-bit Paged Lion optimizer | {optimizer_kwargs}")
        try:
            optimizer_class = bnb.optim.PagedLion8bit
        except AttributeError:
            raise AttributeError(
                "No PagedLion8bit. The version of bitsandbytes installed seems to be old. "
                "Please install 0.39.0 or later. / "
                "PagedLion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください / "
                "未找到 PagedLion8bit，bitsandbytes 版本可能过旧，请安装 0.39.0 或更高版本"
            )

    if optimizer_class is not None and optimizer is None:
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    if optimizer_class is None or optimizer is None:
        return None
    return optimizer_class, optimizer


def _build_paged_optimizer(
    optimizer_name: str,
    trainable_params,
    optimizer_kwargs: dict[str, Any],
    lr,
):
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです / 未安装 bitsandbytes")

    if optimizer_name == "PagedAdamW":
        optimizer_class_name = "PagedAdamW"
        version_hint = "0.39.0"
    else:
        optimizer_class_name = "PagedAdamW32bit"
        version_hint = "0.39.0"

    try:
        optimizer_class = getattr(bnb.optim, optimizer_class_name)
    except AttributeError:
        raise AttributeError(
            f"No {optimizer_class_name}. The version of bitsandbytes installed seems to be old. "
            f"Please install {version_hint} or later. / "
            f"{optimizer_class_name}が定義されていません。インストールされているbitsandbytesのバージョンが古いようです。{version_hint}以上をインストールしてください / "
            f"未找到 {optimizer_class_name}，bitsandbytes 版本可能过旧，请安装 {version_hint} 或更高版本"
        )

    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def _build_sgd_nesterov_optimizer(trainable_params, optimizer_kwargs: dict[str, Any], lr, logger: logging.Logger):
    logger.info(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
    if "momentum" not in optimizer_kwargs:
        logger.info(
            "SGD with Nesterov must be with momentum, set momentum to 0.9 / "
            "SGD with Nesterovはmomentum指定が必須のため0.9に設定します / "
            "SGD with Nesterov 必须设置 momentum，已自动设为 0.9"
        )
        optimizer_kwargs["momentum"] = 0.9

    optimizer_class = torch.optim.SGD
    optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)
    return optimizer_class, optimizer


def _warn_for_dadapt_or_prodigy_learning_rate(trainable_params, lr, logger: logging.Logger) -> None:
    actual_lr = lr
    lr_count = 1
    if type(trainable_params) == list and type(trainable_params[0]) == dict:
        lrs = set()
        actual_lr = trainable_params[0].get("lr", actual_lr)
        for group in trainable_params:
            lrs.add(group.get("lr", actual_lr))
        lr_count = len(lrs)

    if actual_lr <= 0.1:
        logger.warning(
            "learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0 / "
            "学習率が低すぎるようです。D-AdaptationまたはProdigyの使用時は1.0前後の値を指定してください / "
            f"学习率过低。使用 D-Adaptation 或 Prodigy 时建议设为接近 1.0: lr={actual_lr}"
        )
        logger.warning("recommend option: lr=1.0 / 推奨は1.0です / 推荐值：lr=1.0")
    if lr_count > 1:
        logger.warning(
            "when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), "
            "only the first one will take effect / "
            "D-AdaptationまたはProdigyで複数の学習率を指定した場合（Text EncoderとU-Netなど）、最初の学習率のみが有効になります / "
            f"使用 dadaptation 指定多个学习率时（如 Text Encoder 与 U-Net），仅第一个会生效: lr={actual_lr}"
        )


def _build_dadapt_optimizer(
    optimizer_type: str,
    trainable_params,
    optimizer_kwargs: dict[str, Any],
    lr,
    logger: logging.Logger,
):
    try:
        import dadaptation
        import dadaptation.experimental as experimental
    except ImportError:
        raise ImportError("No dadaptation / dadaptation がインストールされていないようです / 未安装 dadaptation")

    if optimizer_type == "DAdaptation".lower() or optimizer_type == "DAdaptAdamPreprint".lower():
        optimizer_class = experimental.DAdaptAdamPreprint
        logger.info(f"use D-Adaptation AdamPreprint optimizer | {optimizer_kwargs}")
    elif optimizer_type == "DAdaptAdaGrad".lower():
        if "eps" not in optimizer_kwargs:
            optimizer_kwargs["eps"] = 1e-6
            logger.info("DAdaptAdaGrad requires eps > 0; defaulting to eps=1e-6 for compatibility.")
        elif float(optimizer_kwargs["eps"]) <= 0:
            logger.warning(
                f"DAdaptAdaGrad received non-positive eps={optimizer_kwargs['eps']}. "
                "Automatically overriding to eps=1e-6."
            )
            optimizer_kwargs["eps"] = 1e-6
        optimizer_class = dadaptation.DAdaptAdaGrad
        logger.info(f"use D-Adaptation AdaGrad optimizer | {optimizer_kwargs}")
    elif optimizer_type == "DAdaptAdam".lower():
        optimizer_class = dadaptation.DAdaptAdam
        logger.info(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")
    elif optimizer_type == "DAdaptAdan".lower():
        optimizer_class = dadaptation.DAdaptAdan
        logger.info(f"use D-Adaptation Adan optimizer | {optimizer_kwargs}")
    elif optimizer_type == "DAdaptAdanIP".lower():
        optimizer_class = experimental.DAdaptAdanIP
        logger.info(f"use D-Adaptation AdanIP optimizer | {optimizer_kwargs}")
    elif optimizer_type == "DAdaptLion".lower():
        optimizer_class = dadaptation.DAdaptLion
        logger.info(f"use D-Adaptation Lion optimizer | {optimizer_kwargs}")
    elif optimizer_type == "DAdaptSGD".lower():
        optimizer_class = dadaptation.DAdaptSGD
        logger.info(f"use D-Adaptation SGD optimizer | {optimizer_kwargs}")
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def _build_prodigy_optimizer(args: argparse.Namespace, trainable_params, optimizer_kwargs: dict[str, Any], lr, logger: logging.Logger):
    try:
        import prodigyopt
    except ImportError:
        raise ImportError("No Prodigy / Prodigy がインストールされていないようです / 未安装 Prodigy")

    logger.info(f"use Prodigy optimizer | {optimizer_kwargs}")
    optimizer_class = prodigyopt.Prodigy
    try:
        prodigy_signature = inspect.signature(optimizer_class.__init__)
        supported_kwargs = set(prodigy_signature.parameters.keys())
    except (TypeError, ValueError):
        supported_kwargs = None

    def _supports_prodigy_kwarg(name: str) -> bool:
        return supported_kwargs is None or name in supported_kwargs

    def _inject_prodigy_kwarg(name: str, value) -> None:
        if value in (None, "") or name in optimizer_kwargs:
            return
        if not _supports_prodigy_kwarg(name):
            logger.info(f"Prodigy runtime does not expose kwarg {name}; skipping injected value.")
            return
        optimizer_kwargs[name] = value

    _inject_prodigy_kwarg("d0", getattr(args, "prodigy_d0", None))
    _inject_prodigy_kwarg("d_coef", getattr(args, "prodigy_d_coef", None))
    _inject_prodigy_kwarg("decouple", getattr(args, "lulynx_prodigy_decouple", None))
    _inject_prodigy_kwarg("use_bias_correction", getattr(args, "lulynx_prodigy_use_bias_correction", None))
    _inject_prodigy_kwarg("safeguard_warmup", getattr(args, "lulynx_prodigy_safeguard_warmup", None))
    _inject_prodigy_kwarg("growth_rate", getattr(args, "lulynx_prodigy_growth_rate", None))

    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def _build_dadapt_or_prodigy_optimizer(
    args: argparse.Namespace,
    trainable_params,
    optimizer_type: str,
    optimizer_kwargs: dict[str, Any],
    lr,
    logger: logging.Logger,
):
    _warn_for_dadapt_or_prodigy_learning_rate(trainable_params, lr, logger)
    if optimizer_type.startswith("DAdapt".lower()):
        return _build_dadapt_optimizer(optimizer_type, trainable_params, optimizer_kwargs, lr, logger)
    return _build_prodigy_optimizer(args, trainable_params, optimizer_kwargs, lr, logger)


def _build_adafactor_optimizer(args: argparse.Namespace, trainable_params, optimizer_kwargs: dict[str, Any], lr, logger: logging.Logger):
    if "relative_step" not in optimizer_kwargs:
        optimizer_kwargs["relative_step"] = True
    if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
        logger.info(
            "set relative_step to True because warmup_init is True / "
            "warmup_initがTrueのためrelative_stepをTrueにします / "
            "因 warmup_init=True，已将 relative_step 设为 True"
        )
        optimizer_kwargs["relative_step"] = True
    logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

    if optimizer_kwargs["relative_step"]:
        logger.info("relative_step is true / relative_stepがtrueです / relative_step 已启用")
        if lr != 0.0:
            logger.warning(
                "learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます / "
                "指定 learning rate 将作为 initial_lr 使用"
            )
        args.learning_rate = None

        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            has_group_lr = False
            for group in trainable_params:
                popped = group.pop("lr", None)
                has_group_lr = has_group_lr or (popped is not None)

            if has_group_lr:
                logger.warning(
                    "unet_lr and text_encoder_lr are ignored / "
                    "unet_lrとtext_encoder_lrは無視されます / "
                    "unet_lr 和 text_encoder_lr 将被忽略"
                )
                args.unet_lr = None
                args.text_encoder_lr = None

        if args.lr_scheduler != "adafactor":
            logger.info("use adafactor_scheduler / スケジューラにadafactor_schedulerを使用します / 使用 adafactor_scheduler")
        args.lr_scheduler = f"adafactor:{lr}"
        lr = None
    else:
        if args.max_grad_norm != 0.0:
            logger.warning(
                "because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / "
                "max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません / "
                "由于设置了 max_grad_norm，clip_grad_norm 会被启用，建议设为 0 以关闭"
            )
        if args.lr_scheduler != "constant_with_warmup":
            logger.warning(
                "constant_with_warmup will be good / "
                "スケジューラはconstant_with_warmupが良いかもしれません / "
                "建议将调度器设为 constant_with_warmup"
            )
        if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
            logger.warning(
                "clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません / 建议将 clip_threshold 设为 1.0"
            )

    optimizer_class = transformers.optimization.Adafactor
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def _build_schedulefree_optimizer(
    optimizer_type: str,
    trainable_params,
    optimizer_kwargs: dict[str, Any],
    lr,
    logger: logging.Logger,
):
    try:
        import schedulefree as sf
    except ImportError:
        raise ImportError("No schedulefree / schedulefreeがインストールされていないようです / 未安装 schedulefree")

    if optimizer_type == "RAdamScheduleFree".lower():
        optimizer_class = sf.RAdamScheduleFree
        logger.info(f"use RAdamScheduleFree optimizer | {optimizer_kwargs}")
    elif optimizer_type == "AdamWScheduleFree".lower():
        optimizer_class = sf.AdamWScheduleFree
        logger.info(f"use AdamWScheduleFree optimizer | {optimizer_kwargs}")
    elif optimizer_type == "SGDScheduleFree".lower():
        optimizer_class = sf.SGDScheduleFree
        logger.info(f"use SGDScheduleFree optimizer | {optimizer_kwargs}")
    else:
        return None

    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def _build_custom_optimizer(args: argparse.Namespace, trainable_params, optimizer_kwargs: dict[str, Any], lr, logger: logging.Logger):
    case_sensitive_optimizer_type = args.optimizer_type
    logger.info(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

    if "." not in case_sensitive_optimizer_type:
        optimizer_module = torch.optim
    else:
        values = case_sensitive_optimizer_type.split(".")
        optimizer_module = importlib.import_module(".".join(values[:-1]))
        case_sensitive_optimizer_type = values[-1]

    optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
    filtered_kwargs = _filter_kwargs_for_optimizer_signature(
        optimizer_class,
        optimizer_kwargs,
        optimizer_label=case_sensitive_optimizer_type,
        logger=logger,
    )
    optimizer = optimizer_class(trainable_params, lr=lr, **filtered_kwargs)
    return optimizer_class, optimizer


def build_optimizer(
    args: argparse.Namespace,
    trainable_params,
    optimizer_type: str,
    optimizer_kwargs: dict[str, Any],
    lr,
    logger: logging.Logger,
):
    if lulynx_optimizer_compat.is_supported_optimizer_name(getattr(args, "optimizer_type", "")):
        compat_result = lulynx_optimizer_compat.build_optimizer(
            args,
            trainable_params,
            optimizer_kwargs,
            lr,
            logger,
        )
        if compat_result is not None:
            return compat_result

    if optimizer_type == "Lion".lower():
        return _build_lion_optimizer(trainable_params, optimizer_kwargs, lr, logger)

    if optimizer_type == "AdamW8bitKahan".lower() or optimizer_type.endswith("8bit".lower()):
        result = _build_8bit_optimizer(optimizer_type, trainable_params, optimizer_kwargs, lr, logger)
        if result is not None:
            return result
    elif optimizer_type == "PagedAdamW".lower():
        logger.info(f"use PagedAdamW optimizer | {optimizer_kwargs}")
        return _build_paged_optimizer("PagedAdamW", trainable_params, optimizer_kwargs, lr)
    elif optimizer_type == "PagedAdamW32bit".lower():
        logger.info(f"use 32-bit PagedAdamW optimizer | {optimizer_kwargs}")
        return _build_paged_optimizer("PagedAdamW32bit", trainable_params, optimizer_kwargs, lr)
    elif optimizer_type == "SGDNesterov".lower():
        return _build_sgd_nesterov_optimizer(trainable_params, optimizer_kwargs, lr, logger)
    elif optimizer_type.startswith("DAdapt".lower()) or optimizer_type == "Prodigy".lower():
        return _build_dadapt_or_prodigy_optimizer(args, trainable_params, optimizer_type, optimizer_kwargs, lr, logger)
    elif optimizer_type == "Adafactor".lower():
        return _build_adafactor_optimizer(args, trainable_params, optimizer_kwargs, lr, logger)
    elif optimizer_type == "AdamW".lower():
        logger.info(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
        return optimizer_class, optimizer
    elif optimizer_type.endswith("schedulefree".lower()):
        result = _build_schedulefree_optimizer(optimizer_type, trainable_params, optimizer_kwargs, lr, logger)
        if result is not None:
            return result

    return _build_custom_optimizer(args, trainable_params, optimizer_kwargs, lr, logger)
