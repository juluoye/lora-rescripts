from __future__ import annotations

import argparse
import ast
import importlib
import logging
from typing import Any

import torch
import transformers


def resolve_optimizer_type(args: argparse.Namespace) -> str:
    optimizer_type = args.optimizer_type
    if args.use_8bit_adam:
        assert (
            not args.use_lion_optimizer
        ), "both option use_8bit_adam and use_lion_optimizer are specified / use_8bit_adamとuse_lion_optimizerの両方のオプションが指定されています"
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_8bit_adam and optimizer_type are specified / use_8bit_adamとoptimizer_typeの両方のオプションが指定されています"
        optimizer_type = "AdamW8bit"
    elif args.use_lion_optimizer:
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_lion_optimizer and optimizer_type are specified / use_lion_optimizerとoptimizer_typeの両方のオプションが指定されています"
        optimizer_type = "Lion"

    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"
    return optimizer_type.lower()


def validate_optimizer_choice(args: argparse.Namespace, optimizer_type: str) -> None:
    if args.fused_backward_pass:
        assert (
            optimizer_type == "Adafactor".lower()
        ), "fused_backward_pass currently only works with optimizer_type Adafactor / fused_backward_passは現在optimizer_type Adafactorでのみ機能します"
        assert (
            args.gradient_accumulation_steps == 1
        ), "fused_backward_pass does not work with gradient_accumulation_steps > 1 / fused_backward_passはgradient_accumulation_steps>1では機能しません"


def parse_optimizer_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    optimizer_kwargs: dict[str, Any] = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            key, value = arg.split("=")
            optimizer_kwargs[key] = ast.literal_eval(value)
    return optimizer_kwargs


def _build_lion_optimizer(trainable_params, optimizer_kwargs: dict[str, Any], lr, logger: logging.Logger):
    try:
        import lion_pytorch
    except ImportError:
        raise ImportError("No lion_pytorch / lion_pytorch がインストールされていないようです")

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
        raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")

    optimizer_class = None
    optimizer = None

    if optimizer_type == "AdamW8bit".lower():
        logger.info(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = bnb.optim.AdamW8bit
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    elif optimizer_type == "SGDNesterov8bit".lower():
        logger.info(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            logger.warning(
                "8-bit SGD with Nesterov must be with momentum, set momentum to 0.9 / "
                "8-bit SGD with Nesterovはmomentum指定が必須のため0.9に設定します"
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
                "No Lion8bit. The version of bitsandbytes installed seems to be old. Please install 0.38.0 or later. / "
                "Lion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.38.0以上をインストールしてください"
            )
    elif optimizer_type == "PagedAdamW8bit".lower():
        logger.info(f"use 8-bit PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            optimizer_class = bnb.optim.PagedAdamW8bit
        except AttributeError:
            raise AttributeError(
                "No PagedAdamW8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / "
                "PagedAdamW8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
            )
    elif optimizer_type == "PagedLion8bit".lower():
        logger.info(f"use 8-bit Paged Lion optimizer | {optimizer_kwargs}")
        try:
            optimizer_class = bnb.optim.PagedLion8bit
        except AttributeError:
            raise AttributeError(
                "No PagedLion8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / "
                "PagedLion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
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
        raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")

    try:
        optimizer_class = getattr(bnb.optim, optimizer_name)
    except AttributeError:
        raise AttributeError(
            f"No {optimizer_name}. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / "
            f"{optimizer_name}が定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
        )

    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def _build_sgd_nesterov_optimizer(trainable_params, optimizer_kwargs: dict[str, Any], lr, logger: logging.Logger):
    logger.info(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
    if "momentum" not in optimizer_kwargs:
        logger.info(
            "SGD with Nesterov must be with momentum, set momentum to 0.9 / "
            "SGD with Nesterovはmomentum指定が必須のため0.9に設定します"
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
            f"学習率が低すぎるようです。D-AdaptationまたはProdigyの使用時は1.0前後の値を指定してください: lr={actual_lr}"
        )
        logger.warning("recommend option: lr=1.0 / 推奨は1.0です")
    if lr_count > 1:
        logger.warning(
            "when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), "
            f"only the first one will take effect / D-AdaptationまたはProdigyで複数の学習率を指定した場合（Text EncoderとU-Netなど）、最初の学習率のみが有効になります: lr={actual_lr}"
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
        raise ImportError("No dadaptation / dadaptation がインストールされていないようです")

    if optimizer_type == "DAdaptation".lower() or optimizer_type == "DAdaptAdamPreprint".lower():
        optimizer_class = experimental.DAdaptAdamPreprint
        logger.info(f"use D-Adaptation AdamPreprint optimizer | {optimizer_kwargs}")
    elif optimizer_type == "DAdaptAdaGrad".lower():
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


def _build_prodigy_optimizer(trainable_params, optimizer_kwargs: dict[str, Any], lr, logger: logging.Logger):
    try:
        import prodigyopt
    except ImportError:
        raise ImportError("No Prodigy / Prodigy がインストールされていないようです")

    logger.info(f"use Prodigy optimizer | {optimizer_kwargs}")
    optimizer_class = prodigyopt.Prodigy
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def _build_dadapt_or_prodigy_optimizer(
    trainable_params,
    optimizer_type: str,
    optimizer_kwargs: dict[str, Any],
    lr,
    logger: logging.Logger,
):
    _warn_for_dadapt_or_prodigy_learning_rate(trainable_params, lr, logger)
    if optimizer_type.startswith("DAdapt".lower()):
        return _build_dadapt_optimizer(optimizer_type, trainable_params, optimizer_kwargs, lr, logger)
    return _build_prodigy_optimizer(trainable_params, optimizer_kwargs, lr, logger)


def _build_adafactor_optimizer(args: argparse.Namespace, trainable_params, optimizer_kwargs: dict[str, Any], lr, logger: logging.Logger):
    if "relative_step" not in optimizer_kwargs:
        optimizer_kwargs["relative_step"] = True
    if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
        logger.info(
            "set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします"
        )
        optimizer_kwargs["relative_step"] = True
    logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

    if optimizer_kwargs["relative_step"]:
        logger.info("relative_step is true / relative_stepがtrueです")
        if lr != 0.0:
            logger.warning("learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
        args.learning_rate = None

        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            has_group_lr = False
            for group in trainable_params:
                popped = group.pop("lr", None)
                has_group_lr = has_group_lr or (popped is not None)

            if has_group_lr:
                logger.warning("unet_lr and text_encoder_lr are ignored / unet_lrとtext_encoder_lrは無視されます")
                args.unet_lr = None
                args.text_encoder_lr = None

        if args.lr_scheduler != "adafactor":
            logger.info("use adafactor_scheduler / スケジューラにadafactor_schedulerを使用します")
        args.lr_scheduler = f"adafactor:{lr}"
        lr = None
    else:
        if args.max_grad_norm != 0.0:
            logger.warning(
                "because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / "
                "max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
            )
        if args.lr_scheduler != "constant_with_warmup":
            logger.warning("constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
        if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
            logger.warning("clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")

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
        raise ImportError("No schedulefree / schedulefreeがインストールされていないようです")

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
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer


def build_optimizer(
    args: argparse.Namespace,
    trainable_params,
    optimizer_type: str,
    optimizer_kwargs: dict[str, Any],
    lr,
    logger: logging.Logger,
):
    if optimizer_type == "Lion".lower():
        return _build_lion_optimizer(trainable_params, optimizer_kwargs, lr, logger)

    if optimizer_type.endswith("8bit".lower()):
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
        return _build_dadapt_or_prodigy_optimizer(trainable_params, optimizer_type, optimizer_kwargs, lr, logger)
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
