from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


_SUPPORTED_COMPAT_OPTIMIZERS = frozenset(
    {
        "compass",
        "fcompass",
        "fishmonger",
        "farmscrop",
        "compassplus",
    }
)


def _normalize_optimizer_base_name(raw_optimizer_type: str) -> str:
    value = str(raw_optimizer_type or "").strip()
    if not value:
        return ""
    return value.rsplit(".", 1)[-1].lower()


def is_supported_optimizer_name(raw_optimizer_type: str) -> bool:
    return _normalize_optimizer_base_name(raw_optimizer_type) in _SUPPORTED_COMPAT_OPTIMIZERS


def _unitwise_norm(value: torch.Tensor) -> torch.Tensor:
    if value.ndim <= 1:
        return value.norm().reshape([1] * value.ndim)
    dim = tuple(range(1, value.ndim))
    return value.norm(dim=dim, keepdim=True)


def _apply_gradient_centralization_(grad: torch.Tensor, *, gc_conv_only: bool) -> None:
    if grad.ndim <= 1:
        return
    if gc_conv_only and grad.ndim <= 3:
        return
    dim = tuple(range(1, grad.ndim))
    grad.add_(-grad.mean(dim=dim, keepdim=True))


def _apply_adaptive_gradient_clipping_(
    grad: torch.Tensor,
    param: torch.Tensor,
    *,
    clipping: float,
    eps: float,
) -> None:
    if clipping <= 0:
        return

    grad_fp32 = grad.detach().float()
    param_fp32 = param.detach().float()
    max_norm = _unitwise_norm(param_fp32).clamp_min(eps).mul_(clipping)
    grad_norm = _unitwise_norm(grad_fp32).clamp_min(eps)
    needs_clip = grad_norm > max_norm
    clipped = grad_fp32 * (max_norm / grad_norm)
    grad.copy_(torch.where(needs_clip, clipped, grad_fp32).to(dtype=grad.dtype))


def _pop_prefixed_kwargs(
    optimizer_kwargs: dict[str, Any],
    *,
    accepted_names: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    accepted = set(accepted_names)
    extracted: dict[str, Any] = {}
    remaining = dict(optimizer_kwargs)
    for key in list(remaining.keys()):
        if key not in accepted:
            continue
        extracted[key] = remaining.pop(key)
    return extracted, remaining


class _DelegatingOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.defaults = optimizer.defaults

    def __getattr__(self, name: str):
        return getattr(self._optimizer, name)

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self._optimizer.param_groups = param_groups

    @property
    def state(self):
        return self._optimizer.state

    @state.setter
    def state(self, state):
        self._optimizer.state = state

    @property
    def defaults(self):
        return self._optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self._optimizer.defaults = defaults

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self._optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        self._optimizer.add_param_group(param_group)
        self.param_groups = self._optimizer.param_groups
        self.state = self._optimizer.state
        self.defaults = self._optimizer.defaults

    def zero_grad(self, set_to_none: bool = True):
        return self._optimizer.zero_grad(set_to_none=set_to_none)

    def train(self):
        train_fn = getattr(self._optimizer, "train", None)
        if callable(train_fn):
            train_fn()

    def eval(self):
        eval_fn = getattr(self._optimizer, "eval", None)
        if callable(eval_fn):
            eval_fn()


class _GradientTransformingOptimizer(_DelegatingOptimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        amp_fac: float = 1.0,
        centralize_gradients: bool = False,
        gc_conv_only: bool = False,
        adaptive_clip: float | None = None,
        adaptive_clip_eps: float = 1e-3,
    ):
        super().__init__(optimizer)
        self._amp_fac = float(amp_fac)
        self._centralize_gradients = bool(centralize_gradients)
        self._gc_conv_only = bool(gc_conv_only)
        self._adaptive_clip = None if adaptive_clip is None else float(adaptive_clip)
        self._adaptive_clip_eps = float(adaptive_clip_eps)

    def _transform_grad_(self, param: torch.Tensor, grad: torch.Tensor) -> None:
        if self._amp_fac != 1.0:
            grad.mul_(self._amp_fac)
        if self._centralize_gradients:
            _apply_gradient_centralization_(grad, gc_conv_only=self._gc_conv_only)
        if self._adaptive_clip is not None:
            _apply_adaptive_gradient_clipping_(
                grad,
                param,
                clipping=self._adaptive_clip,
                eps=self._adaptive_clip_eps,
            )

    def step(self, closure=None):
        for group in self._optimizer.param_groups:
            for param in group.get("params", []):
                grad = getattr(param, "grad", None)
                if grad is None:
                    continue
                self._transform_grad_(param, grad)
        return self._optimizer.step(closure)


class Compass(_GradientTransformingOptimizer):
    pass


class FCompass(_GradientTransformingOptimizer):
    pass


class FishMonger(_GradientTransformingOptimizer):
    pass


class FARMScrop(_GradientTransformingOptimizer):
    pass


class CompassPlus(_GradientTransformingOptimizer):
    pass


def build_optimizer(
    args,
    trainable_params,
    optimizer_kwargs: dict[str, Any],
    lr,
    logger,
):
    base_name = _normalize_optimizer_base_name(getattr(args, "optimizer_type", ""))
    if base_name not in _SUPPORTED_COMPAT_OPTIMIZERS:
        return None

    from pytorch_optimizer import AdEMAMix, AdaBelief, Ranger21, StableAdamW, StableSPAM

    transform_kwargs, inner_kwargs = _pop_prefixed_kwargs(
        optimizer_kwargs,
        accepted_names=(
            "amp_fac",
            "centralize_gradients",
            "gc_conv_only",
            "adaptive_clip",
            "adaptive_clip_eps",
        ),
    )
    transform_kwargs.setdefault("adaptive_clip_eps", 1e-3)

    if base_name == "compass":
        optimizer_class = Compass
        inner_class = StableAdamW
        logger.info(f"use Compass compatibility optimizer | inner=StableAdamW | {optimizer_kwargs}")
        inner_optimizer = inner_class(trainable_params, lr=lr, **inner_kwargs)
        optimizer = optimizer_class(inner_optimizer, **transform_kwargs)
        return optimizer_class, optimizer

    if base_name == "fcompass":
        optimizer_class = FCompass
        inner_class = AdaBelief
        logger.info(f"use FCompass compatibility optimizer | inner=AdaBelief | {optimizer_kwargs}")
        inner_optimizer = inner_class(trainable_params, lr=lr, **inner_kwargs)
        optimizer = optimizer_class(inner_optimizer, **transform_kwargs)
        return optimizer_class, optimizer

    if base_name == "fishmonger":
        optimizer_class = FishMonger
        inner_class = AdEMAMix
        logger.info(f"use FishMonger compatibility optimizer | inner=AdEMAMix | {optimizer_kwargs}")
        inner_optimizer = inner_class(trainable_params, lr=lr, **inner_kwargs)
        optimizer = optimizer_class(inner_optimizer, **transform_kwargs)
        return optimizer_class, optimizer

    if base_name == "farmscrop":
        optimizer_class = FARMScrop
        inner_class = StableSPAM
        if "t_max" not in inner_kwargs and getattr(args, "max_train_steps", None):
            inner_kwargs["t_max"] = int(max(1, args.max_train_steps))
        logger.info(f"use FARMScrop compatibility optimizer | inner=StableSPAM | {optimizer_kwargs}")
        inner_optimizer = inner_class(trainable_params, lr=lr, **inner_kwargs)
        optimizer = optimizer_class(inner_optimizer, **transform_kwargs)
        return optimizer_class, optimizer

    optimizer_class = CompassPlus
    inner_class = Ranger21
    if "num_iterations" not in inner_kwargs:
        inner_kwargs["num_iterations"] = int(max(1, getattr(args, "max_train_steps", 1) or 1))
    inner_kwargs.setdefault("disable_lr_scheduler", True)
    logger.info(f"use CompassPlus compatibility optimizer | inner=Ranger21 | {optimizer_kwargs}")
    inner_optimizer = inner_class(trainable_params, lr=lr, **inner_kwargs)
    optimizer = optimizer_class(inner_optimizer, **transform_kwargs)
    return optimizer_class, optimizer


__all__ = [
    "Compass",
    "CompassPlus",
    "FARMScrop",
    "FCompass",
    "FishMonger",
    "build_optimizer",
    "is_supported_optimizer_name",
]
