from __future__ import annotations

from typing import Any

import torch

from library.adafactor_fused import copy_stochastic_


def _copy_param_data_(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.dtype == torch.bfloat16:
        copy_stochastic_(target, source.float())
        return
    if target.dtype == torch.float16:
        target.copy_(source.to(dtype=torch.float16))
        return
    target.copy_(source.to(dtype=target.dtype))


def _iter_group_params(param_groups) -> list[torch.Tensor]:
    params: list[torch.Tensor] = []
    for group in list(param_groups or []):
        for param in list(group.get("params", []) or []):
            if isinstance(param, torch.Tensor):
                params.append(param)
    return params


def _is_bitsandbytes_optimizer(optimizer: Any) -> bool:
    inner = optimizer
    visited = set()
    while inner is not None and id(inner) not in visited:
        visited.add(id(inner))
        module_name = str(getattr(inner.__class__, "__module__", "") or "").lower()
        if "bitsandbytes" in module_name:
            return True
        next_inner = getattr(inner, "optimizer", None)
        if next_inner is None:
            next_inner = getattr(inner, "_optimizer", None)
        inner = next_inner
    return False


class FullBf16StochasticOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        model_param_groups,
    ):
        self._optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.defaults = optimizer.defaults
        self._model_param_groups = self._clone_param_groups(model_param_groups)
        self._model_params = _iter_group_params(self._model_param_groups)
        self._master_params = _iter_group_params(self.param_groups)
        if len(self._model_params) != len(self._master_params):
            raise ValueError(
                "full_bf16 stochastic optimizer received mismatched model/master parameter counts."
            )
        self._param_pairs = list(zip(self._model_params, self._master_params))
        self._master_device_aligned = False

    def __getattr__(self, name: str):
        return getattr(self._optimizer, name)

    def _align_master_device(self) -> None:
        if self._master_device_aligned:
            return
        for model_param, master_param in self._param_pairs:
            target_device = model_param.device
            if master_param.device != target_device:
                master_param.data = master_param.data.to(device=target_device)
                if master_param.grad is not None:
                    master_param.grad = master_param.grad.to(device=target_device)
        self._master_device_aligned = True

    @staticmethod
    def _clone_param_groups(param_groups):
        if isinstance(param_groups, dict):
            return [dict(param_groups)]
        groups = []
        for group in list(param_groups or []):
            cloned = dict(group)
            cloned["params"] = list(cloned.get("params", []) or [])
            groups.append(cloned)
        return groups

    def enable_fp32_gradient_accumulation(self) -> None:
        for model_param, _ in self._param_pairs:
            if not getattr(model_param, "requires_grad", False):
                continue
            if hasattr(model_param, "grad_dtype"):
                model_param.grad_dtype = torch.float32

    def sync_model_grads_to_master(self) -> None:
        self._align_master_device()
        for model_param, master_param in self._param_pairs:
            grad = getattr(model_param, "grad", None)
            if grad is None:
                master_param.grad = None
                continue
            grad_fp32 = grad.detach().to(device=master_param.device, dtype=torch.float32)
            if master_param.grad is None:
                master_param.grad = grad_fp32.clone()
            else:
                master_param.grad.copy_(grad_fp32)

    def sync_master_params_to_model(self) -> None:
        self._align_master_device()
        with torch.no_grad():
            for model_param, master_param in self._param_pairs:
                _copy_param_data_(model_param.data, master_param.data)

    def sync_model_params_to_master(self) -> None:
        self._align_master_device()
        with torch.no_grad():
            for model_param, master_param in self._param_pairs:
                master_param.data.copy_(model_param.data.to(device=master_param.device, dtype=torch.float32))

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self._optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        raise RuntimeError("full_bf16 stochastic optimizer does not support adding param groups after initialization.")

    def zero_grad(self, set_to_none: bool = True):
        self._optimizer.zero_grad(set_to_none=set_to_none)
        for model_param, _ in self._param_pairs:
            if set_to_none:
                model_param.grad = None
            elif model_param.grad is not None:
                model_param.grad.zero_()

    def step(self, closure=None):
        self.sync_model_grads_to_master()
        loss = self._optimizer.step(closure)
        self.sync_master_params_to_model()
        return loss

    def train(self):
        train_fn = getattr(self._optimizer, "train", None)
        if callable(train_fn):
            train_fn()

    def eval(self):
        eval_fn = getattr(self._optimizer, "eval", None)
        if callable(eval_fn):
            eval_fn()


def _clone_optimizer_param_groups(trainable_params):
    if isinstance(trainable_params, dict):
        trainable_params = [trainable_params]
    items = list(trainable_params)
    if not items:
        return []
    if all(isinstance(item, dict) for item in items):
        groups = []
        for item in items:
            cloned = dict(item)
            cloned["params"] = list(cloned.get("params", []) or [])
            groups.append(cloned)
        return groups
    return [{"params": list(items)}]


def _build_master_param_groups(model_param_groups):
    master_groups = []
    for group in model_param_groups:
        master_group = {key: value for key, value in group.items() if key != "params"}
        master_group["params"] = []
        for param in group.get("params", []):
            master_param = torch.nn.Parameter(param.detach().float().clone(), requires_grad=param.requires_grad)
            master_group["params"].append(master_param)
        master_groups.append(master_group)
    return master_groups


def wrap_optimizer_if_needed(
    args,
    *,
    optimizer: torch.optim.Optimizer,
    trainable_params,
    logger,
    route_label: str,
):
    if not bool(getattr(args, "full_bf16", False)):
        return optimizer
    if _is_bitsandbytes_optimizer(optimizer):
        logger.warning(
            f"{route_label}: full_bf16 stochastic accumulation is currently incompatible with bitsandbytes optimizers. "
            "Falling back to the legacy full_bf16 behavior for this run."
        )
        logger.warning(
            f"{route_label}：full_bf16 stochastic accumulation 当前与 bitsandbytes 优化器不兼容，"
            "本次运行将回退为原有 full_bf16 行为。"
        )
        return optimizer
    if bool(getattr(args, "deepspeed", False)):
        logger.warning(
            f"{route_label}: full_bf16 stochastic accumulation is not enabled under DeepSpeed yet. "
            "Falling back to the legacy full_bf16 behavior for this run."
        )
        logger.warning(
            f"{route_label}：DeepSpeed 路线暂未启用 full_bf16 stochastic accumulation，"
            "本次运行将继续沿用原有 full_bf16 行为。"
        )
        return optimizer

    model_param_groups = _clone_optimizer_param_groups(trainable_params)
    master_param_groups = _build_master_param_groups(model_param_groups)
    optimizer.param_groups = master_param_groups
    wrapped = FullBf16StochasticOptimizer(optimizer, model_param_groups=model_param_groups)
    wrapped.sync_master_params_to_model()
    logger.info(
        f"{route_label}: enabled full_bf16 stochastic accumulation with FP32 master weights and FP32 gradient accumulation."
    )
    logger.info(f"{route_label}：已启用 full_bf16 stochastic accumulation（FP32 主权重 + FP32 梯度累计）。")
    return wrapped


def activate_training_model_grads_if_needed(
    args,
    *,
    optimizer: Any,
):
    if not bool(getattr(args, "full_bf16", False)):
        return
    inner = optimizer
    visited = set()
    while inner is not None and id(inner) not in visited:
        visited.add(id(inner))
        if isinstance(inner, FullBf16StochasticOptimizer):
            inner.enable_fp32_gradient_accumulation()
            return
        inner = getattr(inner, "optimizer", None)


def unwrap_full_bf16_optimizer(optimizer: Any) -> FullBf16StochasticOptimizer | None:
    inner = optimizer
    visited = set()
    while inner is not None and id(inner) not in visited:
        visited.add(id(inner))
        if isinstance(inner, FullBf16StochasticOptimizer):
            return inner
        inner = getattr(inner, "optimizer", None)
    return None


def get_params_for_grad_clipping(default_params, optimizer: Any):
    wrapped = unwrap_full_bf16_optimizer(optimizer)
    if wrapped is None:
        return default_params
    return wrapped._master_params


def sync_master_grads_to_model_if_needed(optimizer: Any) -> None:
    wrapped = unwrap_full_bf16_optimizer(optimizer)
    if wrapped is None:
        return
    for model_param, master_param in wrapped._param_pairs:
        if master_param.grad is None:
            model_param.grad = None
            continue
        master_grad = master_param.grad.detach().to(device=model_param.device, dtype=torch.float32)
        if model_param.grad is None:
            model_param.grad = master_grad.clone()
        else:
            model_param.grad.copy_(master_grad)


__all__ = [
    "FullBf16StochasticOptimizer",
    "activate_training_model_grads_if_needed",
    "get_params_for_grad_clipping",
    "sync_master_grads_to_model_if_needed",
    "unwrap_full_bf16_optimizer",
    "wrap_optimizer_if_needed",
]
