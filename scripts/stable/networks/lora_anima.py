# LoRA network module for Anima
import ast
import json
import math
import os
import re
import weakref
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn.functional as F
from library import network_vram_swap_util
from library.utils import setup_logging
from networks.lora_flux import LoRAModule, LoRAInfModule
from torch.nn.modules.module import _IncompatibleKeys

import logging

setup_logging()
logger = logging.getLogger(__name__)


TRAIN_NORM_PREFIX_ANIMA = "train_norm_unet"
TRAIN_NORM_PREFIX_TEXT_ENCODER = "train_norm_te"
COMFYUI_TRAIN_NORM_PREFIX_ANIMA = "lora_unet"
LOKR_FULL_MATRIX_DIM_SENTINEL = 100000


def _lokr_factorization(dimension: int, factor: int) -> Tuple[int, int]:
    dimension = max(1, int(dimension))
    factor = int(factor)
    if factor > 0 and dimension % factor == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n

    if factor < 0:
        factor = dimension

    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n

    if m > n:
        n, m = m, n
    return m, n


def _parse_bool_arg(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_pissa_method(value) -> str:
    method = str(value or "rsvd").strip().lower() or "rsvd"
    if method not in {"rsvd", "svd"}:
        return "rsvd"
    return method


def _normalize_pissa_export_mode(value) -> str:
    text = str(value or "").strip().lower()
    if text in {"approx", "fast", "lora_fast", "fast_approx", "approximate"}:
        return "approx"
    if "快速" in str(value or ""):
        return "approx"
    return "lossless"


def _vera_kaiming_uniform(shape: Tuple[int, ...], seed: int) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.float32)
    fan_in = shape[1] if len(shape) >= 2 else shape[0]
    fan_in = max(1, int(fan_in))
    bound = math.sqrt(6.0 / float(fan_in))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


@dataclass
class TrainNormRef:
    lora_name: str
    original_name: str
    org_module: torch.nn.Module
    base_params: Dict[str, torch.Tensor] = field(default_factory=dict)

    def named_parameters(self):
        return self.org_module.named_parameters(recurse=False)


def _snapshot_train_norm_base_params(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in module.named_parameters(recurse=False)}


def _train_norm_name_to_comfyui_name(lora_name: str) -> Optional[str]:
    if lora_name.startswith(TRAIN_NORM_PREFIX_ANIMA):
        return COMFYUI_TRAIN_NORM_PREFIX_ANIMA + lora_name[len(TRAIN_NORM_PREFIX_ANIMA):]
    return None


def _comfyui_name_to_train_norm_name(lora_name: str) -> Optional[str]:
    if lora_name.startswith(COMFYUI_TRAIN_NORM_PREFIX_ANIMA):
        return TRAIN_NORM_PREFIX_ANIMA + lora_name[len(COMFYUI_TRAIN_NORM_PREFIX_ANIMA):]
    return None


def _is_comfyui_train_norm_key(key: str) -> bool:
    if "." not in key:
        return False
    lora_name, param_name = key.rsplit(".", 1)
    return (
        lora_name.startswith(COMFYUI_TRAIN_NORM_PREFIX_ANIMA)
        and "norm" in lora_name
        and param_name in {"diff", "diff_b"}
    )


class TrainNormParamProxy(torch.nn.Module):
    def __init__(self, refs: List[TrainNormRef], proxy_name: str):
        super().__init__()
        self.refs = refs
        self.proxy_name = proxy_name

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        del recurse
        memo = set()
        prefix = f"{prefix}." if prefix else ""
        for ref in self.refs:
            for name, param in ref.named_parameters():
                if remove_duplicate and id(param) in memo:
                    continue
                memo.add(id(param))
                yield f"{prefix}{ref.lora_name}.{name}", param


class LoKrModule(torch.nn.Module):
    """
    Anima-specific LoKr module implemented as a linear-layer injector.
    This intentionally targets Linear layers only, matching the current
    verified Anima DiT route.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        factor=8,
        full_matrix=False,
        decompose_both=False,
        unbalanced_factorization=False,
        lokr_shape: Optional[Dict[str, Tuple[int, ...]]] = None,
    ):
        super().__init__()
        if org_module.__class__.__name__ != "Linear":
            raise ValueError(f"LoKrModule only supports Linear layers for Anima, got {org_module.__class__.__name__}")

        self.lora_name = lora_name
        self.lora_dim = int(lora_dim if lora_dim is not None else 4)
        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.full_matrix = bool(full_matrix)
        self.decompose_both = bool(decompose_both)
        self.unbalanced_factorization = bool(unbalanced_factorization)

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.in_features = int(in_dim)
        self.out_features = int(out_dim)
        self.factor = int(factor) if factor is not None else 8
        self.lokr_shape = {str(k): tuple(v) for k, v in (lokr_shape or {}).items()}
        self._build_parameters()

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().cpu().item()
        alpha = self.lora_dim if alpha is None or alpha == 0 else float(alpha)
        if getattr(self, "w1_direct", False) and getattr(self, "w2_direct", False):
            alpha = float(self.lora_dim)
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("lokr_rank", torch.tensor(self.lora_dim))

    @staticmethod
    def _find_factor(in_features: int, out_features: int, target_factor: int) -> int:
        candidates = []
        if target_factor > 0:
            candidates.append(target_factor)
        candidates.extend([16, 12, 8, 6, 4, 3, 2, 1])

        seen = set()
        for factor in candidates:
            if factor in seen or factor <= 0:
                continue
            seen.add(factor)
            if in_features % factor == 0 and out_features % factor == 0:
                return factor
        return 1

    def _build_parameters(self) -> None:
        if self.lokr_shape:
            self.w1_direct = "w1" in self.lokr_shape
            self.w2_direct = "w2" in self.lokr_shape
            missing = []
            if not self.w1_direct:
                for key in ("w1_a", "w1_b"):
                    if key not in self.lokr_shape:
                        missing.append(key)
            if not self.w2_direct:
                for key in ("w2_a", "w2_b"):
                    if key not in self.lokr_shape:
                        missing.append(key)
            if missing:
                raise ValueError(
                    f"LoKrModule {self.lora_name}: incomplete LoKr shape metadata, missing {', '.join(missing)}"
                )

            if self.w1_direct:
                self.lokr_w1 = torch.nn.Parameter(torch.empty(*self.lokr_shape["w1"]))
                torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
            else:
                self.lokr_w1_a = torch.nn.Parameter(torch.empty(*self.lokr_shape["w1_a"]))
                self.lokr_w1_b = torch.nn.Parameter(torch.empty(*self.lokr_shape["w1_b"]))
                torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

            if self.w2_direct:
                self.lokr_w2 = torch.nn.Parameter(torch.empty(*self.lokr_shape["w2"]))
                torch.nn.init.zeros_(self.lokr_w2)
            else:
                self.lokr_w2_a = torch.nn.Parameter(torch.empty(*self.lokr_shape["w2_a"]))
                self.lokr_w2_b = torch.nn.Parameter(torch.empty(*self.lokr_shape["w2_b"]))
                torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
                torch.nn.init.zeros_(self.lokr_w2_b)

            self.lokr_out_dim = self._materialize_w2().shape[0]
            self.lokr_in_dim = self._materialize_w2().shape[1]
            return

        if not self.decompose_both and not self.full_matrix and not self.unbalanced_factorization:
            factor = self._find_factor(self.in_features, self.out_features, self.factor)
            self.factor = factor
            self.lokr_in_dim = self.in_features // factor
            self.lokr_out_dim = self.out_features // factor
            self.w1_direct = True
            self.w2_direct = True
            self.lokr_w1 = torch.nn.Parameter(torch.empty(factor, factor))
            self.lokr_w2 = torch.nn.Parameter(torch.empty(self.lokr_out_dim, self.lokr_in_dim))
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lokr_w2)
            return

        out_l, out_k = _lokr_factorization(self.out_features, self.factor)
        in_m, in_n = _lokr_factorization(self.in_features, self.factor)
        if self.unbalanced_factorization:
            out_l, out_k = out_k, out_l

        self.lokr_out_dim = out_k
        self.lokr_in_dim = in_n
        w1_threshold = max(out_l, in_m) / 2.0
        w2_threshold = max(out_k, in_n) / 2.0
        self.w1_direct = self.full_matrix or not (self.decompose_both and self.lora_dim < w1_threshold)
        self.w2_direct = self.full_matrix or self.lora_dim >= w2_threshold

        if self.w2_direct and not self.full_matrix and self.decompose_both:
            logger.info(
                "Anima LoKr: lora_dim=%s >= %.1f, using direct lokr_w2 full-matrix branch for %s",
                self.lora_dim,
                w2_threshold,
                self.lora_name,
            )

        if self.w1_direct:
            self.lokr_w1 = torch.nn.Parameter(torch.empty(out_l, in_m))
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            self.lokr_w1_a = torch.nn.Parameter(torch.empty(out_l, self.lora_dim))
            self.lokr_w1_b = torch.nn.Parameter(torch.empty(self.lora_dim, in_m))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

        if self.w2_direct:
            self.lokr_w2 = torch.nn.Parameter(torch.empty(out_k, in_n))
            torch.nn.init.zeros_(self.lokr_w2)
        else:
            self.lokr_w2_a = torch.nn.Parameter(torch.empty(out_k, self.lora_dim))
            self.lokr_w2_b = torch.nn.Parameter(torch.empty(self.lora_dim, in_n))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lokr_w2_b)

    def _materialize_w1(self):
        if getattr(self, "w1_direct", hasattr(self, "lokr_w1")):
            return self.lokr_w1
        return self.lokr_w1_a @ self.lokr_w1_b

    def _materialize_w2(self):
        if getattr(self, "w2_direct", hasattr(self, "lokr_w2")):
            return self.lokr_w2
        return self.lokr_w2_a @ self.lokr_w2_b

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _compute_weight(self, device=None, dtype=None):
        weight = torch.kron(self._materialize_w1(), self._materialize_w2()).reshape(
            self.out_features, self.in_features
        )
        if device is not None or dtype is not None:
            weight = weight.to(device=device or weight.device, dtype=dtype or weight.dtype)
        return weight

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return org_forwarded

        lx = x
        if self.dropout is not None and self.training:
            lx = F.dropout(lx, p=self.dropout)

        weight = self._compute_weight(device=lx.device, dtype=lx.dtype)

        if self.rank_dropout is not None and self.training:
            mask = (torch.rand(weight.shape[0], device=lx.device) > self.rank_dropout).to(dtype=weight.dtype)
            weight = weight * mask.view(-1, 1)
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = F.linear(lx, weight)
        return org_forwarded + lx * self.multiplier * scale

    @torch.no_grad()
    def export_standard_lora_weights(self):
        """Export LoKr as standard LoRA-compatible weights via SVD approximation."""
        full_weight = self._compute_weight(device=torch.device("cpu"), dtype=torch.float32).detach()

        out_dim, in_dim = full_weight.shape
        rank = min(self.lora_dim, out_dim, in_dim)
        if rank <= 0:
            raise ValueError(f"LoKr export: invalid rank {rank} for shape {full_weight.shape}")

        U, S, Vh = torch.linalg.svd(full_weight, full_matrices=False)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        scaled_s = torch.sqrt(S.clamp_min(0))
        up_weight = U * scaled_s.unsqueeze(0)
        down_weight = scaled_s.unsqueeze(1) * Vh

        alpha = torch.tensor(float(self.lora_dim), dtype=torch.float32)
        return down_weight.contiguous(), up_weight.contiguous(), alpha

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class LoKrInfModule(LoKrModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        factor=8,
        full_matrix=False,
        decompose_both=False,
        unbalanced_factorization=False,
        lokr_shape: Optional[Dict[str, Tuple[int, ...]]] = None,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            factor=factor,
            full_matrix=full_matrix,
            decompose_both=decompose_both,
            unbalanced_factorization=unbalanced_factorization,
            lokr_shape=lokr_shape,
        )
        self.org_module_ref = [org_module]
        self.enabled = True
        self.network = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device):
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"].to(torch.float)
        org_dtype = org_sd["weight"].dtype
        org_device = org_sd["weight"].device

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        self.load_state_dict(sd, strict=False)
        adapter_weight = self._compute_weight(device=device, dtype=torch.float)
        org_sd["weight"] = (weight.to(device) + self.multiplier * adapter_weight * self.scale).to(dtype)
        self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        return multiplier * self._compute_weight(device=self.device, dtype=torch.float) * self.scale

    def set_region(self, region):
        self.region = region


class LoRAFAModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            **kwargs,
        )
        self._reinitialize_lora_fa_weights(org_module)

    def _reinitialize_lora_fa_weights(self, org_module: torch.nn.Module) -> None:
        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
        else:
            in_dim = org_module.in_features

        std = math.sqrt(2.0 / (in_dim + self.lora_dim))
        if isinstance(self.lora_down, torch.nn.ModuleList):
            for lora_down in self.lora_down:
                torch.nn.init.normal_(lora_down.weight, std=std)
            for lora_up in self.lora_up:
                torch.nn.init.zeros_(lora_up.weight)
        else:
            torch.nn.init.normal_(self.lora_down.weight, std=std)
            torch.nn.init.zeros_(self.lora_up.weight)

    def get_trainable_params(self):
        if isinstance(self.lora_up, torch.nn.ModuleList):
            return [param for module in self.lora_up for param in module.parameters(recurse=False)]
        return [param for param in self.lora_up.parameters(recurse=False)]

    def requires_grad_(self, requires_grad: bool = True):
        if isinstance(self.lora_up, torch.nn.ModuleList):
            for lora_up in self.lora_up:
                lora_up.requires_grad_(requires_grad)
        else:
            self.lora_up.requires_grad_(requires_grad)

        if isinstance(self.lora_down, torch.nn.ModuleList):
            for lora_down in self.lora_down:
                lora_down.requires_grad_(False)
        else:
            self.lora_down.requires_grad_(False)
        return self


class DoRALoRAModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        bypass_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            **kwargs,
        )
        self.org_module_ref = [org_module]
        self.bypass_mode = _parse_bool_arg(bypass_mode, default=False)
        self.enabled = True
        magnitude = self._compute_weight_norm(org_module.weight.detach().to(dtype=torch.float32))
        self.dora_scale = torch.nn.Parameter(magnitude.to(dtype=torch.float32))

    @staticmethod
    def _compute_weight_norm(weight: torch.Tensor) -> torch.Tensor:
        flat = weight.reshape(weight.shape[0], -1)
        return torch.linalg.norm(flat, dim=1).clamp_min(1e-6)

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _get_delta_weight(self, multiplier=None, rank_dropout_mask: Optional[torch.Tensor] = None):
        if multiplier is None:
            multiplier = self.multiplier

        if isinstance(self.lora_down, torch.nn.ModuleList):
            raise NotImplementedError("DoRA does not support split_dims adapters in Anima.")

        if network_vram_swap_util.module_uses_vram_swap(self):
            work_device = self.org_module_ref[0].weight.device
            work_dtype = self.org_module_ref[0].weight.dtype
            down_weight = self.lora_down.weight.to(device=work_device, dtype=work_dtype)
            up_weight = self.lora_up.weight.to(device=work_device, dtype=work_dtype)
        else:
            down_weight = self.lora_down.weight
            up_weight = self.lora_up.weight
            work_dtype = up_weight.dtype
            work_device = up_weight.device

        if rank_dropout_mask is not None:
            mask = rank_dropout_mask.to(device=work_device, dtype=work_dtype)
            if down_weight.ndim == 2:
                down_weight = down_weight * mask.unsqueeze(1)
                up_weight = up_weight * mask.unsqueeze(0)
            else:
                down_weight = down_weight * mask.view(-1, 1, 1, 1)
                up_weight = up_weight * mask.view(1, -1, 1, 1)

        if down_weight.ndim == 2:
            delta = up_weight @ down_weight
        elif tuple(down_weight.shape[2:]) == (1, 1):
            delta = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            delta = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)

        return delta * (float(multiplier) * float(self.scale))

    def _get_bias_view(self, bias: Optional[torch.Tensor], out: torch.Tensor):
        if bias is None:
            return None
        if out.ndim < 2:
            raise ValueError(f"Unsupported DoRA output ndim: {out.ndim}")

        org_module = self.org_module_ref[0]
        if org_module.__class__.__name__ == "Conv2d":
            return bias.view(1, -1, *([1] * (out.ndim - 2)))

        return bias.view(*([1] * (out.ndim - 1)), -1)

    def _get_scale_view(self, out: torch.Tensor, dtype: torch.dtype, device: torch.device):
        scale = (self.dora_scale.to(device=device, dtype=dtype) / self._merged_weight_norm(device, dtype)).clamp_min(1e-6)
        if out.ndim < 2:
            raise ValueError(f"Unsupported DoRA output ndim: {out.ndim}")

        org_module = self.org_module_ref[0]
        if org_module.__class__.__name__ == "Conv2d":
            return scale.view(1, -1, *([1] * (out.ndim - 2)))

        return scale.view(*([1] * (out.ndim - 1)), -1)

    def _merged_weight_norm(self, device: torch.device, dtype: torch.dtype):
        org_weight = self.org_module_ref[0].weight.to(device=device, dtype=dtype)
        delta = self._get_delta_weight().to(device=device, dtype=dtype)
        merged = org_weight + delta
        return self._compute_weight_norm(merged).detach()

    def _compute_merged_weight(self, device: torch.device, dtype: torch.dtype, multiplier=None):
        org_weight = self.org_module_ref[0].weight.to(device=device, dtype=dtype)
        delta = self._get_delta_weight(multiplier=multiplier).to(device=device, dtype=dtype)
        merged = org_weight + delta
        row_scale = (self.dora_scale.to(device=device, dtype=dtype) / self._compute_weight_norm(merged).detach()).clamp_min(1e-6)
        view_shape = [merged.shape[0]] + [1] * (merged.ndim - 1)
        return merged * row_scale.view(*view_shape)

    def get_weight(self, multiplier=None):
        org_weight = self.org_module_ref[0].weight.to(torch.float32)
        merged_weight = self._compute_merged_weight(org_weight.device, torch.float32, multiplier=multiplier)
        return merged_weight - org_weight

    def merge_to(self, sd, dtype, device):
        org_module = self.org_module_ref[0]
        org_sd = org_module.state_dict()
        org_weight = org_sd["weight"]
        org_dtype = org_weight.dtype
        org_device = org_weight.device

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        down_weight = sd["lora_down.weight"].to(torch.float32).to(device)
        up_weight = sd["lora_up.weight"].to(torch.float32).to(device)
        _mag_key = "dora_magnitude" if "dora_magnitude" in sd else "dora_scale"
        magnitude = sd[_mag_key].to(torch.float32).to(device)

        if down_weight.ndim == 2:
            delta = up_weight @ down_weight
        elif tuple(down_weight.shape[2:]) == (1, 1):
            delta = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            delta = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)

        merged = org_weight.to(torch.float32).to(device) + delta * self.scale * self.multiplier
        row_scale = (magnitude / self._compute_weight_norm(merged).detach()).clamp_min(1e-6)
        view_shape = [merged.shape[0]] + [1] * (merged.ndim - 1)
        org_sd["weight"] = (merged * row_scale.view(*view_shape)).to(dtype)
        org_module.load_state_dict(org_sd)

    def _forward_delta(self, x: torch.Tensor, rank_dropout_mask: Optional[torch.Tensor] = None):
        if network_vram_swap_util.module_uses_vram_swap(self):
            lx = network_vram_swap_util.forward_supported_module(self.lora_down, x)
        else:
            lx = self.lora_down(x)
        if self.dropout is not None and self.training:
            lx = F.dropout(lx, p=self.dropout)
        if rank_dropout_mask is not None:
            mask = rank_dropout_mask.to(device=lx.device, dtype=lx.dtype)
            if self.lora_up.__class__.__name__ == "Conv2d":
                mask = mask.view(1, -1, *([1] * (lx.ndim - 2)))
            else:
                mask = mask.view(*([1] * (lx.ndim - 1)), -1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale
        if network_vram_swap_util.module_uses_vram_swap(self):
            return network_vram_swap_util.forward_supported_module(self.lora_up, lx) * self.multiplier * scale
        return self.lora_up(lx) * self.multiplier * scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)

        org_module = self.org_module_ref[0]
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return org_forwarded

        rank_dropout_mask = None
        if self.rank_dropout is not None and self.training:
            rank_dropout_mask = (torch.rand(self.lora_dim, device=x.device) > self.rank_dropout).to(dtype=torch.float32)

        delta_out = self._forward_delta(x, rank_dropout_mask=rank_dropout_mask)
        bias_view = self._get_bias_view(org_module.bias, org_forwarded)
        if bias_view is None:
            base_without_bias = org_forwarded
        else:
            base_without_bias = org_forwarded - bias_view.to(device=org_forwarded.device, dtype=org_forwarded.dtype)

        row_scale = self._get_scale_view(org_forwarded, dtype=org_forwarded.dtype, device=org_forwarded.device)
        adapted = (base_without_bias + delta_out.to(dtype=org_forwarded.dtype)) * row_scale
        if bias_view is not None:
            adapted = adapted + bias_view.to(device=org_forwarded.device, dtype=org_forwarded.dtype)
        return adapted


class VeraModule(torch.nn.Module):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        network_ref=None,
        d_initial: float = 0.1,
        **kwargs,
    ):
        del alpha, kwargs
        super().__init__()
        if org_module.__class__.__name__ != "Linear":
            raise ValueError("Anima VeRA currently supports only Linear modules.")
        if network_ref is None:
            raise ValueError("Anima VeRA requires a parent network reference for shared projection buffers.")

        self.lora_name = lora_name
        self.multiplier = multiplier
        self.lora_dim = int(lora_dim)
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.org_module = org_module
        self.in_features = int(org_module.in_features)
        self.out_features = int(org_module.out_features)
        self._vera_network_ref = weakref.ref(network_ref)

        network_ref.ensure_vera_shared_buffers(self.out_features, self.in_features)

        self.vera_lambda_b = torch.nn.Parameter(torch.zeros(self.out_features, dtype=torch.float32))
        self.vera_lambda_d = torch.nn.Parameter(torch.full((self.lora_dim,), float(d_initial), dtype=torch.float32))

    def _get_network(self):
        network = self._vera_network_ref()
        if network is None:
            raise RuntimeError("Anima VeRA parent network reference is no longer available.")
        return network

    def _get_sliced_projections(self, device: torch.device, dtype: torch.dtype):
        network = self._get_network()
        vera_A = network.get_vera_shared_A()[:, : self.in_features].to(device=device, dtype=dtype)
        vera_B = network.get_vera_shared_B()[: self.out_features, :].to(device=device, dtype=dtype)
        return vera_A, vera_B

    def get_trainable_params(self):
        return [self.vera_lambda_b, self.vera_lambda_d]

    def requires_grad_(self, requires_grad: bool = True):
        self.vera_lambda_b.requires_grad_(requires_grad)
        self.vera_lambda_d.requires_grad_(requires_grad)
        return self

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def compute_forward_delta(self, x):
        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return None

        work_dtype = self.vera_lambda_d.dtype
        vera_A, vera_B = self._get_sliced_projections(x.device, work_dtype)
        lambda_d = self.vera_lambda_d.to(device=x.device, dtype=work_dtype)
        lambda_b = self.vera_lambda_b.to(device=x.device, dtype=work_dtype)

        hidden = F.linear(x.to(work_dtype), lambda_d.unsqueeze(1) * vera_A)
        if self.dropout is not None and self.training:
            hidden = F.dropout(hidden, p=self.dropout)

        if self.rank_dropout is not None and self.training:
            # Create mask with correct shape for broadcasting
            if hidden.ndim == 3:
                # hidden: (batch, seq_len, lora_dim)
                mask = (torch.rand(self.lora_dim, device=hidden.device) > self.rank_dropout).view(1, 1, -1)
            else:
                # hidden: (batch, lora_dim)
                mask = (torch.rand(self.lora_dim, device=hidden.device) > self.rank_dropout).view(1, -1)
            hidden = hidden * mask.to(hidden.dtype)

        delta = F.linear(hidden, lambda_b.unsqueeze(1) * vera_B)
        return delta * self.multiplier

    def forward(self, x):
        org_forwarded = self.org_forward(x)
        delta = self.compute_forward_delta(x)
        if delta is None:
            return org_forwarded
        return org_forwarded + delta.to(org_forwarded.dtype)

    @torch.no_grad()
    def export_standard_lora_weights(self):
        vera_A, vera_B = self._get_sliced_projections(torch.device("cpu"), torch.float32)
        lambda_d = self.vera_lambda_d.detach().to(device="cpu", dtype=torch.float32)
        lambda_b = self.vera_lambda_b.detach().to(device="cpu", dtype=torch.float32)
        down_weight = lambda_d.unsqueeze(1) * vera_A
        up_weight = lambda_b.unsqueeze(1) * vera_B
        alpha = torch.tensor(float(self.lora_dim), dtype=torch.float32)
        return down_weight, up_weight, alpha


class PiSSAModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        pissa_method: str = "rsvd",
        pissa_niter: int = 2,
        pissa_oversample: int = 8,
        pissa_apply_conv2d: bool = False,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            **kwargs,
        )
        self.pissa_method = _normalize_pissa_method(pissa_method)
        self.pissa_niter = max(0, int(pissa_niter if pissa_niter is not None else 2))
        self.pissa_oversample = max(0, int(pissa_oversample if pissa_oversample is not None else 8))
        self.pissa_apply_conv2d = _parse_bool_arg(pissa_apply_conv2d, default=False)
        self._pissa_initialized = False
        self._apply_pissa_init(org_module)
        self._pissa_initial_lora_down_weight = self.lora_down.weight.detach().clone().to(device="cpu", dtype=torch.float32)
        self._pissa_initial_lora_up_weight = self.lora_up.weight.detach().clone().to(device="cpu", dtype=torch.float32)

    @staticmethod
    def _reshape_weight_to_matrix(weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 2:
            return weight
        if weight.ndim == 4 and tuple(weight.shape[2:]) == (1, 1):
            return weight.squeeze(3).squeeze(2)
        raise ValueError("PiSSA only supports Linear and Conv2d 1x1 weights.")

    @staticmethod
    def _reshape_matrix_to_weight(matrix: torch.Tensor, reference_weight: torch.Tensor) -> torch.Tensor:
        if reference_weight.ndim == 2:
            return matrix
        return matrix.unsqueeze(2).unsqueeze(3)

    @torch.no_grad()
    def _apply_pissa_init(self, org_module: torch.nn.Module) -> None:
        weight = org_module.weight.detach()
        is_linear = weight.ndim == 2
        is_conv2d_1x1 = weight.ndim == 4 and tuple(weight.shape[2:]) == (1, 1)
        if not is_linear and not (self.pissa_apply_conv2d and is_conv2d_1x1):
            return

        compute_device = self.lora_up.weight.device
        weight_matrix = self._reshape_weight_to_matrix(weight).to(device=compute_device, dtype=torch.float32)
        out_dim, in_dim = weight_matrix.shape
        rank = min(self.lora_dim, out_dim, in_dim)
        if rank <= 0:
            return

        if self.pissa_method == "rsvd":
            q = min(rank + self.pissa_oversample, min(out_dim, in_dim))
            if q <= 0:
                return
            U, S, V = torch.svd_lowrank(weight_matrix, q=q, niter=self.pissa_niter)
            U = U[:, :rank]
            S = S[:rank]
            Vh = V[:, :rank].transpose(0, 1)
        else:
            U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]

        scale = float(self.scale)
        if scale <= 0:
            return

        principal = (U * S.unsqueeze(0)) @ Vh
        residual = weight_matrix - principal

        scaled_s = torch.sqrt((S / scale).clamp_min(0))
        up_weight = U * scaled_s.unsqueeze(0)
        down_weight = scaled_s.unsqueeze(1) * Vh

        residual = self._reshape_matrix_to_weight(residual, weight)
        up_weight = self._reshape_matrix_to_weight(up_weight, self.lora_up.weight)
        down_weight = self._reshape_matrix_to_weight(down_weight, self.lora_down.weight)

        org_module.weight.copy_(residual.to(device=org_module.weight.device, dtype=org_module.weight.dtype))
        self.lora_up.weight.copy_(up_weight.to(device=self.lora_up.weight.device, dtype=self.lora_up.weight.dtype))
        self.lora_down.weight.copy_(down_weight.to(device=self.lora_down.weight.device, dtype=self.lora_down.weight.dtype))
        self._pissa_initialized = True

    @staticmethod
    def _adapter_weight_to_matrix(up_weight: torch.Tensor, down_weight: torch.Tensor, scale: float) -> torch.Tensor:
        if up_weight.ndim == 4 and down_weight.ndim == 4:
            up_weight = up_weight.squeeze(3).squeeze(2)
            down_weight = down_weight.squeeze(3).squeeze(2)
        return (up_weight @ down_weight) * scale

    def export_standard_lora_weights(self, export_mode: str):
        current_down = self.lora_down.weight.detach().clone().to(device="cpu", dtype=torch.float32)
        current_up = self.lora_up.weight.detach().clone().to(device="cpu", dtype=torch.float32)
        current_alpha = self.alpha.detach().clone().to(device="cpu", dtype=torch.float32)

        if not self._pissa_initialized:
            return current_down, current_up, current_alpha

        export_mode = _normalize_pissa_export_mode(export_mode)
        scale = float(self.scale)

        if export_mode == "approx":
            current_matrix = self._adapter_weight_to_matrix(current_up, current_down, scale)
            initial_matrix = self._adapter_weight_to_matrix(
                self._pissa_initial_lora_up_weight, self._pissa_initial_lora_down_weight, scale
            )
            delta_matrix = current_matrix - initial_matrix
            out_dim, in_dim = delta_matrix.shape
            rank = min(self.lora_dim, out_dim, in_dim)
            if rank <= 0:
                return current_down, current_up, current_alpha

            q = min(rank + 8, min(out_dim, in_dim))
            U, S, V = torch.svd_lowrank(delta_matrix, q=q, niter=2)
            U = U[:, :rank]
            S = S[:rank]
            Vh = V[:, :rank].transpose(0, 1)

            scaled_s = torch.sqrt(S.clamp_min(0))
            up_weight = U * scaled_s.unsqueeze(0)
            down_weight = scaled_s.unsqueeze(1) * Vh
            up_weight = self._reshape_matrix_to_weight(up_weight, current_up)
            down_weight = self._reshape_matrix_to_weight(down_weight, current_down)
            alpha = torch.tensor(float(rank), dtype=torch.float32)
            return down_weight, up_weight, alpha

        up_weight = torch.cat(
            [current_up * scale, -self._pissa_initial_lora_up_weight * scale],
            dim=1,
        )
        down_weight = torch.cat([current_down, self._pissa_initial_lora_down_weight], dim=0)
        alpha = torch.tensor(float(down_weight.shape[0]), dtype=torch.float32)
        return down_weight, up_weight, alpha

def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    adapter_type = str(
        kwargs.get("anima_adapter_type", kwargs.get("adapter_type", kwargs.get("lycoris_algo", "lora")))
    ).strip().lower()
    use_lokr = adapter_type == "lokr"
    use_lora_fa = adapter_type == "lora_fa"
    use_vera = adapter_type == "vera"
    use_dora = adapter_type == "lora" and _parse_bool_arg(kwargs.get("dora_wd", None), default=False)
    lokr_factor = int(kwargs.get("lokr_factor", kwargs.get("factor", 8)) or 8)
    lokr_full_matrix = _parse_bool_arg(kwargs.get("lokr_full_matrix", kwargs.get("full_matrix", None)), default=False)
    if use_lokr and int(network_dim) >= LOKR_FULL_MATRIX_DIM_SENTINEL and not lokr_full_matrix:
        logger.info(
            "Anima LoKr: network_dim=%s >= %s, forcing full_matrix=True (LyCORIS sentinel semantics).",
            network_dim,
            LOKR_FULL_MATRIX_DIM_SENTINEL,
        )
        lokr_full_matrix = True
    lokr_decompose_both = _parse_bool_arg(
        kwargs.get("lokr_decompose_both", kwargs.get("decompose_both", None)), default=False
    )
    lokr_unbalanced_factorization = _parse_bool_arg(kwargs.get("unbalanced_factorization", None), default=False)
    train_norm = _parse_bool_arg(kwargs.get("train_norm", None), default=False)
    bypass_mode = _parse_bool_arg(kwargs.get("bypass_mode", None), default=False)
    pissa_init = _parse_bool_arg(kwargs.get("pissa_init", None), default=False)
    pissa_method = _normalize_pissa_method(kwargs.get("pissa_method", "rsvd"))
    pissa_niter = int(kwargs.get("pissa_niter", 2) or 2)
    pissa_oversample = int(kwargs.get("pissa_oversample", 8) or 8)
    pissa_apply_conv2d = _parse_bool_arg(kwargs.get("pissa_apply_conv2d", None), default=False)
    pissa_export_mode = _normalize_pissa_export_mode(kwargs.get("pissa_export_mode", "lossless"))
    # Keep Anima LoKr checkpoints in native LoKr form by default. Explicitly use
    # lora_compatible only when a standard LoRA-style export is required.
    lokr_export_mode = str(kwargs.get("lokr_export_mode", "native")).strip().lower().replace("-", "_")
    if lokr_export_mode not in ("native", "lora_compatible"):
        logger.warning(f"Invalid lokr_export_mode '{lokr_export_mode}', defaulting to 'native'")
        lokr_export_mode = "native"
    vera_projection_prng_key = int(kwargs.get("vera_projection_prng_key", 0) or 0)
    vera_d_initial = float(kwargs.get("vera_d_initial", 0.1) or 0.1)

    # train LLM adapter
    train_llm_adapter = _parse_bool_arg(kwargs.get("train_llm_adapter", None), default=False)

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        if not isinstance(exclude_patterns, list):
            exclude_patterns = [exclude_patterns]

    # add default exclude patterns
    if train_norm:
        exclude_patterns.append(r".*(_modulation|_embedder|final_layer).*")
    else:
        exclude_patterns.append(r".*(_modulation|_norm|_embedder|final_layer).*")

    # regular expression for module selection: exclude and include
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None:
        include_patterns = ast.literal_eval(include_patterns)
        if not isinstance(include_patterns, list):
            include_patterns = [include_patterns]

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # verbose
    verbose = kwargs.get("verbose", "false")
    if verbose is not None:
        verbose = True if verbose.lower() == "true" else False

    # regex-specific learning rates / dimensions
    def parse_kv_pairs(kv_pair_str: str, is_int: bool) -> Dict[str, float]:
        """
        Parse a string of key-value pairs separated by commas.
        """
        pairs = {}
        for pair in kv_pair_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                logger.warning(f"Invalid format: {pair}, expected 'key=value'")
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                pairs[key] = int(value) if is_int else float(value)
            except ValueError:
                logger.warning(f"Invalid value for {key}: {value}")
        return pairs

    network_reg_lrs = kwargs.get("network_reg_lrs", None)
    if network_reg_lrs is not None:
        reg_lrs = parse_kv_pairs(network_reg_lrs, is_int=False)
    else:
        reg_lrs = None

    network_reg_dims = kwargs.get("network_reg_dims", None)
    if network_reg_dims is not None:
        reg_dims = parse_kv_pairs(network_reg_dims, is_int=True)
    else:
        reg_dims = None

    if (use_lora_fa or use_vera) and pissa_init:
        logger.warning("PiSSA is not supported with Anima LoRA-FA / VeRA. Disabling pissa_init automatically.")
        pissa_init = False

    if use_dora and bypass_mode:
        logger.warning("Anima DoRA does not support bypass_mode. Forcing bypass_mode=False.")
        bypass_mode = False

    if use_dora and pissa_init:
        logger.warning("PiSSA is not supported together with Anima DoRA. Disabling pissa_init automatically.")
        pissa_init = False

    module_class = LoKrModule if use_lokr else (VeraModule if use_vera else (LoRAFAModule if use_lora_fa else (DoRALoRAModule if use_dora else LoRAModule)))
    if not use_lokr and not use_lora_fa and not use_vera and not use_dora and pissa_init:
        module_class = partial(
            PiSSAModule,
            pissa_method=pissa_method,
            pissa_niter=pissa_niter,
            pissa_oversample=pissa_oversample,
            pissa_apply_conv2d=pissa_apply_conv2d,
        )

    network = LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        verbose=verbose,
        module_class=module_class,
        use_lokr=use_lokr,
        lokr_factor=lokr_factor,
        train_norm=train_norm,
        adapter_type=adapter_type,
        vera_projection_prng_key=vera_projection_prng_key,
        vera_d_initial=vera_d_initial,
        use_dora=use_dora,
        bypass_mode=bypass_mode,
        lokr_full_matrix=lokr_full_matrix,
        lokr_decompose_both=lokr_decompose_both,
        lokr_unbalanced_factorization=lokr_unbalanced_factorization,
    )
    network.pissa_export_mode = pissa_export_mode
    network.lokr_export_mode = lokr_export_mode

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = {}
    modules_alpha = {}
    modules_factor = {}
    modules_lokr_shape = {}
    train_llm_adapter = False
    is_lokr = False
    is_dora = False
    train_norm = False
    requested_adapter_type = str(
        kwargs.get("anima_adapter_type", kwargs.get("adapter_type", ""))
    ).strip().lower()
    if requested_adapter_type == "vera" and not for_inference:
        raise RuntimeError(
            "Anima VeRA exports are saved as inference-compatible standard LoRA weights. "
            "Continuing VeRA training from exported adapter weights is not supported; "
            "please resume from save_state / checkpoint instead."
        )
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if lora_name.startswith(TRAIN_NORM_PREFIX_ANIMA) or lora_name.startswith(TRAIN_NORM_PREFIX_TEXT_ENCODER):
            train_norm = True
            continue
        if _is_comfyui_train_norm_key(key):
            train_norm = True
            continue
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lokr_rank" in key:
            modules_dim[lora_name] = int(value.detach().float().cpu().item())
            is_lokr = True
        elif key.endswith(".lokr_w1"):
            modules_factor[lora_name] = value.size(0)
            modules_lokr_shape.setdefault(lora_name, {})["w1"] = tuple(value.shape)
            is_lokr = True
        elif key.endswith(".lokr_w2"):
            modules_lokr_shape.setdefault(lora_name, {})["w2"] = tuple(value.shape)
            is_lokr = True
        elif key.endswith(".lokr_w1_a"):
            modules_lokr_shape.setdefault(lora_name, {})["w1_a"] = tuple(value.shape)
            modules_dim[lora_name] = int(value.size(1))
            is_lokr = True
        elif key.endswith(".lokr_w1_b"):
            modules_lokr_shape.setdefault(lora_name, {})["w1_b"] = tuple(value.shape)
            if lora_name not in modules_dim:
                alpha_value = modules_alpha.get(lora_name)
                modules_dim[lora_name] = int(alpha_value.detach().float().cpu().item()) if alpha_value is not None else 4
            is_lokr = True
        elif key.endswith(".lokr_w2_a"):
            modules_lokr_shape.setdefault(lora_name, {})["w2_a"] = tuple(value.shape)
            modules_dim[lora_name] = int(value.size(1))
            is_lokr = True
        elif key.endswith(".lokr_w2_b"):
            modules_lokr_shape.setdefault(lora_name, {})["w2_b"] = tuple(value.shape)
            if lora_name not in modules_dim:
                alpha_value = modules_alpha.get(lora_name)
                modules_dim[lora_name] = int(alpha_value.detach().float().cpu().item()) if alpha_value is not None else 4
            is_lokr = True
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
        elif "dora_magnitude" in key or "dora_scale" in key:
            is_dora = True

        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    if is_lokr:
        for lora_name in modules_lokr_shape.keys():
            if lora_name not in modules_dim:
                alpha_value = modules_alpha.get(lora_name)
                modules_dim[lora_name] = int(alpha_value.detach().float().cpu().item()) if alpha_value is not None else 4
        for lora_name, dim in modules_dim.items():
            if lora_name not in modules_alpha:
                modules_alpha[lora_name] = dim

    if is_lokr:
        module_class = LoKrInfModule if for_inference else LoKrModule
    else:
        if is_dora:
            module_class = DoRALoRAModule
        elif for_inference:
            module_class = LoRAInfModule
        elif requested_adapter_type == "lora_fa":
            module_class = LoRAFAModule
        elif requested_adapter_type == "vera":
            module_class = LoRAInfModule
        else:
            module_class = LoRAModule

    network = LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_llm_adapter=train_llm_adapter,
        use_lokr=is_lokr,
        lokr_factor=next(iter(modules_factor.values())) if modules_factor else 8,
        modules_factor=modules_factor if is_lokr else None,
        modules_lokr_shape=modules_lokr_shape if is_lokr else None,
        train_norm=train_norm,
        adapter_type=("lokr" if is_lokr else ("lora" if is_dora else requested_adapter_type)),
        use_dora=is_dora,
    )
    return network, weights_sd


class LoRANetwork(torch.nn.Module):
    # Target modules: DiT blocks, embedders, final layer. embedders and final layer are excluded by default.
    ANIMA_TARGET_REPLACE_MODULE = ["Block", "PatchEmbed", "TimestepEmbedding", "FinalLayer"]
    # Target modules: LLM Adapter blocks
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["LLMAdapterTransformerBlock"]
    # Target modules for text encoder (Qwen3)
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["Qwen3Attention", "Qwen3MLP", "Qwen3SdpaAttention", "Qwen3FlashAttention2"]

    LORA_PREFIX_ANIMA = "lora_unet"  # ComfyUI compatible
    LORA_PREFIX_TEXT_ENCODER = "lora_te"  # Qwen3

    def __init__(
        self,
        text_encoders: list,
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        train_llm_adapter: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        reg_dims: Optional[Dict[str, int]] = None,
        reg_lrs: Optional[Dict[str, float]] = None,
        verbose: Optional[bool] = False,
        use_lokr: bool = False,
        lokr_factor: int = 8,
        modules_factor: Optional[Dict[str, int]] = None,
        modules_lokr_shape: Optional[Dict[str, Dict[str, Tuple[int, ...]]]] = None,
        train_norm: bool = False,
        adapter_type: Optional[str] = None,
        vera_projection_prng_key: int = 0,
        vera_d_initial: float = 0.1,
        use_dora: bool = False,
        bypass_mode: bool = False,
        lokr_full_matrix: bool = False,
        lokr_decompose_both: bool = False,
        lokr_unbalanced_factorization: bool = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.train_llm_adapter = train_llm_adapter
        self.reg_dims = reg_dims
        self.reg_lrs = reg_lrs
        self.use_lokr = use_lokr
        self.lokr_factor = lokr_factor
        self.modules_factor = modules_factor or {}
        self.modules_lokr_shape = modules_lokr_shape or {}
        self.lokr_full_matrix = bool(lokr_full_matrix)
        self.lokr_decompose_both = bool(lokr_decompose_both)
        self.lokr_unbalanced_factorization = bool(lokr_unbalanced_factorization)
        self.vera_projection_prng_key = int(vera_projection_prng_key)
        self.vera_d_initial = float(vera_d_initial)
        normalized_adapter_type = str(adapter_type or ("lokr" if self.use_lokr else "lora")).strip().lower()
        self.adapter_type = "lokr" if self.use_lokr else (normalized_adapter_type or "lora")
        self.use_dora = bool(use_dora) and self.adapter_type == "lora"
        self.bypass_mode = bool(bypass_mode) if not self.use_dora else False
        self.train_norm = bool(train_norm)

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None
        self.pissa_export_mode = "lossless"

        if self.adapter_type == "vera":
            self.register_buffer("vera_shared_A", torch.empty((self.lora_dim, 0), dtype=torch.float32), persistent=True)
            self.register_buffer("vera_shared_B", torch.empty((0, self.lora_dim), dtype=torch.float32), persistent=True)

        if modules_dim is not None:
            logger.info(f"create {self.adapter_type.upper()} network from weights")
        else:
            logger.info(f"create {self.adapter_type.upper()} network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            if self.use_lokr:
                logger.info(
                    "Anima LoKr config: factor=%s, full_matrix=%s, decompose_both=%s, unbalanced_factorization=%s",
                    self.lokr_factor,
                    self.lokr_full_matrix,
                    self.lokr_decompose_both,
                    self.lokr_unbalanced_factorization,
                )

        # compile regular expression if specified
        def str_to_re_patterns(patterns: Optional[List[str]]) -> List[re.Pattern]:
            re_patterns = []
            if patterns is not None:
                for pattern in patterns:
                    try:
                        re_pattern = re.compile(pattern)
                    except re.error as e:
                        logger.error(f"Invalid pattern '{pattern}': {e}")
                        continue
                    re_patterns.append(re_pattern)
            return re_patterns

        exclude_re_patterns = str_to_re_patterns(exclude_patterns)
        include_re_patterns = str_to_re_patterns(include_patterns)

        def is_allowed_module(original_name: str) -> bool:
            excluded = any(pattern.fullmatch(original_name) for pattern in exclude_re_patterns)
            included = any(pattern.fullmatch(original_name) for pattern in include_re_patterns)
            if excluded and not included:
                if verbose:
                    logger.info(f"exclude: {original_name}")
                return False
            return True

        def iter_target_modules(root_module: torch.nn.Module, target_replace_modules: List[str]):
            if target_replace_modules is None:
                yield "", root_module
                return

            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    yield name, module

        def build_train_norm_prefix(is_unet: bool, text_encoder_idx: Optional[int]) -> str:
            if is_unet:
                return TRAIN_NORM_PREFIX_ANIMA
            if text_encoder_idx in (None, 0):
                return TRAIN_NORM_PREFIX_TEXT_ENCODER
            return f"{TRAIN_NORM_PREFIX_TEXT_ENCODER}{text_encoder_idx + 1}"

        def is_trainable_norm_module(module: torch.nn.Module) -> bool:
            if "norm" not in module.__class__.__name__.lower():
                return False
            return any(True for _name, _param in module.named_parameters(recurse=False))

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            default_dim: Optional[int] = None,
        ) -> Tuple[List[LoRAModule], List[TrainNormRef], List[str]]:
            prefix = self.LORA_PREFIX_ANIMA if is_unet else self.LORA_PREFIX_TEXT_ENCODER
            norm_prefix = build_train_norm_prefix(is_unet, text_encoder_idx)

            loras = []
            norm_refs = []
            skipped = []
            seen_lora_names = set()
            seen_norm_names = set()
            for scope_name, scope_module in iter_target_modules(root_module, target_replace_modules):
                for child_name, child_module in scope_module.named_modules():
                    original_name = ".".join(part for part in (scope_name, child_name) if part)
                    if not original_name:
                        continue

                    if self.train_norm and original_name not in seen_norm_names and is_trainable_norm_module(child_module):
                        if is_allowed_module(original_name):
                            norm_name = f"{norm_prefix}_{original_name}".replace(".", "_")
                            norm_ref = TrainNormRef(
                                norm_name,
                                original_name,
                                child_module,
                                _snapshot_train_norm_base_params(child_module),
                            )
                            norm_refs.append(norm_ref)
                            seen_norm_names.add(original_name)

                    is_linear = child_module.__class__.__name__ == "Linear"
                    is_conv2d = child_module.__class__.__name__ == "Conv2d"
                    is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                    should_inject = is_linear or (is_conv2d and not self.use_lokr)
                    if not should_inject or original_name in seen_lora_names:
                        continue
                    if not is_allowed_module(original_name):
                        continue

                    lora_name = f"{prefix}.{original_name}".replace(".", "_")
                    if self.adapter_type == "vera" and not is_linear:
                        if is_conv2d_1x1 or not self.use_lokr:
                            skipped.append(lora_name)
                        continue
                    dim = None
                    alpha_val = None
                    factor_val = self.modules_factor.get(lora_name, self.lokr_factor)

                    if modules_dim is not None:
                        if lora_name in modules_dim:
                            dim = modules_dim[lora_name]
                            alpha_val = modules_alpha.get(lora_name, dim)
                    else:
                        if self.reg_dims is not None:
                            for reg, d in self.reg_dims.items():
                                if re.fullmatch(reg, original_name):
                                    dim = d
                                    alpha_val = self.alpha
                                    logger.info(f"Module {original_name} matched with regex '{reg}' -> dim: {dim}")
                                    break
                        if dim is None and (is_linear or is_conv2d_1x1):
                            dim = default_dim if default_dim is not None else self.lora_dim
                            alpha_val = self.alpha

                    if dim is None or dim == 0:
                        if is_linear or (is_conv2d_1x1 and not self.use_lokr):
                            skipped.append(lora_name)
                        continue

                    module_kwargs = dict(
                        dropout=dropout,
                        rank_dropout=rank_dropout,
                        module_dropout=module_dropout,
                    )
                    if self.use_lokr:
                        module_kwargs["factor"] = factor_val
                        module_kwargs["full_matrix"] = self.lokr_full_matrix
                        module_kwargs["decompose_both"] = self.lokr_decompose_both
                        module_kwargs["unbalanced_factorization"] = self.lokr_unbalanced_factorization
                        if lora_name in self.modules_lokr_shape:
                            module_kwargs["lokr_shape"] = self.modules_lokr_shape[lora_name]
                    if self.adapter_type == "vera":
                        module_kwargs["network_ref"] = self
                        module_kwargs["d_initial"] = self.vera_d_initial

                    lora = module_class(
                        lora_name,
                        child_module,
                        self.multiplier,
                        dim,
                        alpha_val,
                        **module_kwargs,
                    )
                    lora.original_name = original_name
                    loras.append(lora)
                    seen_lora_names.add(original_name)
            return loras, norm_refs, skipped

        # Create LoRA for text encoders (Qwen3 - typically not trained for Anima)
        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []
        self.text_encoder_norms: List[TrainNormRef] = []
        skipped_te = []
        if text_encoders is not None:
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue
                logger.info(f"create {self.adapter_type.upper()} for Text Encoder {i+1}:")
                te_loras, te_norms, te_skipped = create_modules(
                    False, i, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
                )
                logger.info(f"create {self.adapter_type.upper()} for Text Encoder {i+1}: {len(te_loras)} modules.")
                self.text_encoder_loras.extend(te_loras)
                self.text_encoder_norms.extend(te_norms)
                if self.train_norm:
                    logger.info(f"create train_norm for Text Encoder {i+1}: {len(te_norms)} modules.")
                skipped_te += te_skipped

        # Create LoRA for DiT blocks
        target_modules = list(LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
        if train_llm_adapter:
            target_modules.extend(LoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE)

        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_norms: List[TrainNormRef]
        self.unet_loras, self.unet_norms, skipped_un = create_modules(True, None, unet, target_modules)

        logger.info(f"create {self.adapter_type.upper()} for Anima DiT: {len(self.unet_loras)} modules.")
        if self.train_norm:
            logger.info(f"create train_norm for Anima DiT: {len(self.unet_norms)} modules.")
        if verbose:
            for lora in self.unet_loras:
                alpha_value = getattr(lora, "alpha", getattr(lora, "lora_dim", "n/a"))
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {alpha_value}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} {self.adapter_type.upper()} modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        # assertion: no duplicate names
        names = set()
        for module_ref in self.text_encoder_loras + self.unet_loras + self.text_encoder_norms + self.unet_norms:
            assert module_ref.lora_name not in names, f"duplicated lora name: {module_ref.lora_name}"
            names.add(module_ref.lora_name)

    def ensure_vera_shared_buffers(self, out_features: int, in_features: int) -> None:
        if self.adapter_type != "vera":
            return

        current_a = self.vera_shared_A
        current_b = self.vera_shared_B
        target_in = max(int(in_features), int(current_a.shape[1]))
        target_out = max(int(out_features), int(current_b.shape[0]))
        if target_in == current_a.shape[1] and target_out == current_b.shape[0]:
            return

        new_a = _vera_kaiming_uniform((self.lora_dim, target_in), self.vera_projection_prng_key)
        new_b = _vera_kaiming_uniform((target_out, self.lora_dim), self.vera_projection_prng_key + 1)
        if current_a.numel() > 0:
            new_a[:, : current_a.shape[1]] = current_a.detach().to(device="cpu", dtype=torch.float32)
        if current_b.numel() > 0:
            new_b[: current_b.shape[0], :] = current_b.detach().to(device="cpu", dtype=torch.float32)

        self._buffers["vera_shared_A"] = new_a.to(device=current_a.device, dtype=current_a.dtype)
        self._buffers["vera_shared_B"] = new_b.to(device=current_b.device, dtype=current_b.dtype)

    def get_vera_shared_A(self) -> torch.Tensor:
        if self.adapter_type != "vera":
            raise RuntimeError("VeRA shared projections are only available when adapter_type=vera.")
        return self.vera_shared_A

    def get_vera_shared_B(self) -> torch.Tensor:
        if self.adapter_type != "vera":
            raise RuntimeError("VeRA shared projections are only available when adapter_type=vera.")
        return self.vera_shared_B

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        if self.adapter_type == "vera":
            has_native_vera_state = any(("vera_lambda_" in key) or key.startswith("vera_shared_") for key in weights_sd.keys())
            if not has_native_vera_state:
                raise RuntimeError(
                    "This Anima VeRA route exports inference-compatible standard LoRA weights. "
                    "Continuing VeRA training from exported adapter weights is not supported. "
                    "Please resume from save_state / checkpoint instead."
                )

        info = self.load_state_dict(weights_sd, False)
        return info

    def _iter_train_norm_refs(self):
        return self.text_encoder_norms + self.unet_norms

    def _iter_train_norm_param_items(self):
        for norm_ref in self._iter_train_norm_refs():
            for name, param in norm_ref.named_parameters():
                yield norm_ref, name, param

    def _apply_train_norm_state_dict(self, state_dict):
        train_norm_refs = self._iter_train_norm_refs()
        norm_ref_map = {norm_ref.lora_name: norm_ref for norm_ref in train_norm_refs}
        comfy_norm_ref_map = {
            comfy_name: norm_ref
            for norm_ref in train_norm_refs
            for comfy_name in [_train_norm_name_to_comfyui_name(norm_ref.lora_name)]
            if comfy_name is not None
        }
        missing_norm_keys = []
        unexpected_norm_keys = []
        expected_norm_keys = {
            f"{norm_ref.lora_name}.{name}"
            for norm_ref, name, _param in self._iter_train_norm_param_items()
        }
        saw_train_norm_key = False

        for key, value in state_dict.items():
            if "." not in key:
                continue

            if _is_comfyui_train_norm_key(key):
                saw_train_norm_key = True
                lora_name, param_name = key.rsplit(".", 1)
                norm_ref = comfy_norm_ref_map.get(lora_name)
                if norm_ref is None:
                    unexpected_norm_keys.append(key)
                    continue

                target_name = "weight" if param_name == "diff" else "bias"
                params_by_name = dict(norm_ref.named_parameters())
                target_param = params_by_name.get(target_name)
                if target_param is None:
                    unexpected_norm_keys.append(key)
                    continue

                base_param = norm_ref.base_params.get(target_name)
                if base_param is None:
                    raise RuntimeError(f"missing base train_norm snapshot for {norm_ref.lora_name}.{target_name}")
                if tuple(target_param.shape) != tuple(value.shape):
                    raise RuntimeError(
                        f"size mismatch for {key}: copying a diff with shape {tuple(value.shape)} "
                        f"to a param with shape {tuple(target_param.shape)}."
                    )
                if tuple(base_param.shape) != tuple(target_param.shape):
                    raise RuntimeError(
                        f"size mismatch for base snapshot {norm_ref.lora_name}.{target_name}: "
                        f"base shape {tuple(base_param.shape)}, current shape {tuple(target_param.shape)}."
                    )

                restored = base_param.to(dtype=torch.float32) + value.detach().cpu().to(dtype=torch.float32)
                target_param.data.copy_(restored.to(device=target_param.device, dtype=target_param.dtype))
                expected_norm_keys.discard(f"{norm_ref.lora_name}.{target_name}")
                continue

            lora_name, param_name = key.split(".", 1)
            if not (
                lora_name.startswith(TRAIN_NORM_PREFIX_ANIMA) or lora_name.startswith(TRAIN_NORM_PREFIX_TEXT_ENCODER)
            ):
                continue
            saw_train_norm_key = True
            if lora_name not in norm_ref_map:
                unexpected_norm_keys.append(key)
                continue

            norm_ref = norm_ref_map[lora_name]
            params_by_name = dict(norm_ref.named_parameters())
            target_param = params_by_name.get(param_name)
            if target_param is None:
                unexpected_norm_keys.append(key)
                continue
            if tuple(target_param.shape) != tuple(value.shape):
                raise RuntimeError(
                    f"size mismatch for {key}: copying a param with shape {tuple(value.shape)} "
                    f"from checkpoint, the shape in current model is {tuple(target_param.shape)}."
                )
            target_param.data.copy_(value.to(device=target_param.device, dtype=target_param.dtype))
            expected_norm_keys.discard(key)

        if saw_train_norm_key:
            missing_norm_keys.extend(sorted(expected_norm_keys))
        return missing_norm_keys, unexpected_norm_keys

    def load_state_dict(self, state_dict, strict=True):
        lora_state_dict = {}
        unexpected_norm_keys = []
        for key, value in state_dict.items():
            if "." not in key:
                lora_state_dict[key] = value
                continue
            lora_name = key.split(".", 1)[0]
            if lora_name.startswith(TRAIN_NORM_PREFIX_ANIMA) or lora_name.startswith(TRAIN_NORM_PREFIX_TEXT_ENCODER):
                continue
            if _is_comfyui_train_norm_key(key):
                continue
            lora_state_dict[key] = value

        info = super().load_state_dict(lora_state_dict, strict=False)
        missing_norm_keys, norm_unexpected = self._apply_train_norm_state_dict(state_dict)
        unexpected_norm_keys.extend(norm_unexpected)

        missing_keys = list(info.missing_keys)
        unexpected_keys = list(info.unexpected_keys)

        if strict:
            missing_keys.extend(missing_norm_keys)
            unexpected_keys.extend(unexpected_norm_keys)
            if missing_keys or unexpected_keys:
                error_msgs = []
                if unexpected_keys:
                    error_msgs.append(
                        "Unexpected key(s) in state_dict: {}.".format(", ".join(f'"{k}"' for k in unexpected_keys))
                    )
                if missing_keys:
                    error_msgs.append(
                        "Missing key(s) in state_dict: {}.".format(", ".join(f'"{k}"' for k in missing_keys))
                    )
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t" + "\n\t".join(error_msgs)
                )
            return _IncompatibleKeys([], [])

        return _IncompatibleKeys(missing_keys + missing_norm_keys, unexpected_keys + unexpected_norm_keys)

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable {self.adapter_type.upper()} for text encoder: {len(self.text_encoder_loras)} modules")
            if self.train_norm and self.text_encoder_norms:
                logger.info(f"enable train_norm for text encoder: {len(self.text_encoder_norms)} modules")
        else:
            self.text_encoder_loras = []
            self.text_encoder_norms = []

        if apply_unet:
            logger.info(f"enable {self.adapter_type.upper()} for DiT: {len(self.unet_loras)} modules")
            if self.train_norm and self.unet_norms:
                logger.info(f"enable train_norm for DiT: {len(self.unet_norms)} modules")
        else:
            self.unet_loras = []
            self.unet_norms = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(TRAIN_NORM_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_ANIMA):
                apply_unet = True
            elif key.startswith(TRAIN_NORM_PREFIX_ANIMA):
                apply_unet = True

        if apply_text_encoder:
            logger.info(f"enable {self.adapter_type.upper()} for text encoder")
        else:
            self.text_encoder_loras = []
            self.text_encoder_norms = []

        if apply_unet:
            logger.info(f"enable {self.adapter_type.upper()} for DiT")
        else:
            self.unet_loras = []
            self.unet_norms = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        self._apply_train_norm_state_dict(weights_sd)

        logger.info("weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def _set_adapter_requires_grad(self, requires_grad: bool) -> None:
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.requires_grad_(requires_grad)
        for _norm_ref, _name, param in self._iter_train_norm_param_items():
            param.requires_grad_(requires_grad)

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        if text_encoder_lr is None or (isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0):
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            pass  # already a list with one element

        self._set_adapter_requires_grad(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, norms, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}, "norm": {}}
            reg_groups = {}
            reg_lrs_list = list(self.reg_lrs.items()) if self.reg_lrs is not None else []

            for lora in loras:
                matched_reg_lr = None
                for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                    if re.fullmatch(regex_str, lora.original_name):
                        matched_reg_lr = (i, reg_lr)
                        logger.info(f"Module {lora.original_name} matched regex '{regex_str}' -> LR {reg_lr}")
                        break

                for name, param in lora.named_parameters():
                    if not param.requires_grad:
                        continue
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {"lora": {}, "plus": {}, "norm": {}, "lr": reg_lr}
                        if loraplus_ratio is not None and "lora_up" in name:
                            reg_groups[group_key]["plus"][f"{lora.lora_name}.{name}"] = param
                        else:
                            reg_groups[group_key]["lora"][f"{lora.lora_name}.{name}"] = param
                        continue

                    if loraplus_ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            for norm_ref in norms:
                matched_reg_lr = None
                for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                    if re.fullmatch(regex_str, norm_ref.original_name):
                        matched_reg_lr = (i, reg_lr)
                        logger.info(f"Norm module {norm_ref.original_name} matched regex '{regex_str}' -> LR {reg_lr}")
                        break

                for name, param in norm_ref.named_parameters():
                    if not param.requires_grad:
                        continue
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {"lora": {}, "plus": {}, "norm": {}, "lr": reg_lr}
                        reg_groups[group_key]["norm"][f"{norm_ref.lora_name}.{name}"] = param
                        continue

                    param_groups["norm"][f"{norm_ref.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for group_key, group in reg_groups.items():
                reg_lr = group["lr"]
                for key in ("lora", "plus", "norm"):
                    param_data = {"params": group[key].values()}
                    if len(param_data["params"]) == 0:
                        continue
                    if key == "plus":
                        param_data["lr"] = reg_lr * loraplus_ratio if loraplus_ratio is not None else reg_lr
                    else:
                        param_data["lr"] = reg_lr
                    if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                        logger.info("NO LR skipping!")
                        continue
                    params.append(param_data)
                    desc = f"reg_lr_{group_key.split('_')[-1]}"
                    if key == "plus":
                        desc += " plus"
                    elif key == "norm":
                        desc += " norm"
                    descriptions.append(desc)

            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr
                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "norm" if key == "norm" else "")
            return params, descriptions

        if self.text_encoder_loras or self.text_encoder_norms:
            loraplus_ratio = self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
            te1_loras = [lora for lora in self.text_encoder_loras if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER)]
            te1_norms = [norm_ref for norm_ref in self.text_encoder_norms if norm_ref.lora_name.startswith(TRAIN_NORM_PREFIX_TEXT_ENCODER)]
            if len(te1_loras) > 0 or len(te1_norms) > 0:
                logger.info(
                    f"Text Encoder 1 (Qwen3): {len(te1_loras)} adapter modules, {len(te1_norms)} norm modules, LR {text_encoder_lr[0]}"
                )
                params, descriptions = assemble_params(te1_loras, te1_norms, text_encoder_lr[0], loraplus_ratio)
                all_params.extend(params)
                lr_descriptions.extend(["textencoder 1" + (" " + d if d else "") for d in descriptions])

        if self.unet_loras or self.unet_norms:
            logger.info(
                f"Anima DiT: {len(self.unet_loras)} adapter modules, {len(self.unet_norms)} norm modules, "
                f"LR {unet_lr if unet_lr is not None else default_lr}"
            )
            params, descriptions = assemble_params(
                self.unet_loras,
                self.unet_norms,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        pass  # not supported

    def prepare_grad_etc(self, text_encoder, unet):
        self._set_adapter_requires_grad(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        params = []
        seen = set()
        for param in self.parameters():
            if not param.requires_grad:
                continue
            if id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
        for _norm_ref, _name, param in self._iter_train_norm_param_items():
            if not param.requires_grad:
                continue
            if id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
        return params

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for norm_ref, name, param in self._iter_train_norm_param_items():
            key = f"{prefix}{norm_ref.lora_name}.{name}"
            state_dict[key] = param if keep_vars else param.detach()
        return state_dict

    def get_extra_ema_modules(self):
        modules = []
        if self.text_encoder_norms:
            modules.append(("network_text_encoder_norms", TrainNormParamProxy(self.text_encoder_norms, "text_encoder_norms")))
        if self.unet_norms:
            modules.append(("network_unet_norms", TrainNormParamProxy(self.unet_norms, "unet_norms")))
        return modules

    def _prepare_adapter_export_metadata(self, metadata):
        if self.adapter_type == "vera":
            metadata = {} if metadata is None else dict(metadata)
            original_network_module = str(metadata.get("ss_network_module", "") or "").strip() or "networks.lora_anima"
            original_adapter_type = str(metadata.get("ss_anima_adapter_type", "") or "").strip().lower() or "vera"
            metadata.setdefault("ss_training_network_module", original_network_module)
            metadata.setdefault("ss_training_anima_adapter_type", original_adapter_type)
            metadata["ss_network_module"] = "networks.lora_anima"
            metadata["ss_anima_adapter_type"] = "lora"
            metadata["ss_adapter_variant"] = "vera"
            metadata["ss_vera_compatible_export"] = "true"
            return metadata

        if self.adapter_type == "lokr":
            metadata = {} if metadata is None else dict(metadata)
            lokr_export_mode = str(getattr(self, "lokr_export_mode", "native")).strip().lower().replace("-", "_")
            if lokr_export_mode not in ("native", "lora_compatible"):
                lokr_export_mode = "native"

            metadata.setdefault("ss_training_network_module", "networks.lora_anima")
            metadata.setdefault("ss_training_anima_adapter_type", "lokr")
            metadata["ss_training_lokr_factor"] = str(getattr(self, "lokr_factor", ""))
            metadata["ss_training_lokr_full_matrix"] = str(bool(getattr(self, "lokr_full_matrix", False))).lower()
            metadata["ss_training_lokr_decompose_both"] = str(bool(getattr(self, "lokr_decompose_both", False))).lower()
            metadata["ss_training_lokr_unbalanced_factorization"] = str(
                bool(getattr(self, "lokr_unbalanced_factorization", False))
            ).lower()
            metadata["ss_network_module"] = "networks.lora_anima"
            metadata["ss_adapter_variant"] = "lokr"
            metadata["ss_lokr_export_mode"] = lokr_export_mode
            if lokr_export_mode == "lora_compatible":
                metadata["ss_anima_adapter_type"] = "lora"
                metadata["ss_lokr_compatible_export"] = "true"
                metadata.pop("ss_lokr_native_export", None)
            else:
                metadata["ss_anima_adapter_type"] = "lokr"
                metadata["ss_lokr_native_export"] = "true"
                metadata.pop("ss_lokr_compatible_export", None)
            return metadata

        if self.adapter_type != "lora_fa":
            return metadata

        metadata = {} if metadata is None else dict(metadata)
        metadata["ss_anima_adapter_type"] = "lora_fa"
        metadata["ss_adapter_variant"] = "lora_fa"
        metadata["ss_lora_fa_compatible_export"] = "true"
        return metadata

    def _prepare_vera_export_for_save(self, state_dict, metadata):
        if self.adapter_type != "vera":
            return state_dict, metadata

        vera_loras = [lora for lora in (self.text_encoder_loras + self.unet_loras) if isinstance(lora, VeraModule)]
        if not vera_loras:
            return state_dict, metadata

        state_dict = dict(state_dict)
        for lora in vera_loras:
            down_weight, up_weight, alpha = lora.export_standard_lora_weights()
            state_dict[f"{lora.lora_name}.lora_down.weight"] = down_weight
            state_dict[f"{lora.lora_name}.lora_up.weight"] = up_weight
            state_dict[f"{lora.lora_name}.alpha"] = alpha
            state_dict.pop(f"{lora.lora_name}.vera_lambda_b", None)
            state_dict.pop(f"{lora.lora_name}.vera_lambda_d", None)

        state_dict.pop("vera_shared_A", None)
        state_dict.pop("vera_shared_B", None)

        metadata = self._prepare_adapter_export_metadata(metadata)
        raw_network_args = metadata.get("ss_network_args") if metadata is not None else None
        if raw_network_args:
            try:
                parsed = json.loads(raw_network_args)
                if isinstance(parsed, dict):
                    parsed.pop("anima_adapter_type", None)
                    parsed.pop("adapter_type", None)
                    parsed.pop("vera_projection_prng_key", None)
                    parsed.pop("vera_d_initial", None)
                elif isinstance(parsed, list):
                    parsed = [
                        item
                        for item in parsed
                        if not str(item).startswith(
                            ("anima_adapter_type=", "adapter_type=", "vera_projection_prng_key=", "vera_d_initial=")
                        )
                    ]
                metadata["ss_network_args"] = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass

        return state_dict, metadata

    def _prepare_lokr_export_for_save(self, state_dict, metadata):
        """Prepare LoKr weights for native save or explicit LoRA-compatible export."""
        if self.adapter_type != "lokr":
            return state_dict, metadata

        lokr_loras = [lora for lora in (self.text_encoder_loras + self.unet_loras) if isinstance(lora, LoKrModule)]
        if not lokr_loras:
            return state_dict, metadata

        lokr_export_mode = str(getattr(self, "lokr_export_mode", "native")).strip().lower().replace("-", "_")
        if lokr_export_mode not in ("native", "lora_compatible"):
            lokr_export_mode = "native"

        if lokr_export_mode == "native":
            state_dict = dict(state_dict)
            for lora in lokr_loras:
                self._prepare_native_lokr_module_for_comfyui(state_dict, lora)

            metadata = {} if metadata is None else dict(metadata)
            metadata["ss_network_module"] = "networks.lora_anima"
            metadata["ss_anima_adapter_type"] = "lokr"
            metadata["ss_adapter_variant"] = "lokr"
            metadata["ss_lokr_native_export"] = "true"
            metadata["ss_lokr_export_mode"] = "native"
            metadata["ss_lokr_rank_exported"] = "false"
            metadata["ss_lokr_scale_export_format"] = "comfyui_baked_single_scale"
            metadata.pop("ss_lokr_compatible_export", None)
            return state_dict, metadata

        state_dict = dict(state_dict)
        for lora in lokr_loras:
            down_weight, up_weight, alpha = lora.export_standard_lora_weights()
            state_dict[f"{lora.lora_name}.lora_down.weight"] = down_weight.contiguous()
            state_dict[f"{lora.lora_name}.lora_up.weight"] = up_weight.contiguous()
            state_dict[f"{lora.lora_name}.alpha"] = alpha
            state_dict.pop(f"{lora.lora_name}.lokr_w1", None)
            state_dict.pop(f"{lora.lora_name}.lokr_w2", None)
            state_dict.pop(f"{lora.lora_name}.lokr_w1_a", None)
            state_dict.pop(f"{lora.lora_name}.lokr_w1_b", None)
            state_dict.pop(f"{lora.lora_name}.lokr_w2_a", None)
            state_dict.pop(f"{lora.lora_name}.lokr_w2_b", None)
            state_dict.pop(f"{lora.lora_name}.lokr_rank", None)

        if lokr_export_mode == "lora_compatible":
            metadata = {} if metadata is None else dict(metadata)
            metadata.setdefault("ss_training_network_module", "networks.lora_anima")
            metadata.setdefault("ss_training_anima_adapter_type", "lokr")
            metadata["ss_network_module"] = "networks.lora_anima"
            metadata["ss_anima_adapter_type"] = "lora"
            metadata["ss_adapter_variant"] = "lokr"
            metadata["ss_lokr_compatible_export"] = "true"
            metadata["ss_lokr_export_mode"] = "lora_compatible"
            metadata.pop("ss_lokr_native_export", None)

            raw_network_args = metadata.get("ss_network_args")
            if raw_network_args:
                try:
                    parsed = json.loads(raw_network_args)
                    if isinstance(parsed, dict):
                        parsed.pop("anima_adapter_type", None)
                        parsed.pop("adapter_type", None)
                        parsed.pop("lokr_factor", None)
                        parsed.pop("lokr_export_mode", None)
                        parsed.pop("full_matrix", None)
                        parsed.pop("lokr_full_matrix", None)
                        parsed.pop("decompose_both", None)
                        parsed.pop("lokr_decompose_both", None)
                        parsed.pop("unbalanced_factorization", None)
                    elif isinstance(parsed, list):
                        parsed = [
                            item
                            for item in parsed
                            if not str(item).startswith(
                                (
                                    "anima_adapter_type=",
                                    "adapter_type=",
                                    "lokr_factor=",
                                    "lokr_export_mode=",
                                    "full_matrix=",
                                    "lokr_full_matrix=",
                                    "decompose_both=",
                                    "lokr_decompose_both=",
                                    "unbalanced_factorization=",
                                )
                            )
                        ]
                    metadata["ss_network_args"] = json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    pass

        return state_dict, metadata

    def _native_lokr_neutral_alpha(self, state_dict, lora):
        lora_name = lora.lora_name
        ranks = []
        for key in (f"{lora_name}.lokr_w1_b", f"{lora_name}.lokr_w2_b"):
            tensor = state_dict.get(key)
            if tensor is not None and tensor.ndim >= 1:
                ranks.append(int(tensor.shape[0]))

        if ranks and all(rank == ranks[0] for rank in ranks):
            return float(ranks[0])

        lora_dim = getattr(lora, "lora_dim", None)
        if lora_dim is not None:
            return float(lora_dim)
        if ranks:
            return float(ranks[0])
        return 1.0

    def _prepare_native_lokr_module_for_comfyui(self, state_dict, lora):
        lora_name = lora.lora_name
        effective_scale = float(getattr(lora, "scale", 1.0))

        w2_key = f"{lora_name}.lokr_w2"
        w2_b_key = f"{lora_name}.lokr_w2_b"
        if w2_key in state_dict:
            state_dict[w2_key] = (state_dict[w2_key].detach().clone() * effective_scale).contiguous()
        elif w2_b_key in state_dict:
            state_dict[w2_b_key] = (state_dict[w2_b_key].detach().clone() * effective_scale).contiguous()

        state_dict[f"{lora_name}.alpha"] = torch.tensor(
            self._native_lokr_neutral_alpha(state_dict, lora),
            dtype=torch.float32,
        )
        state_dict.pop(f"{lora_name}.lokr_rank", None)

    def _prepare_train_norm_comfyui_export_for_save(self, state_dict, metadata):
        if not self.train_norm:
            return state_dict, metadata

        state_dict = dict(state_dict)
        converted = 0
        suffix_by_param = {"weight": "diff", "bias": "diff_b"}
        for norm_ref, name, param in self._iter_train_norm_param_items():
            comfy_lora_name = _train_norm_name_to_comfyui_name(norm_ref.lora_name)
            if comfy_lora_name is None:
                continue

            suffix = suffix_by_param.get(name)
            if suffix is None:
                continue

            base_param = norm_ref.base_params.get(name)
            if base_param is None:
                raise RuntimeError(f"missing base train_norm snapshot for {norm_ref.lora_name}.{name}")
            if tuple(base_param.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"size mismatch for base snapshot {norm_ref.lora_name}.{name}: "
                    f"base shape {tuple(base_param.shape)}, current shape {tuple(param.shape)}."
                )

            diff = param.detach().cpu().to(dtype=torch.float32) - base_param.to(dtype=torch.float32)
            state_dict[f"{comfy_lora_name}.{suffix}"] = diff.contiguous()
            state_dict.pop(f"{norm_ref.lora_name}.{name}", None)
            converted += 1

        if converted > 0:
            metadata = {} if metadata is None else dict(metadata)
            metadata["ss_train_norm_export_format"] = "comfyui_diff"
            metadata["ss_train_norm_exported_count"] = str(converted)
        return state_dict, metadata

    def _prepare_dora_export_metadata(self, metadata):
        if not self.use_dora:
            return metadata

        metadata = {} if metadata is None else dict(metadata)
        metadata["ss_anima_adapter_type"] = "lora"
        metadata["ss_adapter_variant"] = "dora"
        metadata["ss_dora_compatible_export"] = "true"

        raw_network_args = metadata.get("ss_network_args")
        if raw_network_args:
            try:
                parsed = json.loads(raw_network_args)
                if isinstance(parsed, dict):
                    parsed["dora_wd"] = True
                    parsed["bypass_mode"] = False
                elif isinstance(parsed, list):
                    parsed = [
                        item
                        for item in parsed
                        if not str(item).startswith(("dora_wd=", "bypass_mode="))
                    ]
                    parsed.append("dora_wd=True")
                    parsed.append("bypass_mode=False")
                metadata["ss_network_args"] = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass
        return metadata

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        state_dict, metadata = self._prepare_vera_export_for_save(state_dict, metadata)
        state_dict, metadata = self._prepare_lokr_export_for_save(state_dict, metadata)
        state_dict, metadata = self._prepare_train_norm_comfyui_export_for_save(state_dict, metadata)
        metadata = self._prepare_adapter_export_metadata(metadata)
        metadata = self._prepare_dora_export_metadata(metadata)
        state_dict, metadata = self._prepare_pissa_export_for_save(state_dict, metadata)

        save_as_safetensors = os.path.splitext(file)[1] == ".safetensors"
        if dtype is not None or save_as_safetensors:
            from library import train_util

            state_dict = train_util.prepare_safetensors_state_dict(state_dict, dtype=dtype)

        if save_as_safetensors:
            from safetensors.torch import save_file

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def _prepare_pissa_export_for_save(self, state_dict, metadata):
        pissa_loras = [lora for lora in (self.text_encoder_loras + self.unet_loras) if isinstance(lora, PiSSAModule)]
        if not pissa_loras:
            return state_dict, metadata

        export_mode = _normalize_pissa_export_mode(getattr(self, "pissa_export_mode", "lossless"))
        for lora in pissa_loras:
            down_weight, up_weight, alpha = lora.export_standard_lora_weights(export_mode)
            state_dict[f"{lora.lora_name}.lora_down.weight"] = down_weight
            state_dict[f"{lora.lora_name}.lora_up.weight"] = up_weight
            state_dict[f"{lora.lora_name}.alpha"] = alpha

        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)

        metadata["ss_pissa_compatible_export"] = "true"
        metadata["ss_pissa_export_mode"] = export_mode

        raw_network_args = metadata.get("ss_network_args")
        if raw_network_args:
            try:
                parsed = json.loads(raw_network_args)
                if isinstance(parsed, dict):
                    for key in (
                        "pissa_init",
                        "pissa_method",
                        "pissa_niter",
                        "pissa_oversample",
                        "pissa_apply_conv2d",
                        "pissa_export_mode",
                    ):
                        parsed.pop(key, None)
                elif isinstance(parsed, list):
                    parsed = [
                        item for item in parsed
                        if not str(item).startswith(
                            (
                                "pissa_init=",
                                "pissa_method=",
                                "pissa_niter=",
                                "pissa_oversample=",
                                "pissa_apply_conv2d=",
                                "pissa_export_mode=",
                            )
                        )
                    ]
                metadata["ss_network_args"] = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass

        return state_dict, metadata

    def backup_weights(self):
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        if self.adapter_type == "vera":
            norms = []
            keys_scaled = 0
            for lora in self.text_encoder_loras + self.unet_loras:
                if not isinstance(lora, VeraModule):
                    continue
                down, up, alpha = lora.export_standard_lora_weights()
                down = down.to(device)
                up = up.to(device)
                alpha = alpha.to(device)
                dim = down.shape[0]
                scale = alpha / dim
                updown = (up @ down) * scale
                norm = updown.norm().clamp(min=max_norm_value / 2)
                desired = torch.clamp(norm, max=max_norm_value)
                ratio = desired / norm
                if float(ratio) != 1.0:
                    sqrt_ratio = torch.sqrt(ratio)
                    with torch.no_grad():
                        lora.vera_lambda_b.mul_(sqrt_ratio.to(device=lora.vera_lambda_b.device, dtype=lora.vera_lambda_b.dtype))
                        lora.vera_lambda_d.mul_(sqrt_ratio.to(device=lora.vera_lambda_d.device, dtype=lora.vera_lambda_d.dtype))
                    keys_scaled += 1
                norms.append(float((updown.norm() * ratio).item()))

            if not norms:
                return 0, 0, 0
            return keys_scaled, sum(norms) / len(norms), max(norms)

        if self.use_dora:
            return 0, 0, 0

        if self.use_lokr:
            return 0, 0, 0

        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)

