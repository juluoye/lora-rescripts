from __future__ import annotations

import importlib
import importlib.metadata as metadata
from typing import Any, Callable

import torch
from torch.nn import functional as F


def _resolve_package_version(*package_names: str) -> str:
    for package_name in package_names:
        try:
            return metadata.version(package_name)
        except Exception:
            continue
    return ""


def _run_sdpa_fixed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str,
    is_causal: bool,
    sm_scale: float | None,
) -> torch.Tensor:
    if tensor_layout == "HND":
        q_sdpa = q
        k_sdpa = k
        v_sdpa = v
    elif tensor_layout == "NHD":
        q_sdpa = q.permute(0, 2, 1, 3).contiguous()
        k_sdpa = k.permute(0, 2, 1, 3).contiguous()
        v_sdpa = v.permute(0, 2, 1, 3).contiguous()
    else:
        raise ValueError(f"Unsupported tensor_layout: {tensor_layout}")

    sdpa_kwargs = {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": is_causal,
    }
    if sm_scale is not None:
        sdpa_kwargs["scale"] = sm_scale

    out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, **sdpa_kwargs)
    if tensor_layout == "NHD":
        out = out.permute(0, 2, 1, 3).contiguous()
    return out


def _run_sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    is_causal: bool,
    sm_scale: float | None,
) -> torch.Tensor:
    q_offsets = [int(x) for x in cu_seqlens_q.detach().to(device="cpu", dtype=torch.int64).tolist()]
    k_offsets = [int(x) for x in cu_seqlens_k.detach().to(device="cpu", dtype=torch.int64).tolist()]

    outputs: list[torch.Tensor] = []
    for batch_index in range(len(q_offsets) - 1):
        q_start = q_offsets[batch_index]
        q_end = q_offsets[batch_index + 1]
        k_start = k_offsets[batch_index]
        k_end = k_offsets[batch_index + 1]

        if q_end <= q_start:
            outputs.append(q.new_empty((0, q.shape[1], q.shape[2])))
            continue

        q_slice = q[q_start:q_end].permute(1, 0, 2).unsqueeze(0).contiguous()
        k_slice = k[k_start:k_end].permute(1, 0, 2).unsqueeze(0).contiguous()
        v_slice = v[k_start:k_end].permute(1, 0, 2).unsqueeze(0).contiguous()

        out_hnd = _run_sdpa_fixed(
            q_slice,
            k_slice,
            v_slice,
            tensor_layout="HND",
            is_causal=is_causal,
            sm_scale=sm_scale,
        )
        outputs.append(out_hnd.squeeze(0).permute(1, 0, 2).contiguous())

    if not outputs:
        return q.new_empty((0, q.shape[1], q.shape[2]))
    return torch.cat(outputs, dim=0)


def _spas_supports_fixed_kernel(q: torch.Tensor, *, tensor_layout: str) -> bool:
    if not isinstance(q, torch.Tensor) or q.device.type != "cuda" or q.ndim != 4:
        return False

    if tensor_layout == "HND":
        seq_len = int(q.shape[-2])
    elif tensor_layout == "NHD":
        seq_len = int(q.shape[-3])
    else:
        return False

    head_dim = int(q.shape[-1])
    return seq_len >= 128 and head_dim in {64, 128}


def _build_spas_fixed_wrapper(kernel_fn: Callable[..., Any]) -> Callable[..., Any]:
    def _wrapped(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        tensor_layout: str = "HND",
        is_causal: bool = False,
        sm_scale: float | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if not _spas_supports_fixed_kernel(q, tensor_layout=tensor_layout):
            return _run_sdpa_fixed(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale)

        return kernel_fn(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=sm_scale,
            tensor_layout=tensor_layout,
        )

    return _wrapped


def _build_spas_varlen_wrapper(fixed_wrapper: Callable[..., Any]) -> Callable[..., Any]:
    def _wrapped(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        *,
        is_causal: bool = False,
        sm_scale: float | None = None,
        **_: Any,
    ) -> torch.Tensor:
        del max_seqlen_q, max_seqlen_k

        q_offsets = [int(x) for x in cu_seqlens_q.detach().to(device="cpu", dtype=torch.int64).tolist()]
        k_offsets = [int(x) for x in cu_seqlens_k.detach().to(device="cpu", dtype=torch.int64).tolist()]

        outputs: list[torch.Tensor] = []
        for batch_index in range(len(q_offsets) - 1):
            q_start = q_offsets[batch_index]
            q_end = q_offsets[batch_index + 1]
            k_start = k_offsets[batch_index]
            k_end = k_offsets[batch_index + 1]

            if q_end <= q_start:
                outputs.append(q.new_empty((0, q.shape[1], q.shape[2])))
                continue

            q_slice = q[q_start:q_end].unsqueeze(0).contiguous()
            k_slice = k[k_start:k_end].unsqueeze(0).contiguous()
            v_slice = v[k_start:k_end].unsqueeze(0).contiguous()

            if _spas_supports_fixed_kernel(q_slice, tensor_layout="NHD"):
                out_slice = fixed_wrapper(
                    q_slice,
                    k_slice,
                    v_slice,
                    tensor_layout="NHD",
                    is_causal=is_causal,
                    sm_scale=sm_scale,
                )
            else:
                out_slice = _run_sdpa_varlen(
                    q[q_start:q_end],
                    k[k_start:k_end],
                    v[k_start:k_end],
                    cu_seqlens_q=torch.tensor([0, q_end - q_start], device=q.device, dtype=torch.int32),
                    cu_seqlens_k=torch.tensor([0, k_end - k_start], device=k.device, dtype=torch.int32),
                    is_causal=is_causal,
                    sm_scale=sm_scale,
                )
                outputs.append(out_slice)
                continue

            outputs.append(out_slice.squeeze(0).contiguous())

        if not outputs:
            return q.new_empty((0, q.shape[1], q.shape[2]))
        return torch.cat(outputs, dim=0)

    return _wrapped


def _load_package_sageattention_symbols() -> tuple[Callable[..., Any], Callable[..., Any], str]:
    sage_module = importlib.import_module("sageattention")
    sageattn = getattr(sage_module, "sageattn", None)
    sageattn_varlen = getattr(sage_module, "sageattn_varlen", None)
    if not callable(sageattn) or not callable(sageattn_varlen):
        raise ImportError("required SageAttention symbols are missing")
    return sageattn, sageattn_varlen, "sageattention"


def _load_spas_sageattention_symbols() -> tuple[Callable[..., Any], Callable[..., Any], str]:
    spas_module = importlib.import_module("spas_sage_attn")
    fixed_kernel = (
        getattr(spas_module, "spas_sage2_attn_meansim_cuda", None)
        or getattr(spas_module, "spas_sage_attn_meansim_cuda", None)
    )
    if not callable(fixed_kernel):
        raise ImportError("required SpargeAttn2 symbols are missing")

    fixed_wrapper = _build_spas_fixed_wrapper(fixed_kernel)
    varlen_wrapper = _build_spas_varlen_wrapper(fixed_wrapper)
    return fixed_wrapper, varlen_wrapper, "spas_sage_attn"


def load_runtime_sageattention_version() -> str:
    return _resolve_package_version("sageattention", "spas_sage_attn")


def load_runtime_sageattention_core_module() -> Any:
    try:
        return importlib.import_module("sageattention.core")
    except Exception:
        return importlib.import_module("spas_sage_attn.core")


def load_runtime_sageattention_symbols() -> tuple[Callable[..., Any], Callable[..., Any], str]:
    try:
        sageattn, sageattn_varlen, source = _load_package_sageattention_symbols()
    except Exception as exc:
        try:
            sageattn, sageattn_varlen, source = _load_spas_sageattention_symbols()
        except Exception as spas_exc:
            raise ImportError(f"package import failed: {exc}; spas import failed: {spas_exc}") from spas_exc
    return sageattn, sageattn_varlen, source


def probe_runtime_sageattention() -> dict[str, Any]:
    result: dict[str, Any] = {
        "ready": False,
        "importable": False,
        "source": "",
        "source_root": "",
        "reason": "",
    }

    try:
        sageattn, sageattn_varlen, source = load_runtime_sageattention_symbols()
    except Exception as exc:
        result["reason"] = str(exc)
        return result

    result["importable"] = True
    result["ready"] = callable(sageattn) and callable(sageattn_varlen)
    result["source"] = source
    if source == "sageattention":
        try:
            result["source_root"] = str(importlib.import_module("sageattention").__file__ or "")
        except Exception:
            result["source_root"] = ""
    elif source == "spas_sage_attn":
        try:
            result["source_root"] = str(importlib.import_module("spas_sage_attn").__file__ or "")
        except Exception:
            result["source_root"] = ""
    if not result["ready"]:
        result["reason"] = "required SageAttention symbols are missing"
    return result
