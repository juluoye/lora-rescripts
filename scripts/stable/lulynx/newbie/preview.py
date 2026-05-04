from __future__ import annotations

import argparse
import math
import os
import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from accelerate import Accelerator
from PIL import Image

from library.device_utils import clean_memory_on_device
from library.train_sampling_util import load_prompts

from .bridge import (
    create_newbie_transport,
    load_newbie_clip_model_and_tokenizer,
    load_newbie_text_encoder_and_tokenizer,
    load_newbie_vae,
    resolve_dtype,
)
from .cache import _autocast_context
from .memory import release_newbie_runtime_modules


_PREVIEW_RESOURCE_LOCK = threading.RLock()
_PREVIEW_PREWARM_LOCK = threading.Lock()
_PREVIEW_RESOURCE_CACHE: dict[tuple[str, str, bool], dict[str, object]] = {}
_PREVIEW_PREWARM_STARTED: set[tuple[str, str, bool]] = set()
_PREVIEW_PREWARM_THREADS: dict[tuple[str, str, bool], threading.Thread] = {}


def _preview_resource_cache_key(config) -> tuple[str, str, bool]:
    return (
        str(getattr(config, "pretrained_model_name_or_path", "") or ""),
        str(getattr(config, "mixed_precision", "bf16") or "bf16").strip().lower(),
        bool(getattr(config, "trust_remote_code", True)),
    )


def _load_preview_resources_to_cpu(config) -> dict[str, object]:
    text_encoder, tokenizer = load_newbie_text_encoder_and_tokenizer(
        config.pretrained_model_name_or_path,
        mixed_precision=config.mixed_precision,
        trust_remote_code=config.trust_remote_code,
    )
    clip_model, clip_tokenizer = load_newbie_clip_model_and_tokenizer(
        config.pretrained_model_name_or_path,
        mixed_precision=config.mixed_precision,
    )
    vae = load_newbie_vae(config.pretrained_model_name_or_path, trust_remote_code=config.trust_remote_code)

    try:
        text_encoder = text_encoder.to("cpu")
    except Exception:
        pass
    try:
        clip_model = clip_model.to("cpu")
    except Exception:
        pass
    try:
        vae = vae.to("cpu")
    except Exception:
        pass

    text_encoder.eval()
    clip_model.eval()
    vae.eval()
    text_encoder.requires_grad_(False)
    clip_model.requires_grad_(False)
    vae.requires_grad_(False)

    return {
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "clip_model": clip_model,
        "clip_tokenizer": clip_tokenizer,
        "vae": vae,
    }


def prewarm_preview_resources(config) -> bool:
    if not bool(getattr(config, "enable_preview", False)):
        return False
    if not str(getattr(config, "sample_prompts", "") or "").strip():
        return False

    cache_key = _preview_resource_cache_key(config)
    with _PREVIEW_RESOURCE_LOCK:
        if cache_key in _PREVIEW_RESOURCE_CACHE:
            return False
        if cache_key in _PREVIEW_PREWARM_STARTED:
            return False

    with _PREVIEW_PREWARM_LOCK:
        with _PREVIEW_RESOURCE_LOCK:
            if cache_key in _PREVIEW_RESOURCE_CACHE or cache_key in _PREVIEW_PREWARM_STARTED:
                return False
            _PREVIEW_PREWARM_STARTED.add(cache_key)

        try:
            started_at = time.perf_counter()
            print("[newbie-preview] prewarm start: loading preview models to CPU cache...", flush=True)
            resources = _load_preview_resources_to_cpu(config)
            elapsed = time.perf_counter() - started_at
            with _PREVIEW_RESOURCE_LOCK:
                _PREVIEW_RESOURCE_CACHE[cache_key] = resources
            print(f"[newbie-preview] prewarm ready: preview models cached on CPU in {elapsed:.1f}s", flush=True)
            return True
        except Exception as exc:
            print(f"[newbie-preview][warn] prewarm failed, will fall back to on-demand loading: {exc}", flush=True)
            return False
        finally:
            with _PREVIEW_RESOURCE_LOCK:
                _PREVIEW_PREWARM_STARTED.discard(cache_key)


def start_preview_prewarm_async(config) -> bool:
    if not bool(getattr(config, "enable_preview", False)):
        return False
    if not str(getattr(config, "sample_prompts", "") or "").strip():
        return False

    cache_key = _preview_resource_cache_key(config)
    with _PREVIEW_RESOURCE_LOCK:
        if cache_key in _PREVIEW_RESOURCE_CACHE:
            return False
        existing_thread = _PREVIEW_PREWARM_THREADS.get(cache_key)
        if existing_thread is not None and existing_thread.is_alive():
            return False

        worker = threading.Thread(
            target=prewarm_preview_resources,
            args=(config,),
            name="newbie-preview-prewarm",
            daemon=True,
        )
        _PREVIEW_PREWARM_THREADS[cache_key] = worker
        worker.start()
        return True


def _acquire_preview_resources(config) -> tuple[dict[str, object], bool]:
    cache_key = _preview_resource_cache_key(config)
    with _PREVIEW_RESOURCE_LOCK:
        cached = _PREVIEW_RESOURCE_CACHE.get(cache_key)
        if cached is not None:
            return cached, True
    return _load_preview_resources_to_cpu(config), False


def _should_sample(args: argparse.Namespace, epoch, steps: int) -> bool:
    if steps == 0:
        return bool(getattr(args, "sample_at_first", False))

    sample_every_n_steps = getattr(args, "sample_every_n_steps", None)
    sample_every_n_epochs = getattr(args, "sample_every_n_epochs", None)

    if sample_every_n_steps is None and sample_every_n_epochs is None:
        return False

    if sample_every_n_epochs is not None:
        return epoch is not None and int(epoch) % int(sample_every_n_epochs) == 0

    return epoch is None and int(steps) % int(sample_every_n_steps) == 0


def _to_preview_args(config) -> argparse.Namespace:
    sample_prompts = str(getattr(config, "sample_prompts", "") or "").strip() or None
    sample_every_n_steps = getattr(config, "sample_every_n_steps", None)
    sample_every_n_epochs = getattr(config, "sample_every_n_epochs", None)
    sample_seed = getattr(config, "sample_seed", None)

    try:
        if sample_every_n_steps in ("", None):
            sample_every_n_steps = None
        else:
            sample_every_n_steps = int(sample_every_n_steps)
    except (TypeError, ValueError):
        sample_every_n_steps = None

    try:
        if sample_every_n_epochs in ("", None):
            sample_every_n_epochs = None
        else:
            sample_every_n_epochs = int(sample_every_n_epochs)
    except (TypeError, ValueError):
        sample_every_n_epochs = None

    try:
        if sample_seed in ("", None):
            sample_seed = None
        else:
            sample_seed = int(sample_seed)
            if sample_seed == 0:
                sample_seed = None
    except (TypeError, ValueError):
        sample_seed = None

    return argparse.Namespace(
        sample_prompts=sample_prompts,
        sample_at_first=bool(getattr(config, "sample_at_first", False)),
        sample_every_n_steps=sample_every_n_steps,
        sample_every_n_epochs=sample_every_n_epochs,
        output_dir=str(config.output_dir),
        output_name=str(config.output_name),
        sample_width=int(getattr(config, "sample_width", config.resolution_width) or config.resolution_width),
        sample_height=int(getattr(config, "sample_height", config.resolution_height) or config.resolution_height),
        sample_steps=int(getattr(config, "sample_steps", 24) or 24),
        sample_cfg=float(getattr(config, "sample_cfg", 7.0) or 7.0),
        sample_seed=sample_seed,
        sample_sampler=str(getattr(config, "sample_sampler", "euler_a") or "euler_a").strip().lower(),
        mixed_precision=str(getattr(config, "mixed_precision", "bf16") or "bf16").strip().lower(),
    )


def _build_negative_condition(
    *,
    cap_feats: torch.Tensor,
    cap_mask: torch.Tensor,
    clip_text_pooled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    neg_cap_feats = torch.zeros_like(cap_feats)
    neg_cap_mask = torch.zeros_like(cap_mask)
    neg_clip_text_pooled = torch.zeros_like(clip_text_pooled)
    return neg_cap_feats, neg_cap_mask, neg_clip_text_pooled


@torch.no_grad()
def _encode_prompt(
    prompt: str,
    *,
    config,
    text_encoder,
    tokenizer,
    clip_model,
    clip_tokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gemma_prefix = str(getattr(config, "gemma3_prompt", "") or "")
    gemma_text = f"{gemma_prefix}{prompt}" if gemma_prefix else prompt
    target_dtype = torch.bfloat16 if str(getattr(config, "mixed_precision", "bf16")).lower() == "bf16" else (
        torch.float16 if str(getattr(config, "mixed_precision", "bf16")).lower() == "fp16" else torch.float32
    )

    with _autocast_context(device, target_dtype):
        gemma_inputs = tokenizer(
            [gemma_text],
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=int(getattr(config, "newbie_gemma_max_token_length", 512) or 512),
            return_tensors="pt",
        ).to(device)
        gemma_outputs = text_encoder(**gemma_inputs, output_hidden_states=True)
        cap_feats = gemma_outputs.hidden_states[-2].to(dtype=target_dtype)
        cap_mask = gemma_inputs.attention_mask

        clip_inputs = clip_tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=int(getattr(config, "newbie_clip_max_token_length", 2048) or 2048),
            return_tensors="pt",
        ).to(device)
        clip_text_pooled = clip_model.get_text_features(**clip_inputs).to(dtype=target_dtype)

    return cap_feats, cap_mask, clip_text_pooled


def _decode_latents(vae, latents: torch.Tensor) -> Image.Image:
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.13025) or 0.13025)
    image = vae.decode(latents / scaling_factor).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].detach().float().cpu()
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255.0).round().clip(0, 255).astype("uint8")
    return Image.fromarray(image)


def _build_prompt_text(prompt_dict: dict[str, Any]) -> str:
    return str(prompt_dict.get("prompt", "") or "").strip()


def _get_prompt_seed(prompt_dict: dict[str, Any], fallback_seed: int | None) -> int | None:
    seed = prompt_dict.get("seed", fallback_seed)
    try:
        if seed in ("", None):
            return None
        seed = int(seed)
        if seed == 0:
            return None
        return seed
    except (TypeError, ValueError):
        return None


def _get_prompt_width_height(prompt_dict: dict[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    width = int(prompt_dict.get("width", args.sample_width) or args.sample_width)
    height = int(prompt_dict.get("height", args.sample_height) or args.sample_height)
    width = max(64, width - width % 16)
    height = max(64, height - height % 16)
    return width, height


def _get_prompt_steps_cfg(prompt_dict: dict[str, Any], args: argparse.Namespace) -> tuple[int, float]:
    sample_steps = int(prompt_dict.get("sample_steps", args.sample_steps) or args.sample_steps)
    sample_cfg = float(prompt_dict.get("scale", args.sample_cfg) or args.sample_cfg)
    return max(1, sample_steps), sample_cfg


def _resolve_preview_solver_name(sample_sampler: str) -> str:
    normalized = str(sample_sampler or "euler_a").strip().lower()
    mapping = {
        "euler": "euler",
        "euler_a": "euler",
        "heun": "heun",
        "dpm_2": "dopri5",
        "dpm_2_a": "dopri5",
        "dpmsolver": "dopri5",
        "dpmsolver++": "dopri5",
    }
    return mapping.get(normalized, "euler")


@torch.no_grad()
def sample_images(
    accelerator: Accelerator,
    config,
    model,
    *,
    epoch=None,
    steps: int = 0,
) -> None:
    if not bool(getattr(config, "enable_preview", False)):
        return

    args = _to_preview_args(config)
    if not args.sample_prompts or not _should_sample(args, epoch, steps):
        return
    if not os.path.isfile(args.sample_prompts):
        return

    prompts = load_prompts(args.sample_prompts)
    if not prompts:
        return

    save_dir = Path(args.output_dir) / "sample"
    save_dir.mkdir(parents=True, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        cuda_rng_state = None

    unwrapped_model = accelerator.unwrap_model(model)
    restore_training = bool(getattr(unwrapped_model, "training", False))
    unwrapped_model.eval()

    device = accelerator.device
    model_dtype = resolve_dtype(getattr(config, "mixed_precision", "bf16"))
    text_encoder = tokenizer = clip_model = clip_tokenizer = vae = None
    resources_from_cache = False
    try:
        stage_started_at = time.perf_counter()
        resources, resources_from_cache = _acquire_preview_resources(config)
        text_encoder = resources["text_encoder"]
        tokenizer = resources["tokenizer"]
        clip_model = resources["clip_model"]
        clip_tokenizer = resources["clip_tokenizer"]
        vae = resources["vae"]
        acquisition_elapsed = time.perf_counter() - stage_started_at
        print(
            f"[newbie-preview] resources {'reused from CPU cache' if resources_from_cache else 'loaded on demand'} "
            f"in {acquisition_elapsed:.1f}s",
            flush=True,
        )

        move_started_at = time.perf_counter()
        print("[newbie-preview] moving preview models to GPU...", flush=True)
        text_encoder = text_encoder.to(device)
        clip_model = clip_model.to(device)
        vae = vae.to(device)
        print(f"[newbie-preview] GPU move ready in {time.perf_counter() - move_started_at:.1f}s", flush=True)
        text_encoder.eval()
        clip_model.eval()
        vae.eval()
        text_encoder.requires_grad_(False)
        clip_model.requires_grad_(False)
        vae.requires_grad_(False)

        transport, _ = create_newbie_transport(
            repo_root=config.repo_root,
            resolution=max(args.sample_width, args.sample_height),
        )
        sampler = transport.__class__.__module__
        del sampler  # keep linter quiet in runtime-only code paths
        from lulynx_newbie_upstream_transport.transport import Sampler as UpstreamSampler

        solver = UpstreamSampler(transport).sample_ode(
            sampling_method=_resolve_preview_solver_name(args.sample_sampler),
            num_steps=max(2, int(args.sample_steps)),
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            do_shift=True,
        )

        for prompt_dict in prompts:
            prompt = _build_prompt_text(prompt_dict)
            if not prompt:
                continue

            width, height = _get_prompt_width_height(prompt_dict, args)
            sample_steps, sample_cfg = _get_prompt_steps_cfg(prompt_dict, args)
            seed = _get_prompt_seed(prompt_dict, args.sample_seed)

            if seed is None:
                seed = random.randint(1, 2**31 - 1)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            print(
                f"[newbie-preview] sampling start: {width}x{height}, steps={sample_steps}, cfg={sample_cfg:g}, seed={seed}",
                flush=True,
            )

            cap_feats, cap_mask, clip_text_pooled = _encode_prompt(
                prompt,
                config=config,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                clip_model=clip_model,
                clip_tokenizer=clip_tokenizer,
                device=device,
            )
            neg_cap_feats, neg_cap_mask, neg_clip_text_pooled = _build_negative_condition(
                cap_feats=cap_feats,
                cap_mask=cap_mask,
                clip_text_pooled=clip_text_pooled,
            )

            latent_height = height // 8
            latent_width = width // 8
            generator = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn(
                1,
                int(getattr(vae.config, "latent_channels", 16) or 16),
                latent_height,
                latent_width,
                device=device,
                dtype=model_dtype,
                generator=generator,
            )

            if sample_cfg != 1.0:
                doubled_latents = torch.cat([latents, latents], dim=0)
                doubled_timesteps = None

                def _cfg_model(x, t, **kwargs):
                    nonlocal doubled_timesteps
                    if doubled_timesteps is None or doubled_timesteps.shape[0] != x.shape[0]:
                        doubled_timesteps = t
                    with _autocast_context(device, model_dtype):
                        return unwrapped_model.forward_with_cfg(
                            x,
                            doubled_timesteps,
                            torch.cat([cap_feats, neg_cap_feats], dim=0),
                            torch.cat([cap_mask, neg_cap_mask], dim=0),
                            cfg_scale=sample_cfg,
                            clip_text_pooled=torch.cat([clip_text_pooled, neg_clip_text_pooled], dim=0),
                        )

                sampled = solver(doubled_latents, _cfg_model)[-1][:1]
            else:
                def _model(x, t, **kwargs):
                    with _autocast_context(device, model_dtype):
                        return unwrapped_model(
                            x,
                            t,
                            cap_feats,
                            cap_mask,
                            clip_text_pooled=clip_text_pooled,
                        )

                sampled = solver(latents, _model)[-1]

            image = _decode_latents(vae, sampled)
            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            suffix = f"e{int(epoch):06d}" if epoch is not None else f"{int(steps):06d}"
            enum_index = int(prompt_dict.get("enum", 0) or 0)
            output_name = (
                f"{args.output_name}_{suffix}_{enum_index:02d}_{ts_str}_{seed}.png"
                if args.output_name
                else f"{suffix}_{enum_index:02d}_{ts_str}_{seed}.png"
            )
            image.save(save_dir / output_name)
            print(f"[newbie-preview] image saved: {save_dir / output_name}", flush=True)
    finally:
        if restore_training:
            unwrapped_model.train(True)
        if torch.cuda.is_available() and cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        torch.set_rng_state(rng_state)
        if resources_from_cache:
            for module in (text_encoder, clip_model, vae):
                if module is None:
                    continue
                try:
                    module.to("cpu")
                except Exception:
                    pass
            clean_memory_on_device(device)
        else:
            release_newbie_runtime_modules(text_encoder, clip_model, vae, device=device)
