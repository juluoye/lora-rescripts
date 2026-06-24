from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
if "--save-only" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

DISPLAY_TIMESTEP_MIN = 0
DISPLAY_TIMESTEP_MAX = 1000
PRACTICAL_DENSITY_Y_LIMIT = 0.005


def is_density_display_mode(display_mode: str) -> bool:
    return display_mode.startswith("density")


def compute_histogram(
    values: np.ndarray,
    bins: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    display_mode: str = "density",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    density = is_density_display_mode(display_mode)
    hist, edges = np.histogram(values, bins=bins, weights=weights, density=density)
    widths = np.diff(edges)
    area = float(np.sum(hist * widths)) if hist.size else 0.0
    total = float(np.sum(weights)) if weights is not None else float(values.size)
    return hist, edges, area, total


def draw_histogram(
    ax,
    hist: np.ndarray,
    edges: np.ndarray,
    *,
    color: str,
    title: str,
    ylabel: str,
    summary_text: str,
    y_limit: float,
) -> None:
    ax.cla()
    ax.bar(edges[:-1], hist, width=np.diff(edges), align="edge", color=color, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel(ylabel)
    ax.set_xlim(DISPLAY_TIMESTEP_MIN, DISPLAY_TIMESTEP_MAX)
    ax.set_ylim(0.0, y_limit)
    ax.text(
        0.98,
        0.95,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )


def format_histogram_summary(
    *,
    display_mode: str,
    area: float,
    total: float,
    peak: float,
    y_limit: float,
    total_label: str,
) -> str:
    if is_density_display_mode(display_mode):
        summary = f"area≈{area:.4f} peak≈{peak:.4f}"
    else:
        summary = f"{total_label}≈{total:.1f} peak≈{peak:.1f}"
    if peak > y_limit:
        summary += " clipped"
    return summary


def resolve_histogram_y_limit(
    hist_bins: np.ndarray,
    *,
    display_mode: str,
    sample_hist: np.ndarray,
    effective_hist: np.ndarray,
) -> float:
    if display_mode == "density(compare)":
        return PRACTICAL_DENSITY_Y_LIMIT
    if display_mode == "density(zoom)":
        return max(float(sample_hist.max(initial=0.0)), float(effective_hist.max(initial=0.0)), 1e-6) * 1.08
    return max(float(sample_hist.max(initial=0.0)), float(effective_hist.max(initial=0.0)), 1e-6) * 1.08


def set_axis_visibility(ax, visible: bool) -> None:
    ax.set_visible(visible)
    for child in ax.get_children():
        try:
            child.set_visible(visible)
        except Exception:
            continue


def apply_shift_density(u: np.ndarray, shift: float) -> np.ndarray:
    if shift <= 0:
        raise ValueError("shift must be > 0")
    if shift == 1.0:
        return u
    return (u * shift) / (1.0 + (shift - 1.0) * u)


def normalize_timestep_positions(timesteps: np.ndarray, min_timestep: int, max_timestep: int) -> np.ndarray:
    max_index = max_timestep - 1
    if max_index <= min_timestep:
        return np.zeros_like(timesteps, dtype=np.float64)
    normalized = (timesteps.astype(np.float64) - float(min_timestep)) / float(max_index - min_timestep)
    return np.clip(normalized, 0.0, 1.0)


def density_to_timesteps(u: np.ndarray, min_timestep: int, max_timestep: int) -> np.ndarray:
    t_range = max_timestep - min_timestep
    if t_range <= 0:
        return np.full_like(u, fill_value=max_timestep, dtype=np.int64)
    timesteps = np.floor(min_timestep + u * t_range).astype(np.int64)
    return np.clip(timesteps, min_timestep, max_timestep - 1)


def compute_sdxl_loss_weight_curve(mode: str, normalized_timesteps: np.ndarray, scale: float, shift: float) -> np.ndarray:
    if mode in {"none", "uniform"}:
        return np.ones_like(normalized_timesteps, dtype=np.float64)

    if mode == "linear":
        curve = 1.0 - normalized_timesteps
    elif mode == "cosine":
        curve = 0.5 + 0.5 * np.cos(normalized_timesteps * np.pi)
    else:
        if scale == 0:
            noise_side = np.full_like(normalized_timesteps, 0.5, dtype=np.float64)
        else:
            noise_side = 1.0 / (1.0 + np.exp(-((normalized_timesteps - 0.5) * scale)))

        if mode == "shift":
            noise_side = apply_shift_density(noise_side, shift)
        elif mode != "sigmoid":
            raise ValueError(f"unsupported weight mode: {mode}")

        curve = 1.0 - noise_side

    curve = np.clip(curve, 1e-3, None)
    mean = float(curve.mean()) if curve.size else 1.0
    if mean > 0:
        curve = curve / mean
    return curve


def compute_anima_loss_weight_curve(mode: str, sigmas: np.ndarray, *, _scale: float, _shift: float) -> np.ndarray:
    if mode in {"uniform", "none"}:
        curve = np.ones_like(sigmas, dtype=np.float64)
    elif mode == "sigma_sqrt":
        curve = np.power(np.clip(sigmas, 1e-4, None), -2.0)
    elif mode == "cosmap":
        bot = 1.0 - 2.0 * sigmas + 2.0 * np.square(sigmas)
        curve = 2.0 / (np.pi * np.clip(bot, 1e-6, None))
    else:
        curve = np.ones_like(sigmas, dtype=np.float64)

    curve = np.clip(curve, 1e-3, None)
    mean = float(curve.mean()) if curve.size else 1.0
    if mean > 0:
        curve = curve / mean
    return curve


def get_lin_function(x1: float = 256.0, y1: float = 0.5, x2: float = 4096.0, y2: float = 1.15):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return lambda x: slope * x + intercept


def get_anima_flux_packed_seq_len(width: int, height: int) -> float:
    # Anima computes FLUX shift from latent-space dimensions: H/8, W/8, then packed by 2x2.
    latent_h = max(1, width // 8)
    latent_w = max(1, height // 8)
    return max(1.0, float((latent_h // 2) * (latent_w // 2)))


def time_shift(mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 1e-6, 1.0 - 1e-6)
    return math.exp(mu) / (math.exp(mu) + np.power(1.0 / t - 1.0, sigma))


def sample_sdxl_density(mode: str, scale: float, shift: float, size: int, rng: np.random.Generator) -> np.ndarray:
    if mode == "uniform":
        return rng.random(size)

    x = rng.standard_normal(size)
    if scale == 0:
        u = np.full(size, 0.5, dtype=np.float64)
    else:
        u = 1.0 / (1.0 + np.exp(-(x * scale)))

    if mode == "shift":
        u = apply_shift_density(u, shift)
    elif mode != "sigmoid":
        raise ValueError(f"unsupported mode: {mode}")

    return u


def sample_anima_density(
    mode: str,
    scale: float,
    shift: float,
    size: int,
    rng: np.random.Generator,
    *,
    width: int,
    height: int,
    weighting_scheme: str,
    logit_mean: float,
    logit_std: float,
    mode_scale: float,
) -> np.ndarray:
    if mode == "uniform":
        return rng.random(size)
    if mode == "sigma":
        if weighting_scheme == "logit_normal":
            samples = rng.normal(loc=logit_mean, scale=max(logit_std, 1e-3), size=size)
            return 1.0 / (1.0 + np.exp(-samples))
        if weighting_scheme == "mode":
            u = rng.random(size)
            return 1.0 - u - mode_scale * (np.cos(np.pi * u / 2.0) ** 2 - 1.0 + u)
        return rng.random(size)

    x = rng.standard_normal(size)
    if scale == 0:
        u = np.full(size, 0.5, dtype=np.float64)
    else:
        u = 1.0 / (1.0 + np.exp(-(x * scale)))

    if mode == "sigmoid":
        return u
    if mode == "shift":
        return apply_shift_density(u, shift)
    if mode == "flux_shift":
        packed_seq_len = get_anima_flux_packed_seq_len(width, height)
        mu = get_lin_function(y1=0.5, y2=1.15)(packed_seq_len)
        return time_shift(mu, 1.0, u)

    raise ValueError(f"unsupported anima mode: {mode}")


def build_sdxl_preview(
    *,
    preview_mode: str = "sdxl",
    sample_mode: str,
    sample_scale: float,
    sample_shift: float,
    weight_mode: str,
    weight_scale: float,
    weight_shift: float,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    min_timestep: int,
    max_timestep: int,
    samples: int,
    seed: int,
    width: int = 1024,
    height: int = 1024,
):
    rng = np.random.default_rng(seed)
    x = np.linspace(-4.0, 4.0, 400)
    base = 1.0 / (1.0 + np.exp(-(x * sample_scale))) if sample_scale != 0 else np.full_like(x, 0.5)
    if sample_mode == "shift":
        sample_curve = apply_shift_density(base, sample_shift)
    elif sample_mode == "sigmoid":
        sample_curve = base
    else:
        sample_curve = (x - x.min()) / (x.max() - x.min())

    sampled_positions = sample_sdxl_density(sample_mode, sample_scale, sample_shift, samples, rng)
    timesteps = density_to_timesteps(sampled_positions, min_timestep, max_timestep)

    discrete_timesteps = np.arange(min_timestep, max_timestep, dtype=np.int64)
    discrete_normalized = normalize_timestep_positions(discrete_timesteps, min_timestep, max_timestep)
    loss_weights_curve = compute_sdxl_loss_weight_curve(weight_mode, discrete_normalized, weight_scale, weight_shift)

    sampled_normalized = normalize_timestep_positions(timesteps, min_timestep, max_timestep)
    sampled_loss_weights = compute_sdxl_loss_weight_curve(weight_mode, sampled_normalized, weight_scale, weight_shift)

    return {
        "sample_curve_x": x,
        "sample_curve_y": sample_curve,
        "timesteps": timesteps,
        "curve_timesteps": discrete_timesteps,
        "loss_weights_curve": loss_weights_curve,
        "sampled_loss_weights": sampled_loss_weights,
        "sample_curve_title": "Sampling Transform Curve",
        "sample_curve_xlabel": "standard normal sample x",
        "sample_curve_ylabel": "mapped cumulative position u",
        "weight_curve_title": "Timestep Loss Weight Curve",
        "weight_curve_ylabel": "normalized loss weight",
        "mode_label": "sdxl",
    }


def build_anima_preview(
    *,
    preview_mode: str = "anima",
    sample_mode: str,
    sample_scale: float,
    sample_shift: float,
    weight_mode: str,
    weight_scale: float,
    weight_shift: float,
    logit_mean: float,
    logit_std: float,
    mode_scale: float,
    min_timestep: int,
    max_timestep: int,
    samples: int,
    seed: int,
    width: int,
    height: int,
):
    rng = np.random.default_rng(seed)
    x = np.linspace(-4.0, 4.0, 400)
    base = 1.0 / (1.0 + np.exp(-(x * sample_scale))) if sample_scale != 0 else np.full_like(x, 0.5)
    if sample_mode == "shift":
        sample_curve = apply_shift_density(base, sample_shift)
    elif sample_mode == "flux_shift":
        packed_seq_len = get_anima_flux_packed_seq_len(width, height)
        mu = get_lin_function(y1=0.5, y2=1.15)(packed_seq_len)
        sample_curve = time_shift(mu, 1.0, base)
    elif sample_mode in {"sigmoid", "sigma"}:
        sample_curve = base if sample_mode == "sigmoid" else (x - x.min()) / (x.max() - x.min())
    else:
        sample_curve = (x - x.min()) / (x.max() - x.min())

    sampled_positions = sample_anima_density(
        sample_mode,
        sample_scale,
        sample_shift,
        samples,
        rng,
        width=width,
        height=height,
        weighting_scheme=weight_mode,
        logit_mean=logit_mean,
        logit_std=logit_std,
        mode_scale=mode_scale,
    )
    timesteps = density_to_timesteps(sampled_positions, min_timestep, max_timestep)

    discrete_timesteps = np.arange(min_timestep, max_timestep, dtype=np.int64)
    discrete_sigmas = np.clip(discrete_timesteps.astype(np.float64) / 1000.0, 1e-4, 1.0)
    loss_weights_curve = compute_anima_loss_weight_curve(weight_mode, discrete_sigmas, _scale=weight_scale, _shift=weight_shift)

    sampled_sigmas = np.clip(timesteps.astype(np.float64) / 1000.0, 1e-4, 1.0)
    sampled_loss_weights = compute_anima_loss_weight_curve(weight_mode, sampled_sigmas, _scale=weight_scale, _shift=weight_shift)

    return {
        "sample_curve_x": x,
        "sample_curve_y": sample_curve,
        "timesteps": timesteps,
        "curve_timesteps": discrete_timesteps,
        "loss_weights_curve": loss_weights_curve,
        "sampled_loss_weights": sampled_loss_weights,
        "sample_curve_title": "Flow Timestep Sampling Curve",
        "sample_curve_xlabel": "standard normal sample x",
        "sample_curve_ylabel": "mapped sigma position",
        "weight_curve_title": "Anima Weighting Scheme Curve",
        "weight_curve_ylabel": "normalized loss weight",
        "mode_label": "anima",
    }


def build_preview(**state):
    preview_args = dict(state)
    preview_args.pop("hist_display_mode", None)
    preview_mode = preview_args["preview_mode"]
    if preview_mode == "anima":
        return build_anima_preview(**preview_args)
    return build_sdxl_preview(**preview_args)


def draw_preview_figure(fig, axes, state: dict) -> None:
    preview = build_preview(**state)
    ax_sample_curve, ax_sample_hist = axes[0]
    ax_weight_curve, ax_effective = axes[1]

    hist_bins = np.linspace(DISPLAY_TIMESTEP_MIN, DISPLAY_TIMESTEP_MAX, 51)
    display_mode = state["hist_display_mode"]

    ax_sample_curve.cla()
    ax_sample_curve.plot(preview["sample_curve_x"], preview["sample_curve_y"], lw=2, color="#4f7cff")
    ax_sample_curve.set_title(preview["sample_curve_title"])
    ax_sample_curve.set_xlabel(preview["sample_curve_xlabel"])
    ax_sample_curve.set_ylabel(preview["sample_curve_ylabel"])
    ax_sample_curve.set_ylim(-0.02, 1.02)

    ax_weight_curve.cla()
    ax_weight_curve.plot(preview["curve_timesteps"], preview["loss_weights_curve"], lw=2, color="#ff7a45")
    ax_weight_curve.set_title(preview["weight_curve_title"])
    ax_weight_curve.set_xlabel("timestep")
    ax_weight_curve.set_ylabel(preview["weight_curve_ylabel"])
    ax_weight_curve.set_xlim(DISPLAY_TIMESTEP_MIN, DISPLAY_TIMESTEP_MAX)

    sample_hist, sample_edges, sample_area, sample_total = compute_histogram(
        preview["timesteps"],
        hist_bins,
        display_mode=display_mode,
    )
    effective_hist, effective_edges, effective_area, effective_total = compute_histogram(
        preview["timesteps"],
        hist_bins,
        weights=preview["sampled_loss_weights"],
        display_mode=display_mode,
    )
    shared_y_max = resolve_histogram_y_limit(
        hist_bins,
        display_mode=display_mode,
        sample_hist=sample_hist,
        effective_hist=effective_hist,
    )

    if is_density_display_mode(display_mode):
        sample_ylabel = "sample density"
        effective_ylabel = "weighted sample density"
    else:
        sample_ylabel = "sample count"
        effective_ylabel = "weighted sample count"

    sample_summary = format_histogram_summary(
        display_mode=display_mode,
        area=sample_area,
        total=sample_total,
        peak=float(sample_hist.max(initial=0.0)),
        y_limit=shared_y_max,
        total_label="count",
    )
    effective_summary = format_histogram_summary(
        display_mode=display_mode,
        area=effective_area,
        total=effective_total,
        peak=float(effective_hist.max(initial=0.0)),
        y_limit=shared_y_max,
        total_label="weight sum",
    )

    draw_histogram(
        ax_sample_hist,
        sample_hist,
        sample_edges,
        color="#4f7cff",
        title="Discrete Timestep Distribution",
        ylabel=sample_ylabel,
        summary_text=sample_summary,
        y_limit=shared_y_max,
    )
    draw_histogram(
        ax_effective,
        effective_hist,
        effective_edges,
        color="#30b08f",
        title="Effective Training Emphasis",
        ylabel=effective_ylabel,
        summary_text=effective_summary,
        y_limit=shared_y_max,
    )

    sample_part = f"sample={state['sample_mode']}"
    if state["sample_mode"] in {"sigmoid", "shift", "flux_shift"}:
        sample_part += f" (scale={state['sample_scale']:.2f}"
        if state["sample_mode"] == "shift":
            sample_part += f", shift={state['sample_shift']:.2f}"
        sample_part += ")"
    if state["preview_mode"] == "anima" and state["sample_mode"] == "flux_shift":
        packed_seq_len = int(get_anima_flux_packed_seq_len(state["width"], state["height"]))
        sample_part += f" packed_seq={packed_seq_len}"

    weight_part = f"loss_weight={state['weight_mode']}"
    if state["preview_mode"] == "sdxl" and state["weight_mode"] in {"sigmoid", "shift"}:
        weight_part += f" (scale={state['weight_scale']:.2f}"
        if state["weight_mode"] == "shift":
            weight_part += f", shift={state['weight_shift']:.2f}"
        weight_part += ")"

    extra_parts = []
    if state["preview_mode"] == "anima" and state["weight_mode"] == "logit_normal":
        extra_parts.append(f"logit=({state['logit_mean']:.2f},{state['logit_std']:.2f})")
    if state["preview_mode"] == "anima" and state["weight_mode"] == "mode":
        extra_parts.append(f"mode_scale={state['mode_scale']:.2f}")

    title_parts = [
        f"preview={preview['mode_label']}",
        sample_part,
        weight_part,
        *extra_parts,
        f"min={state['min_timestep']} max={state['max_timestep']}",
        f"size={state['width']}x{state['height']}",
    ]
    fig.suptitle(" | ".join(title_parts))


def save_preview_snapshot(
    output: Path,
    *,
    preview_mode: str = "sdxl",
    mode: str = "uniform",
    scale: float = 1.0,
    shift: float = 1.0,
    weight_mode: str = "none",
    weight_scale: float = 1.0,
    weight_shift: float = 1.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    min_timestep: int = 0,
    max_timestep: int = 1000,
    width: int = 1024,
    height: int = 1024,
    samples: int = 40000,
    seed: int = 1337,
    hist_display_mode: str = "density(compare)",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.92, hspace=0.34, wspace=0.24)

    state = {
        "preview_mode": preview_mode,
        "sample_mode": mode,
        "sample_scale": scale,
        "sample_shift": shift,
        "weight_mode": weight_mode,
        "weight_scale": weight_scale,
        "weight_shift": weight_shift,
        "logit_mean": logit_mean,
        "logit_std": logit_std,
        "mode_scale": mode_scale,
        "min_timestep": min_timestep,
        "max_timestep": max_timestep,
        "width": width,
        "height": height,
        "samples": samples,
        "seed": seed,
        "hist_display_mode": hist_display_mode,
    }
    if state["max_timestep"] <= state["min_timestep"]:
        state["max_timestep"] = state["min_timestep"] + 1

    draw_preview_figure(fig, axes, state)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def render_interactive(
    initial_output: Path | None,
    *,
    preview_mode: str = "sdxl",
    mode: str = "uniform",
    scale: float = 1.0,
    shift: float = 1.0,
    weight_mode: str = "none",
    weight_scale: float = 1.0,
    weight_shift: float = 1.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    min_timestep: int = 0,
    max_timestep: int = 1000,
    width: int = 1024,
    height: int = 1024,
    samples: int = 40000,
    seed: int = 1337,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_sample_curve, ax_sample_hist = axes[0]
    ax_weight_curve, ax_effective = axes[1]
    plt.subplots_adjust(left=0.08, bottom=0.46, right=0.80, hspace=0.34, wspace=0.24)

    state = {
        "preview_mode": preview_mode,
        "sample_mode": mode,
        "sample_scale": scale,
        "sample_shift": shift,
        "weight_mode": weight_mode,
        "weight_scale": weight_scale,
        "weight_shift": weight_shift,
        "logit_mean": logit_mean,
        "logit_std": logit_std,
        "mode_scale": mode_scale,
        "min_timestep": min_timestep,
        "max_timestep": max_timestep,
        "width": width,
        "height": height,
        "samples": samples,
        "seed": seed,
        "hist_display_mode": "density(compare)",
    }

    sample_mode_choices = {
        "sdxl": ("uniform", "sigmoid", "shift"),
        "anima": ("sigma", "uniform", "sigmoid", "shift", "flux_shift"),
    }
    weight_mode_choices = {
        "sdxl": ("none", "linear", "cosine", "sigmoid", "shift"),
        "anima": ("uniform", "logit_normal", "mode", "sigma_sqrt", "cosmap", "none"),
    }

    ax_sample_mode = plt.axes([0.83, 0.62, 0.14, 0.19])
    sample_options = sample_mode_choices[preview_mode]
    sample_mode_index = sample_options.index(mode) if mode in sample_options else 0
    radio_sample_mode = RadioButtons(ax_sample_mode, sample_options, active=sample_mode_index)

    ax_weight_mode = plt.axes([0.83, 0.30, 0.14, 0.27])
    weight_options = weight_mode_choices[preview_mode]
    weight_mode_index = weight_options.index(weight_mode) if weight_mode in weight_options else 0
    radio_weight_mode = RadioButtons(ax_weight_mode, weight_options, active=weight_mode_index)

    ax_hist_mode = plt.axes([0.83, 0.20, 0.14, 0.08])
    hist_mode_options = ("density(compare)", "density(zoom)", "count")
    radio_hist_mode = RadioButtons(ax_hist_mode, hist_mode_options, active=0)

    ax_sample_scale = plt.axes([0.12, 0.38, 0.58, 0.025])
    ax_sample_shift = plt.axes([0.12, 0.345, 0.58, 0.025])
    ax_weight_scale = plt.axes([0.12, 0.31, 0.58, 0.025])
    ax_weight_shift = plt.axes([0.12, 0.275, 0.58, 0.025])
    ax_logit_mean = plt.axes([0.12, 0.24, 0.58, 0.025])
    ax_logit_std = plt.axes([0.12, 0.205, 0.58, 0.025])
    ax_mode_scale = plt.axes([0.12, 0.17, 0.58, 0.025])
    ax_min = plt.axes([0.12, 0.135, 0.58, 0.025])
    ax_max = plt.axes([0.12, 0.10, 0.58, 0.025])
    ax_width = plt.axes([0.12, 0.065, 0.58, 0.025])
    ax_height = plt.axes([0.12, 0.03, 0.58, 0.025])

    slider_sample_scale = Slider(ax_sample_scale, "sample_scale", 0.0, 8.0, valinit=state["sample_scale"], valstep=0.1)
    slider_sample_shift = Slider(ax_sample_shift, "sample_shift", 0.05, 8.0, valinit=state["sample_shift"], valstep=0.05)
    slider_weight_scale = Slider(ax_weight_scale, "weight_scale", 0.0, 16.0, valinit=state["weight_scale"], valstep=0.1)
    slider_weight_shift = Slider(ax_weight_shift, "weight_shift", 0.05, 8.0, valinit=state["weight_shift"], valstep=0.05)
    slider_logit_mean = Slider(ax_logit_mean, "logit_mean", -4.0, 4.0, valinit=state["logit_mean"], valstep=0.1)
    slider_logit_std = Slider(ax_logit_std, "logit_std", 0.1, 4.0, valinit=state["logit_std"], valstep=0.1)
    slider_mode_scale = Slider(ax_mode_scale, "mode_scale", 0.0, 4.0, valinit=state["mode_scale"], valstep=0.05)
    slider_min = Slider(ax_min, "min_timestep", 0, 999, valinit=state["min_timestep"], valstep=1)
    slider_max = Slider(ax_max, "max_timestep", 1, 1000, valinit=state["max_timestep"], valstep=1)
    slider_width = Slider(ax_width, "width", 256, 4096, valinit=state["width"], valstep=64)
    slider_height = Slider(ax_height, "height", 256, 4096, valinit=state["height"], valstep=64)

    slider_axes = {
        "sample_scale": ax_sample_scale,
        "sample_shift": ax_sample_shift,
        "weight_scale": ax_weight_scale,
        "weight_shift": ax_weight_shift,
        "logit_mean": ax_logit_mean,
        "logit_std": ax_logit_std,
        "mode_scale": ax_mode_scale,
        "min_timestep": ax_min,
        "max_timestep": ax_max,
        "width": ax_width,
        "height": ax_height,
    }

    def refresh_control_visibility():
        sample_mode = state["sample_mode"]
        weight_mode = state["weight_mode"]
        is_anima = state["preview_mode"] == "anima"

        visible_axes = {
            "sample_scale": sample_mode in {"sigmoid", "shift", "flux_shift"},
            "sample_shift": sample_mode == "shift",
            "weight_scale": (not is_anima) and weight_mode in {"sigmoid", "shift"},
            "weight_shift": (not is_anima) and weight_mode == "shift",
            "logit_mean": is_anima and weight_mode == "logit_normal",
            "logit_std": is_anima and weight_mode == "logit_normal",
            "mode_scale": is_anima and weight_mode == "mode",
            "min_timestep": True,
            "max_timestep": True,
            "width": is_anima and sample_mode == "flux_shift",
            "height": is_anima and sample_mode == "flux_shift",
        }

        for key, ax in slider_axes.items():
            set_axis_visibility(ax, visible_axes.get(key, False))

    def render():
        refresh_control_visibility()
        draw_preview_figure(fig, axes, state)
        fig.canvas.draw_idle()

        if initial_output is not None:
            fig.savefig(initial_output, dpi=140)

    def update(_=None):
        state["sample_mode"] = radio_sample_mode.value_selected
        state["sample_scale"] = float(slider_sample_scale.val)
        state["sample_shift"] = float(slider_sample_shift.val)
        state["weight_mode"] = radio_weight_mode.value_selected
        state["hist_display_mode"] = radio_hist_mode.value_selected
        state["weight_scale"] = float(slider_weight_scale.val)
        state["weight_shift"] = float(slider_weight_shift.val)
        state["logit_mean"] = float(slider_logit_mean.val)
        state["logit_std"] = float(slider_logit_std.val)
        state["mode_scale"] = float(slider_mode_scale.val)
        state["min_timestep"] = int(slider_min.val)
        state["max_timestep"] = int(slider_max.val)
        state["width"] = int(slider_width.val)
        state["height"] = int(slider_height.val)
        if state["max_timestep"] <= state["min_timestep"]:
            state["max_timestep"] = state["min_timestep"] + 1
        render()

    radio_sample_mode.on_clicked(update)
    radio_weight_mode.on_clicked(update)
    radio_hist_mode.on_clicked(update)
    slider_sample_scale.on_changed(update)
    slider_sample_shift.on_changed(update)
    slider_weight_scale.on_changed(update)
    slider_weight_shift.on_changed(update)
    slider_logit_mean.on_changed(update)
    slider_logit_std.on_changed(update)
    slider_mode_scale.on_changed(update)
    slider_min.on_changed(update)
    slider_max.on_changed(update)
    slider_width.on_changed(update)
    slider_height.on_changed(update)

    update()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Preview timestep sampling and weighting curves for SDXL or Anima.")
    parser.add_argument("--output", type=Path, default=None, help="Optional PNG path to save the latest preview frame.")
    parser.add_argument("--preview_mode", choices=["sdxl", "anima"], default="sdxl", help="Preview formula set to use.")
    parser.add_argument("--mode", default="uniform", help="Initial timestep sampling mode.")
    parser.add_argument("--scale", type=float, default=1.0, help="Initial sampling sigmoid scale.")
    parser.add_argument("--shift", type=float, default=1.0, help="Initial sampling shift factor.")
    parser.add_argument("--weight_mode", default="none", help="Initial timestep loss weighting mode.")
    parser.add_argument("--weight_scale", type=float, default=1.0, help="Initial timestep loss weighting sigmoid scale.")
    parser.add_argument("--weight_shift", type=float, default=1.0, help="Initial timestep loss weighting shift factor.")
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Initial logit-normal mean for Anima sigma weighting.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Initial logit-normal std for Anima sigma weighting.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Initial mode weighting scale for Anima sigma weighting.")
    parser.add_argument("--min_timestep", type=int, default=0, help="Initial minimum timestep.")
    parser.add_argument("--max_timestep", type=int, default=1000, help="Initial maximum timestep.")
    parser.add_argument("--width", type=int, default=1024, help="Preview width for resolution-dependent modes.")
    parser.add_argument("--height", type=int, default=1024, help="Preview height for resolution-dependent modes.")
    parser.add_argument("--samples", type=int, default=40000, help="Initial histogram sample count.")
    parser.add_argument("--seed", type=int, default=1337, help="Initial random seed.")
    parser.add_argument("--save-only", action="store_true", help="Render a single PNG snapshot and exit without opening the interactive window.")
    args = parser.parse_args()
    if args.save_only:
        if args.output is None:
            raise SystemExit("--save-only requires --output")
        save_preview_snapshot(
            args.output,
            preview_mode=args.preview_mode,
            mode=args.mode,
            scale=args.scale,
            shift=args.shift,
            weight_mode=args.weight_mode,
            weight_scale=args.weight_scale,
            weight_shift=args.weight_shift,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
            min_timestep=args.min_timestep,
            max_timestep=args.max_timestep,
            width=args.width,
            height=args.height,
            samples=args.samples,
            seed=args.seed,
        )
        return
    render_interactive(
        args.output,
        preview_mode=args.preview_mode,
        mode=args.mode,
        scale=args.scale,
        shift=args.shift,
        weight_mode=args.weight_mode,
        weight_scale=args.weight_scale,
        weight_shift=args.weight_shift,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
        min_timestep=args.min_timestep,
        max_timestep=args.max_timestep,
        width=args.width,
        height=args.height,
        samples=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
