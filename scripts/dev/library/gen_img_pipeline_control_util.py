from __future__ import annotations

import inspect
import torch

import tools.original_control_net as original_control_net


def prepare_scheduler_step_kwargs(scheduler, eta):
    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta
    return extra_step_kwargs


def prepare_control_conditions(
    *,
    control_nets,
    control_net_lllites,
    control_net_enabled,
    is_sdxl,
    clip_guide_images,
    num_latent_input,
    batch_size,
):
    guided_hints = None
    each_control_net_enabled = None

    if control_nets:
        if not is_sdxl:
            guided_hints = original_control_net.get_guided_hints(
                control_nets, num_latent_input, batch_size, clip_guide_images
            )
        else:
            clip_guide_images = clip_guide_images * 0.5 + 0.5
        each_control_net_enabled = [control_net_enabled] * len(control_nets)

    if control_net_lllites:
        if control_net_enabled:
            for control_net, _ in control_net_lllites:
                with torch.no_grad():
                    control_net.set_cond_image(clip_guide_images)
        else:
            for control_net, _ in control_net_lllites:
                control_net.set_cond_image(None)

        each_control_net_enabled = [control_net_enabled] * len(control_net_lllites)

    return extra_control_state(
        guided_hints=guided_hints,
        each_control_net_enabled=each_control_net_enabled,
        clip_guide_images=clip_guide_images,
    )


class extra_control_state:
    def __init__(self, *, guided_hints, each_control_net_enabled, clip_guide_images):
        self.guided_hints = guided_hints
        self.each_control_net_enabled = each_control_net_enabled
        self.clip_guide_images = clip_guide_images


__all__ = ["prepare_control_conditions", "prepare_scheduler_step_kwargs"]
