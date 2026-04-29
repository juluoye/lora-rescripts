from __future__ import annotations

import logging

import torch

import tools.original_control_net as original_control_net

logger = logging.getLogger(__name__)


def run_pipeline_step(
    *,
    step_index,
    total_steps,
    timestep,
    latents,
    scheduler,
    num_latent_input,
    control_net_lllites,
    control_nets,
    each_control_net_enabled,
    text_embeddings,
    batch_size,
    regional_network,
    unet,
    is_sdxl,
    vector_embeddings,
    clip_guide_images,
    guided_hints,
    control_net_enabled,
    do_classifier_free_guidance,
    negative_scale,
    guidance_scale,
    extra_step_kwargs,
    mask,
    init_latents_orig,
    img2img_noise,
):
    latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    if control_net_lllites and each_control_net_enabled is not None:
        for j, ((control_net, ratio), enabled) in enumerate(zip(control_net_lllites, each_control_net_enabled)):
            if not enabled or ratio >= 1.0:
                continue
            if ratio < step_index / total_steps:
                logger.info(f"ControlNetLLLite {j} is disabled (ratio={ratio} at {step_index} / {total_steps})")
                control_net.set_cond_image(None)
                each_control_net_enabled[j] = False
    if control_nets and is_sdxl and each_control_net_enabled is not None:
        for j, ((control_net, ratio), enabled) in enumerate(zip(control_nets, each_control_net_enabled)):
            if not enabled or ratio >= 1.0:
                continue
            if ratio < step_index / total_steps:
                logger.info(f"ControlNet {j} is disabled (ratio={ratio} at {step_index} / {total_steps})")
                each_control_net_enabled[j] = False

    if control_nets and control_net_enabled and not is_sdxl:
        if regional_network:
            num_sub_and_neg_prompts = len(text_embeddings) // batch_size
            text_emb_last = text_embeddings[num_sub_and_neg_prompts - 2 :: num_sub_and_neg_prompts]
        else:
            text_emb_last = text_embeddings

        noise_pred = original_control_net.call_unet_and_control_net(
            step_index,
            num_latent_input,
            unet,
            control_nets,
            guided_hints,
            step_index / total_steps,
            latent_model_input,
            timestep,
            text_embeddings,
            text_emb_last,
        ).sample
    elif control_nets:
        input_resi_add_list = []
        mid_add_list = []
        for (control_net, _), enabled in zip(control_nets, each_control_net_enabled):
            if not enabled:
                continue
            input_resi_add, mid_add = control_net(
                latent_model_input, timestep, text_embeddings, vector_embeddings, clip_guide_images
            )
            input_resi_add_list.append(input_resi_add)
            mid_add_list.append(mid_add)

        if len(input_resi_add_list) == 0:
            noise_pred = unet(latent_model_input, timestep, text_embeddings, vector_embeddings)
        else:
            if len(input_resi_add_list) > 1:
                input_resi_add_mean = []
                for k in range(len(input_resi_add_list[0])):
                    input_resi_add_mean.append(
                        torch.mean(torch.stack([input_resi_add_list[j][k] for j in range(len(input_resi_add_list))], dim=0))
                    )
                input_resi_add = input_resi_add_mean
                mid_add = torch.mean(torch.stack(mid_add_list), dim=0)

            noise_pred = unet(latent_model_input, timestep, text_embeddings, vector_embeddings, input_resi_add, mid_add)
    elif is_sdxl:
        noise_pred = unet(latent_model_input, timestep, text_embeddings, vector_embeddings)
    else:
        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

    if do_classifier_free_guidance:
        if negative_scale is None:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred_negative, noise_pred_text, noise_pred_uncond = noise_pred.chunk(num_latent_input)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_uncond)
                - negative_scale * (noise_pred_negative - noise_pred_uncond)
            )

    latents = scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs).prev_sample

    if mask is not None:
        init_latents_proper = scheduler.add_noise(init_latents_orig, img2img_noise, torch.tensor([timestep]))
        latents = (init_latents_proper * mask) + (latents * (1 - mask))

    return latents, each_control_net_enabled


__all__ = ["run_pipeline_step"]
