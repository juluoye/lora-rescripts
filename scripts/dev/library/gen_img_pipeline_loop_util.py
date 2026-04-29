from __future__ import annotations

from tqdm import tqdm

from library.gen_img_pipeline_gradual_latent_util import update_gradual_latent_for_step
from library.gen_img_pipeline_step_util import run_pipeline_step


def run_pipeline_denoising_loop(
    *,
    scheduler,
    gradual_latent,
    gradual_latent_state,
    timesteps,
    latents,
    callback_steps,
    callback,
    is_cancelled_callback,
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
    for step_index, timestep in enumerate(tqdm(timesteps)):
        resized_size, gradual_latent_state = update_gradual_latent_for_step(
            scheduler=scheduler,
            gradual_latent=gradual_latent,
            state=gradual_latent_state,
            timestep=timestep,
        )
        latents, each_control_net_enabled = run_pipeline_step(
            step_index=step_index,
            total_steps=len(timesteps),
            timestep=timestep,
            latents=latents,
            scheduler=scheduler,
            num_latent_input=num_latent_input,
            control_net_lllites=control_net_lllites,
            control_nets=control_nets,
            each_control_net_enabled=each_control_net_enabled,
            text_embeddings=text_embeddings,
            batch_size=batch_size,
            regional_network=regional_network,
            unet=unet,
            is_sdxl=is_sdxl,
            vector_embeddings=vector_embeddings,
            clip_guide_images=clip_guide_images,
            guided_hints=guided_hints,
            control_net_enabled=control_net_enabled,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_scale=negative_scale,
            guidance_scale=guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
            mask=mask,
            init_latents_orig=init_latents_orig,
            img2img_noise=img2img_noise,
        )

        if step_index % callback_steps == 0:
            if callback is not None:
                callback(step_index, timestep, latents)
            if is_cancelled_callback is not None and is_cancelled_callback():
                return None, each_control_net_enabled, gradual_latent_state

    return latents, each_control_net_enabled, gradual_latent_state


__all__ = ["run_pipeline_denoising_loop"]
