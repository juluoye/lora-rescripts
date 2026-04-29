from __future__ import annotations

from typing import List

import library.gen_img_dynamic_prompt_util as gen_img_dynamic_prompt_util
import library.gen_img_process_batch_util as gen_img_process_batch_util
import library.gen_img_prompt_iteration_util as gen_img_prompt_iteration_util
import library.gen_img_prompt_runtime_util as gen_img_prompt_runtime_util


def run_generation_iterations(
    *,
    args,
    prompter,
    seed_random,
    highres_fix,
    upscaler,
    batch_data_cls,
    batch_data_base_cls,
    batch_data_ext_cls,
    pipe,
    vae,
    dtype,
    device,
    scheduler_num_noises_per_step,
    noise_manager,
    control_nets,
    control_net_lllites,
    networks,
    network_default_muls,
    regional_network,
    unet,
    network_pre_calc,
    max_embeddings_multiples,
    latent_channels,
    downsampling_factor,
    is_sdxl,
    logger,
    pyramid_noise_like_fn,
    init_images,
    mask_images,
    guide_images,
):
    def process_batch(batch: List[batch_data_cls], highres_fix, highres_1st=False):
        return gen_img_process_batch_util.process_generation_batch(
            batch,
            highres_fix=highres_fix,
            highres_1st=highres_1st,
            args=args,
            upscaler=upscaler,
            batch_data_cls=batch_data_cls,
            batch_data_base_cls=batch_data_base_cls,
            batch_data_ext_cls=batch_data_ext_cls,
            pipe=pipe,
            vae=vae,
            dtype=dtype,
            device=device,
            scheduler_num_noises_per_step=scheduler_num_noises_per_step,
            noise_manager=noise_manager,
            control_nets=control_nets,
            control_net_lllites=control_net_lllites,
            networks=networks,
            network_default_muls=network_default_muls,
            regional_network=regional_network,
            unet=unet,
            network_pre_calc=network_pre_calc,
            max_embeddings_multiples=max_embeddings_multiples,
            latent_channels=latent_channels,
            downsampling_factor=downsampling_factor,
            is_sdxl=is_sdxl,
            logger=logger,
            pyramid_noise_like_fn=pyramid_noise_like_fn,
        )

    for gen_iter in range(args.n_iter):
        logger.info(f"iteration {gen_iter+1}/{args.n_iter}")
        if args.iter_same_seed:
            iter_seed = seed_random.randint(0, 2**32 - 1)
        else:
            iter_seed = None

        if args.shuffle_prompts:
            prompter.shuffle()

        prompt_index = 0
        global_step = 0
        batch_data = []
        while True:
            raw_prompt = gen_img_prompt_runtime_util.next_raw_prompt(
                args=args,
                prompter=prompter,
                pipe=pipe,
                seed_random=seed_random,
                iter_seed=iter_seed,
                prompt_index=prompt_index,
                global_step=global_step,
                logger=logger,
            )
            if raw_prompt is None:
                break

            expanded_prompt_batch = gen_img_prompt_runtime_util.expand_prompt_variants(
                raw_prompt,
                images_per_prompt=args.images_per_prompt,
                seed_random=seed_random,
                logger=logger,
                handle_dynamic_prompt_variants_fn=gen_img_dynamic_prompt_util.handle_dynamic_prompt_variants,
            )
            raw_prompts = expanded_prompt_batch.raw_prompts
            seeds = expanded_prompt_batch.seeds

            prompt_state = None
            seed = None
            for pi in range(args.images_per_prompt if len(raw_prompts) == 1 else len(raw_prompts)):
                raw_prompt = raw_prompts[pi] if len(raw_prompts) > 1 else raw_prompts[0]
                prepared_prompt_iteration = gen_img_prompt_iteration_util.prepare_prompt_iteration(
                    raw_prompt,
                    should_reparse=pi == 0 or len(raw_prompts) > 1,
                    prompt_state=prompt_state,
                    args=args,
                    prompter=prompter,
                    prompt_index=prompt_index,
                    networks=networks,
                    logger=logger,
                    unet=unet,
                    pipe=pipe,
                    seeds=seeds,
                    iter_seed=iter_seed,
                    seed_random=seed_random,
                    previous_seed=seed if pi > 0 else None,
                    global_step=global_step,
                    highres_fix=highres_fix,
                    init_images=init_images,
                    mask_images=mask_images,
                    guide_images=guide_images,
                    control_nets=control_nets,
                    control_net_lllites=control_net_lllites,
                    regional_network=regional_network,
                    batch_data_cls=batch_data_cls,
                    batch_data_base_cls=batch_data_base_cls,
                    batch_data_ext_cls=batch_data_ext_cls,
                )
                prompt_state = prepared_prompt_iteration.state
                seed = prepared_prompt_iteration.seed
                batch_entry = prepared_prompt_iteration.batch_entry
                if len(batch_data) > 0 and batch_data[-1].ext != batch_entry.ext:
                    process_batch(batch_data, highres_fix)
                    batch_data.clear()

                batch_data.append(batch_entry)
                if len(batch_data) == args.batch_size:
                    process_batch(batch_data, highres_fix)
                    batch_data.clear()

                global_step += 1

            prompt_index += 1

        if len(batch_data) > 0:
            process_batch(batch_data, highres_fix)
            batch_data.clear()


__all__ = ["run_generation_iterations"]
