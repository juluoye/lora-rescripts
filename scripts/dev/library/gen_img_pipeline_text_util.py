from __future__ import annotations

import logging

import torch

from library.gen_img_text_embedding_util import get_weighted_text_embeddings

logger = logging.getLogger(__name__)


def prepare_pipeline_text_embeddings(
    *,
    is_sdxl,
    tokenizers,
    text_encoders,
    get_token_replacer_fn,
    prompt,
    negative_prompt,
    batch_size,
    do_classifier_free_guidance,
    negative_scale,
    max_embeddings_multiples,
    clip_skip,
    device,
    emb_normalize_mode,
    extra_kwargs,
):
    if negative_prompt is None:
        negative_prompt = [""] * batch_size
    elif isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * batch_size
    if batch_size != len(negative_prompt):
        raise ValueError(
            f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
            f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
            " the batch size of `prompt`."
        )

    tes_text_embs = []
    tes_uncond_embs = []
    tes_real_uncond_embs = []
    text_pool = None
    uncond_pool = None

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        token_replacer = get_token_replacer_fn(tokenizer)

        text_embeddings, text_pool, uncond_embeddings, uncond_pool, _ = get_weighted_text_embeddings(
            is_sdxl,
            tokenizer,
            text_encoder,
            prompt=prompt,
            uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
            max_embeddings_multiples=max_embeddings_multiples,
            clip_skip=clip_skip,
            token_replacer=token_replacer,
            device=device,
            emb_normalize_mode=emb_normalize_mode,
            **extra_kwargs,
        )
        tes_text_embs.append(text_embeddings)
        tes_uncond_embs.append(uncond_embeddings)

        if negative_scale is not None:
            _, _, real_uncond_embeddings, _, _ = get_weighted_text_embeddings(
                is_sdxl,
                tokenizer,
                text_encoder,
                prompt=prompt,
                uncond_prompt=[""] * batch_size,
                max_embeddings_multiples=max_embeddings_multiples,
                clip_skip=clip_skip,
                token_replacer=token_replacer,
                device=device,
                emb_normalize_mode=emb_normalize_mode,
                **extra_kwargs,
            )
            tes_real_uncond_embs.append(real_uncond_embeddings)

    text_embeddings = tes_text_embs[0]
    uncond_embeddings = tes_uncond_embs[0]
    for i in range(1, len(tes_text_embs)):
        text_embeddings = torch.cat([text_embeddings, tes_text_embs[i]], dim=2)
        if do_classifier_free_guidance:
            uncond_embeddings = torch.cat([uncond_embeddings, tes_uncond_embs[i]], dim=2)

    if do_classifier_free_guidance:
        if negative_scale is None:
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        else:
            real_uncond_embeddings = tes_real_uncond_embs[0]
            for i in range(1, len(tes_real_uncond_embs)):
                real_uncond_embeddings = torch.cat([real_uncond_embeddings, tes_real_uncond_embs[i]], dim=2)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings, real_uncond_embeddings])

    return text_embeddings, text_pool, uncond_pool


__all__ = ["prepare_pipeline_text_embeddings"]
