from __future__ import annotations

import os

import torch

import library.model_util as model_util


def load_textual_inversion_embeddings(args, *, is_sdxl: bool, tokenizers, text_encoders, pipe, logger):
    if not args.textual_inversion_embeddings:
        return

    token_ids_embeds1 = []
    token_ids_embeds2 = []
    for embeds_file in args.textual_inversion_embeddings:
        if model_util.is_safetensors(embeds_file):
            from safetensors.torch import load_file

            data = load_file(embeds_file)
        else:
            data = torch.load(embeds_file, map_location="cpu")

        if "string_to_param" in data:
            data = data["string_to_param"]
        if is_sdxl:
            embeds1 = data["clip_l"]
            embeds2 = data["clip_g"]
        else:
            embeds1 = next(iter(data.values()))
            embeds2 = None

        num_vectors_per_token = embeds1.size()[0]
        token_string = os.path.splitext(os.path.basename(embeds_file))[0]
        token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

        num_added_tokens1 = tokenizers[0].add_tokens(token_strings)
        num_added_tokens2 = tokenizers[1].add_tokens(token_strings) if is_sdxl else 0
        assert num_added_tokens1 == num_vectors_per_token and (
            num_added_tokens2 == 0 or num_added_tokens2 == num_vectors_per_token
        ), (
            f"tokenizer has same word to token string (filename): {embeds_file}"
            + f" / 指定した名前（ファイル名）のトークンが既に存在します: {embeds_file}"
        )

        token_ids1 = tokenizers[0].convert_tokens_to_ids(token_strings)
        token_ids2 = tokenizers[1].convert_tokens_to_ids(token_strings) if is_sdxl else None
        logger.info(f"Textual Inversion embeddings `{token_string}` loaded. Tokens are added: {token_ids1} and {token_ids2}")
        assert min(token_ids1) == token_ids1[0] and token_ids1[-1] == token_ids1[0] + len(token_ids1) - 1, "token ids1 is not ordered"
        assert not is_sdxl or (
            min(token_ids2) == token_ids2[0] and token_ids2[-1] == token_ids2[0] + len(token_ids2) - 1
        ), "token ids2 is not ordered"
        assert len(tokenizers[0]) - 1 == token_ids1[-1], f"token ids 1 is not end of tokenize: {len(tokenizers[0])}"
        assert not is_sdxl or len(tokenizers[1]) - 1 == token_ids2[-1], f"token ids 2 is not end of tokenize: {len(tokenizers[1])}"

        if num_vectors_per_token > 1:
            pipe.add_token_replacement(0, token_ids1[0], token_ids1)
            if is_sdxl:
                pipe.add_token_replacement(1, token_ids2[0], token_ids2)

        token_ids_embeds1.append((token_ids1, embeds1))
        if is_sdxl:
            token_ids_embeds2.append((token_ids2, embeds2))

    text_encoders[0].resize_token_embeddings(len(tokenizers[0]))
    token_embeds1 = text_encoders[0].get_input_embeddings().weight.data
    for token_ids, embeds in token_ids_embeds1:
        for token_id, embed in zip(token_ids, embeds):
            token_embeds1[token_id] = embed

    if is_sdxl:
        text_encoders[1].resize_token_embeddings(len(tokenizers[1]))
        token_embeds2 = text_encoders[1].get_input_embeddings().weight.data
        for token_ids, embeds in token_ids_embeds2:
            for token_id, embed in zip(token_ids, embeds):
                token_embeds2[token_id] = embed


__all__ = ["load_textual_inversion_embeddings"]
