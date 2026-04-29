from __future__ import annotations

import logging
import re
from typing import List, Optional, Union

import torch
import library.train_util as train_util
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]| 
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    r"""
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    text = text.replace("BREAK", "\\BREAK\\")

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1] and res[i][0].strip() != "BREAK" and res[i + 1][0].strip() != "BREAK":
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(tokenizer: CLIPTokenizer, token_replacer, prompt: List[str], max_length: int):
    tokens = []
    weights = []
    truncated = False

    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            if word.strip() == "BREAK":
                pad_len = tokenizer.model_max_length - (len(text_token) % tokenizer.model_max_length)
                logger.info(f"BREAK pad_len: {pad_len}")
                for i in range(pad_len):
                    text_token.append(tokenizer.pad_token_id)
                    text_weight.append(1.0)
                continue

            token = tokenizer(word).input_ids[1:-1]
            token = token_replacer(token)

            text_token += token
            text_weight += [weight] * len(token)
            if len(text_token) > max_length:
                truncated = True
                break
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("warning: Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    is_sdxl: bool,
    text_encoder: CLIPTextModel,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,
):
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        pool = None
        for i in range(max_embeddings_multiples):
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:
                text_input_chunk[:, -1] = text_input[0, -1]
            else:
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:
                        text_input_chunk[j, 1] = eos

            enc_out = text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
            text_embedding = enc_out["hidden_states"][-clip_skip]
            if not is_sdxl:
                text_embedding = text_encoder.text_model.final_layer_norm(text_embedding)
            if pool is None:
                pool = enc_out.get("text_embeds", None)
                if pool is not None:
                    pool = train_util.pool_workaround(text_encoder, enc_out["last_hidden_state"], text_input_chunk, eos)

            if no_boseos_middle:
                if i == 0:
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    text_embedding = text_embedding[:, 1:]
                else:
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        enc_out = text_encoder(text_input, output_hidden_states=True, return_dict=True)
        text_embeddings = enc_out["hidden_states"][-clip_skip]
        if not is_sdxl:
            text_embeddings = text_encoder.text_model.final_layer_norm(text_embeddings)
        pool = enc_out.get("text_embeds", None)
        if pool is not None:
            pool = train_util.pool_workaround(text_encoder, enc_out["last_hidden_state"], text_input, eos)
    return text_embeddings, pool


def get_weighted_text_embeddings(
    is_sdxl: bool,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 1,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    clip_skip: int = 1,
    token_replacer=None,
    device=None,
    emb_normalize_mode: Optional[str] = "original",
    **kwargs,
):
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    new_prompts = []
    for p in prompt:
        new_prompts.extend(p.split(" AND "))
    prompt = new_prompts

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(tokenizer, token_replacer, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(tokenizer, token_replacer, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [token[1:-1] for token in tokenizer(prompt, max_length=max_length, truncation=True).input_ids]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [token[1:-1] for token in tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

    text_embeddings, text_pool = get_unweighted_text_embeddings(
        is_sdxl,
        text_encoder,
        prompt_tokens,
        tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )

    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=device)
    if uncond_prompt is not None:
        uncond_embeddings, uncond_pool = get_unweighted_text_embeddings(
            is_sdxl,
            text_encoder,
            uncond_tokens,
            tokenizer.model_max_length,
            clip_skip,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=device)

    if (not skip_parsing) and (not skip_weighting):
        if emb_normalize_mode == "abs":
            previous_mean = text_embeddings.float().abs().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= prompt_weights.unsqueeze(-1)
            current_mean = text_embeddings.float().abs().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
            if uncond_prompt is not None:
                previous_mean = uncond_embeddings.float().abs().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= uncond_weights.unsqueeze(-1)
                current_mean = uncond_embeddings.float().abs().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        elif emb_normalize_mode == "none":
            text_embeddings *= prompt_weights.unsqueeze(-1)
            if uncond_prompt is not None:
                uncond_embeddings *= uncond_weights.unsqueeze(-1)
        else:
            previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= prompt_weights.unsqueeze(-1)
            current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
            if uncond_prompt is not None:
                previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= uncond_weights.unsqueeze(-1)
                current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, text_pool, uncond_embeddings, uncond_pool, prompt_tokens
    return text_embeddings, text_pool, None, None, prompt_tokens


__all__ = [
    "get_prompts_with_weights",
    "get_unweighted_text_embeddings",
    "get_weighted_text_embeddings",
    "pad_tokens_and_weights",
    "parse_prompt_attention",
]
