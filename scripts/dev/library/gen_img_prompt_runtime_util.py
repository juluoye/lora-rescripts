from __future__ import annotations

import re
from typing import NamedTuple


class ExpandedPromptBatch(NamedTuple):
    raw_prompts: list[str]
    seeds: list[int] | None


def next_raw_prompt(*, args, prompter, pipe, seed_random, iter_seed, prompt_index: int, global_step: int, logger):
    if args.interactive:
        valid = False
        raw_prompt = None
        while not valid:
            logger.info("\nType prompt:")
            try:
                raw_prompt = input()
            except EOFError:
                break

            valid = len(raw_prompt.strip().split(" --")[0].strip()) > 0

        return raw_prompt if valid else None

    return prompter(args, pipe, seed_random, iter_seed, prompt_index, global_step)


def expand_prompt_variants(raw_prompt, *, images_per_prompt: int, seed_random, logger, handle_dynamic_prompt_variants_fn):
    seeds = None
    match = re.search(r" --d ([\d+,]+)", raw_prompt, re.IGNORECASE)
    if match:
        seeds = [int(seed) for seed in match[0][5:].split(",")]
        logger.info(f"seeds: {seeds}")
        raw_prompt = raw_prompt[: match.start()] + raw_prompt[match.end() :]

    raw_prompts, prompt_seeds = handle_dynamic_prompt_variants_fn(
        raw_prompt,
        images_per_prompt,
        seed_random,
        seeds,
        logger_override=logger,
    )
    if prompt_seeds is not None:
        seeds = prompt_seeds

    return ExpandedPromptBatch(raw_prompts, seeds)


__all__ = ["ExpandedPromptBatch", "expand_prompt_variants", "next_raw_prompt"]
