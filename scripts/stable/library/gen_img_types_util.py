from __future__ import annotations

import random
from typing import Any, List, NamedTuple, Optional, Tuple


class BatchDataBase(NamedTuple):
    step: int
    prompt: str
    negative_prompt: str
    seed: int
    init_image: Any
    mask_image: Any
    clip_prompt: str
    guide_image: Any
    raw_prompt: str
    file_name: Optional[str]


class BatchDataExt(NamedTuple):
    width: int
    height: int
    original_width: int
    original_height: int
    original_width_negative: int
    original_height_negative: int
    crop_left: int
    crop_top: int
    steps: int
    scale: float
    negative_scale: float
    strength: float
    network_muls: Tuple[float]
    num_sub_prompts: int


class BatchData(NamedTuple):
    return_latents: bool
    base: BatchDataBase
    ext: BatchDataExt


class ListPrompter:
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
        self.index = 0

    def shuffle(self):
        random.shuffle(self.prompts)

    def __len__(self):
        return len(self.prompts)

    def __call__(self, *args, **kwargs):
        if self.index >= len(self.prompts):
            self.index = 0
            return None

        prompt = self.prompts[self.index]
        self.index += 1
        return prompt


__all__ = ["BatchData", "BatchDataBase", "BatchDataExt", "ListPrompter"]
