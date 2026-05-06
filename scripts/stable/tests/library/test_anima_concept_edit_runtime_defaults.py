import logging
from argparse import Namespace
from unittest.mock import patch

import torch

from library.anima_concept_edit_util import (
    AnimaConceptEditDataset,
    apply_anima_concept_edit_runtime_defaults,
)
from library.concept_edit_util import ConceptEditDataset


def test_apply_anima_concept_edit_runtime_defaults_disables_incompatible_caches():
    args = Namespace(
        model_train_type="anima-multi-addift",
        concept_edit_mode=None,
        dataset_class="",
        max_train_epochs=5,
        max_train_steps=12,
        cache_latents=True,
        cache_latents_to_disk=True,
        cache_text_encoder_outputs=True,
        cache_text_encoder_outputs_to_disk=True,
        network_train_unet_only=False,
        network_train_text_encoder_only=True,
    )

    apply_anima_concept_edit_runtime_defaults(args, logging.getLogger(__name__))

    assert args.concept_edit_mode == "multi-addift"
    assert args.dataset_class == "library.anima_concept_edit_util.AnimaConceptEditDataset"
    assert args.max_train_epochs is None
    assert args.cache_latents is False
    assert args.cache_latents_to_disk is False
    assert args.cache_text_encoder_outputs is False
    assert args.cache_text_encoder_outputs_to_disk is False
    assert args.network_train_unet_only is True
    assert args.network_train_text_encoder_only is False


def test_anima_concept_edit_dataset_expands_mask_channels_to_16():
    dataset = object.__new__(AnimaConceptEditDataset)
    sample = {
        "concept_edit_masks": torch.ones(1, 1, 8, 8),
        "loss_weights": torch.ones(1),
    }

    with patch.object(ConceptEditDataset, "__getitem__", autospec=True, return_value=dict(sample)):
        batch = dataset.__getitem__(0)

    assert batch["concept_edit_masks"].shape == (1, 16, 8, 8)
