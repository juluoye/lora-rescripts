import os
import tempfile
import sys
import types
import importlib.machinery
import numpy as np

import torch

cv2_stub = sys.modules.setdefault("cv2", types.ModuleType("cv2"))
if getattr(cv2_stub, "__spec__", None) is None:
    cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)

from library import strategy_base as strategy_base_module
from library.strategy_sdxl import SdxlTextEncodingStrategy
from library.strategy_sdxl import SdxlTextEncoderOutputsCachingStrategy
from library.train_dataset_util import save_text_encoder_outputs_to_disk


def test_encode_tokens_with_weights_preserves_hidden_state_mean():
    strategy = SdxlTextEncodingStrategy()

    hidden_states1 = torch.ones((1, 77, 4), dtype=torch.float32)
    hidden_states2 = torch.full((1, 77, 6), 2.0, dtype=torch.float32)
    pool2 = torch.zeros((1, 6), dtype=torch.float32)

    strategy.encode_tokens = lambda tokenize_strategy, models, tokens_list: [  # type: ignore[method-assign]
        hidden_states1.clone(),
        hidden_states2.clone(),
        pool2.clone(),
    ]

    weights1 = torch.ones((1, 1, 77), dtype=torch.float32)
    weights2 = torch.ones((1, 1, 77), dtype=torch.float32)
    weights1[:, :, 10:20] = 1.5
    weights2[:, :, 30:40] = 0.5

    out_hidden_states1, out_hidden_states2, out_pool2 = strategy.encode_tokens_with_weights(
        tokenize_strategy=None,
        models=[],
        tokens_list=[],
        weights_list=[weights1, weights2],
    )

    assert torch.allclose(out_hidden_states1.float().mean(), hidden_states1.float().mean(), atol=1e-5)
    assert torch.allclose(out_hidden_states2.float().mean(), hidden_states2.float().mean(), atol=1e-5)
    assert torch.equal(out_pool2, pool2)


def test_sdxl_text_cache_safetensors_bf16_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "sample.png")
        with open(image_path, "wb") as handle:
            handle.write(b"stub")

        strategy = SdxlTextEncoderOutputsCachingStrategy(
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=False,
        )
        strategy._disk_cache_format = "safetensors"
        strategy._disk_cache_dtype = "bf16"

        class MockInfo:
            def __init__(self, absolute_path: str):
                self.absolute_path = absolute_path
                self.caption = "test caption"
                self.text_encoder_outputs_npz = strategy.get_outputs_npz_path(absolute_path)
                self.text_encoder_outputs_cache_root = tmpdir
                self.text_encoder_outputs_disk_cache_ref = None

        info = MockInfo(image_path)

        strategy.queue_safetensors_payload(
            info,
            {
                "values": {
                    "hidden_state1": torch.randn(77, 4, dtype=torch.float32),
                    "hidden_state2": torch.randn(77, 6, dtype=torch.float32),
                    "pool2": torch.randn(6, dtype=torch.float32),
                },
                "metadata": {},
            },
        )
        strategy.finalize_caching()

        assert info.text_encoder_outputs_disk_cache_ref is not None
        loaded_hidden_state1, loaded_hidden_state2, loaded_pool2 = strategy.load_outputs_npz(info.text_encoder_outputs_disk_cache_ref)

        assert isinstance(loaded_hidden_state1, torch.Tensor)
        assert isinstance(loaded_hidden_state2, torch.Tensor)
        assert isinstance(loaded_pool2, torch.Tensor)
        assert loaded_hidden_state1.dtype == torch.bfloat16
        assert loaded_hidden_state2.dtype == torch.bfloat16
        assert loaded_pool2.dtype == torch.bfloat16


def test_sdxl_text_cache_catalog_index_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "sample.png")
        with open(image_path, "wb") as handle:
            handle.write(b"stub")

        strategy = SdxlTextEncoderOutputsCachingStrategy(
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=False,
        )
        strategy._disk_cache_format = "safetensors"
        strategy._disk_cache_dtype = "fp16"

        class MockInfo:
            def __init__(self, absolute_path: str):
                self.absolute_path = absolute_path
                self.caption = "test caption"
                self.text_encoder_outputs_npz = strategy.get_outputs_npz_path(absolute_path)
                self.text_encoder_outputs_cache_root = tmpdir
                self.text_encoder_outputs_disk_cache_ref = None

        info = MockInfo(image_path)
        strategy.queue_safetensors_payload(
            info,
            {
                "values": {
                    "hidden_state1": torch.randn(77, 4, dtype=torch.float32),
                    "hidden_state2": torch.randn(77, 6, dtype=torch.float32),
                    "pool2": torch.randn(6, dtype=torch.float32),
                },
                "metadata": {},
            },
        )
        strategy.finalize_caching()

        cache_dir = strategy._get_safetensors_cache_dir(tmpdir)
        index_path = os.path.join(cache_dir, "_catalog.json")
        assert os.path.exists(index_path)

        reloaded = SdxlTextEncoderOutputsCachingStrategy(
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=False,
        )
        reloaded._disk_cache_format = "safetensors"
        reloaded._disk_cache_dtype = "fp16"

        info2 = MockInfo(image_path)
        assert reloaded.is_disk_cached_outputs_expected_for_info(info2.text_encoder_outputs_npz, info=info2) is True
        assert info2.text_encoder_outputs_disk_cache_ref is not None
        loaded_hidden_state1, loaded_hidden_state2, loaded_pool2 = reloaded.load_outputs_npz(info2.text_encoder_outputs_disk_cache_ref)
        assert isinstance(loaded_hidden_state1, np.ndarray)
        assert isinstance(loaded_hidden_state2, np.ndarray)
        assert isinstance(loaded_pool2, np.ndarray)
        assert loaded_hidden_state1.dtype == np.float16
        assert loaded_hidden_state2.dtype == np.float16
        assert loaded_pool2.dtype == np.float16


def test_legacy_text_cache_npz_write_respects_strategy_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "legacy_te_outputs.npz")
        strategy = SdxlTextEncoderOutputsCachingStrategy(
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=False,
        )
        strategy._disk_cache_format = "npz"
        strategy._disk_cache_dtype = "fp16"

        previous_strategy = strategy_base_module.TextEncoderOutputsCachingStrategy.get_strategy()
        strategy_base_module.TextEncoderOutputsCachingStrategy._strategy = strategy
        try:
            save_text_encoder_outputs_to_disk(
                npz_path,
                torch.randn(77, 4, dtype=torch.float32),
                torch.randn(77, 6, dtype=torch.float32),
                torch.randn(6, dtype=torch.float32),
            )
        finally:
            strategy_base_module.TextEncoderOutputsCachingStrategy._strategy = previous_strategy

        with np.load(npz_path, allow_pickle=False) as archive:
            assert archive["hidden_state1"].dtype == np.float16
            assert archive["hidden_state2"].dtype == np.float16
            assert archive["pool2"].dtype == np.float16
