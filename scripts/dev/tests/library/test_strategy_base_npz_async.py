import os
import sys
import tempfile
import types
from importlib.machinery import ModuleSpec
from unittest.mock import patch

import numpy as np

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__spec__ = ModuleSpec("cv2", loader=None)
    sys.modules["cv2"] = cv2_stub

from library.strategy_base import LatentsCachingStrategy, configure_latents_cache_runtime


class DummyLatentsCachingStrategy(LatentsCachingStrategy):
    @property
    def cache_suffix(self):
        return "_dummy.npz"

    def get_latents_npz_path(self, absolute_path, image_size):
        stem = os.path.splitext(os.path.basename(absolute_path))[0]
        return os.path.join(os.path.dirname(absolute_path), f"{stem}_dummy.npz")

    def is_disk_cached_latents_expected(self, bucket_reso, npz_path, flip_aug, alpha_mask):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask)

    def cache_batch_latents(self, model, batch, flip_aug, alpha_mask, random_crop):
        raise NotImplementedError


def test_latents_npz_async_write_flushes_to_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_a = os.path.join(tmpdir, "a.png")
        image_b = os.path.join(tmpdir, "b.png")
        npz_a = os.path.join(tmpdir, "a_dummy.npz")
        npz_b = os.path.join(tmpdir, "b_dummy.npz")

        with patch.dict(os.environ, {"MIKAZUKI_LATENTS_NPZ_WRITE_WORKERS": "2"}):
            strategy = DummyLatentsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

            assert strategy._npz_write_executor is not None

            strategy._queue_latents_npz_write(
                npz_a,
                np.full((4, 8, 8), 1.5, dtype=np.float32),
                (64, 64),
                (0, 0, 64, 64),
            )
            strategy._queue_latents_npz_write(
                npz_b,
                np.full((4, 8, 8), 2.5, dtype=np.float32),
                (64, 64),
                (0, 0, 64, 64),
            )

            strategy.flush_pending_disk_cache()

            assert os.path.exists(npz_a)
            assert os.path.exists(npz_b)

            with np.load(npz_a, allow_pickle=False) as data_a:
                np.testing.assert_allclose(data_a["latents"], np.full((4, 8, 8), 1.5, dtype=np.float32))
                np.testing.assert_array_equal(data_a["original_size"], np.array((64, 64)))
                np.testing.assert_array_equal(data_a["crop_ltrb"], np.array((0, 0, 64, 64)))

            with np.load(npz_b, allow_pickle=False) as data_b:
                np.testing.assert_allclose(data_b["latents"], np.full((4, 8, 8), 2.5, dtype=np.float32))

            strategy.finalize_caching()
            assert strategy._npz_write_executor is None
            assert strategy._pending_npz_writes == []


def test_latents_npz_write_worker_count_defaults_and_is_configurable():
    with patch.dict(os.environ, {}, clear=False):
        configure_latents_cache_runtime(npz_write_workers=None)
        default_strategy = DummyLatentsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)
        try:
            assert default_strategy._npz_write_workers == 2
        finally:
            default_strategy.finalize_caching()

        configure_latents_cache_runtime(npz_write_workers=5)
        configured_strategy = DummyLatentsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)
        try:
            assert configured_strategy._npz_write_workers == 5
        finally:
            configured_strategy.finalize_caching()

        configure_latents_cache_runtime(npz_write_workers=None)
