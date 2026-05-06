# Anima Strategy Classes

import hashlib
import os
import random
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import safetensors.torch
import torch

from library import anima_utils, train_util
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy
from library import qwen_image_autoencoder_kl
from library.latents_disk_cache import (
    LatentsDiskCacheRef,
    build_safetensors_cache_dir,
    build_safetensors_shard_stem,
    build_safetensors_sidecar_path,
    load_safetensors_shard_manifest,
    safe_open_torch_cpu,
    save_safetensors_shard_manifest,
)
from library.safetensors_utils import mem_eff_save_file

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaTokenizeStrategy(TokenizeStrategy):
    """Tokenize strategy for Anima: dual tokenization with Qwen3 + T5.

    Qwen3 tokens are used for the text encoder.
    T5 tokens are used as target input IDs for the LLM Adapter (NOT encoded by T5).

    Can be initialized with either pre-loaded tokenizer objects or paths to load from.
    """

    def __init__(
        self,
        qwen3_tokenizer=None,
        t5_tokenizer=None,
        qwen3_max_length: int = 512,
        t5_max_length: int = 512,
        qwen3_path: Optional[str] = None,
        t5_tokenizer_path: Optional[str] = None,
    ) -> None:
        # Load tokenizers from paths if not provided directly
        if qwen3_tokenizer is None:
            if qwen3_path is None:
                raise ValueError("Either qwen3_tokenizer or qwen3_path must be provided")
            qwen3_tokenizer = anima_utils.load_qwen3_tokenizer(qwen3_path)
        if t5_tokenizer is None:
            t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)

        self.qwen3_tokenizer = qwen3_tokenizer
        self.qwen3_max_length = qwen3_max_length
        self.t5_tokenizer = t5_tokenizer
        self.t5_max_length = t5_max_length

    @staticmethod
    def _resolve_fallback_token_id(tokenizer) -> int:
        token_id = getattr(tokenizer, "eos_token_id", None)
        if token_id is None:
            token_id = getattr(tokenizer, "pad_token_id", None)
        if token_id is None:
            token_id = 0
        return int(token_id)

    def _ensure_min_sequence_length(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            return input_ids.long(), attention_mask.long()
        if input_ids.shape[1] > 0:
            return input_ids.long(), attention_mask.long()

        batch_size = input_ids.shape[0]
        fallback_token_id = self._resolve_fallback_token_id(tokenizer)
        input_ids = torch.full((batch_size, 1), fallback_token_id, dtype=torch.long)
        attention_mask = torch.ones((batch_size, 1), dtype=torch.long)
        return input_ids, attention_mask

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        # Tokenize with Qwen3
        qwen3_encoding = self.qwen3_tokenizer.batch_encode_plus(
            text, return_tensors="pt", truncation=True, padding=True, max_length=self.qwen3_max_length
        )
        qwen3_input_ids, qwen3_attn_mask = self._ensure_min_sequence_length(
            qwen3_encoding["input_ids"],
            qwen3_encoding["attention_mask"],
            self.qwen3_tokenizer,
        )

        # Tokenize with T5 (for LLM Adapter target tokens)
        t5_encoding = self.t5_tokenizer.batch_encode_plus(
            text, return_tensors="pt", truncation=True, padding=True, max_length=self.t5_max_length
        )
        t5_input_ids, t5_attn_mask = self._ensure_min_sequence_length(
            t5_encoding["input_ids"],
            t5_encoding["attention_mask"],
            self.t5_tokenizer,
        )
        return [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]


class AnimaTextEncodingStrategy(TextEncodingStrategy):
    """Text encoding strategy for Anima.

    Encodes Qwen3 tokens through the Qwen3 text encoder to get hidden states.
    T5 tokens are passed through unchanged (only used by LLM Adapter).
    """

    def __init__(self) -> None:
        super().__init__()

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Encode Qwen3 tokens and return embeddings + T5 token IDs.

        Args:
            models: [qwen3_text_encoder]
            tokens: [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

        Returns:
            [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
        """
        # Do not handle dropout here; handled dataset-side or in drop_cached_text_encoder_outputs()

        qwen3_text_encoder = models[0]
        qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask = tokens

        encoder_device = qwen3_text_encoder.device

        qwen3_input_ids = qwen3_input_ids.to(encoder_device, dtype=torch.long)
        qwen3_attn_mask = qwen3_attn_mask.to(encoder_device, dtype=torch.long)
        outputs = qwen3_text_encoder(input_ids=qwen3_input_ids, attention_mask=qwen3_attn_mask)
        prompt_embeds = outputs.last_hidden_state
        prompt_embeds[~qwen3_attn_mask.bool()] = 0

        return [prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

    def drop_cached_text_encoder_outputs(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        t5_input_ids: torch.Tensor,
        t5_attn_mask: torch.Tensor,
        caption_dropout_rates: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Apply dropout to cached text encoder outputs.

        Called during training when using cached outputs.
        Replaces dropped items with pre-cached unconditional embeddings (from encoding "")
        to match diffusion-pipe-main behavior.
        """
        if caption_dropout_rates is None or torch.all(caption_dropout_rates == 0.0).item():
            return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]

        # Clone to avoid in-place modification of cached tensors
        prompt_embeds = prompt_embeds.clone()
        if attn_mask is not None:
            attn_mask = attn_mask.clone()
        if t5_input_ids is not None:
            t5_input_ids = t5_input_ids.clone()
        if t5_attn_mask is not None:
            t5_attn_mask = t5_attn_mask.clone()

        for i in range(prompt_embeds.shape[0]):
            if random.random() < caption_dropout_rates[i].item():
                # Use pre-cached unconditional embeddings
                prompt_embeds[i] = 0
                if attn_mask is not None:
                    attn_mask[i] = 0
                if t5_input_ids is not None:
                    t5_input_ids[i, 0] = 1  # Set to </s> token ID
                    t5_input_ids[i, 1:] = 0
                if t5_attn_mask is not None:
                    t5_attn_mask[i, 0] = 1
                    t5_attn_mask[i, 1:] = 0

        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]


class AnimaTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Caching strategy for Anima text encoder outputs.

    Caches: prompt_embeds (float), attn_mask (int), t5_input_ids (int), t5_attn_mask (int)
    """

    ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_anima_te.npz"
    ANIMA_TEXT_ENCODER_NAMESPACE = "anima-text"
    CACHE_FORMAT_VERSION = 3
    _cache_upgrade_notice_logged = False

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
        self._safetensors_catalog_cache: dict[str, dict[str, dict[str, object]]] = {}
        self._pending_entries: list[dict[str, object]] = []
        self._pending_cache_root: Optional[str] = None
        self._written_shards_by_cache_root: dict[str, set[str]] = {}
        self._sequence_by_cache_root: dict[str, int] = {}
        self._max_pending_entries = max(64, max(1, int(batch_size or 1)) * 16)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def _resolve_cache_root(self, image_abs_path: str) -> str:
        return os.path.abspath(os.path.dirname(image_abs_path))

    def _resolve_cache_root_for_info(self, info) -> str:
        explicit_root = str(getattr(info, "text_encoder_outputs_cache_root", "") or "").strip()
        if explicit_root:
            return os.path.abspath(explicit_root)
        return self._resolve_cache_root(info.absolute_path)

    def _build_cache_key(self, info) -> str:
        cache_root = self._resolve_cache_root_for_info(info)
        relative_path = os.path.relpath(os.path.abspath(info.absolute_path), cache_root).replace("\\", "/")
        caption_hash = hashlib.sha1(str(info.caption or "").encode("utf-8", errors="ignore")).hexdigest()[:16]
        dropout = float(getattr(info, "caption_dropout_rate", 0.0) or 0.0)
        return f"{relative_path}#caption={caption_hash}#dropout={dropout:.6f}"

    def _get_cache_dir(self, cache_root: str) -> str:
        return build_safetensors_cache_dir(cache_root, self.ANIMA_TEXT_ENCODER_NAMESPACE)

    def _load_catalog(self, cache_root: str) -> dict[str, dict[str, object]]:
        normalized_cache_root = os.path.abspath(cache_root)
        cached = self._safetensors_catalog_cache.get(normalized_cache_root)
        if cached is not None:
            return cached

        cache_dir = self._get_cache_dir(normalized_cache_root)
        catalog: dict[str, dict[str, object]] = {}
        if os.path.isdir(cache_dir):
            for entry in os.scandir(cache_dir):
                if not entry.is_file() or not entry.name.lower().endswith(".json"):
                    continue
                try:
                    manifest = load_safetensors_shard_manifest(entry.path)
                except Exception as ex:
                    logger.warning(f"failed to load Anima text-cache manifest {entry.path}: {ex}")
                    continue

                if str(manifest.get("namespace") or "") != self.ANIMA_TEXT_ENCODER_NAMESPACE:
                    continue

                shard_file = str(manifest.get("shard_file") or "")
                if not shard_file:
                    continue
                shard_path = os.path.abspath(os.path.join(cache_dir, shard_file))
                if not os.path.exists(shard_path):
                    continue

                for manifest_entry in manifest.get("entries") or []:
                    cache_key = str(manifest_entry.get("cache_key") or "").strip()
                    entry_key = str(manifest_entry.get("entry_key") or "").strip()
                    if not cache_key or not entry_key:
                        continue
                    catalog[cache_key] = {
                        "path": shard_path,
                        "entry_key": entry_key,
                        "source_mtime_ns": int(manifest_entry.get("source_mtime_ns", 0) or 0),
                        "source_size": int(manifest_entry.get("source_size", 0) or 0),
                    }

        self._safetensors_catalog_cache[normalized_cache_root] = catalog
        return catalog

    def _register_written_shard(self, cache_root: str, shard_path: str) -> None:
        normalized_cache_root = os.path.abspath(cache_root)
        written = self._written_shards_by_cache_root.setdefault(normalized_cache_root, set())
        written.add(os.path.abspath(shard_path))

    def _prune_stale_shards(self, cache_root: str) -> None:
        normalized_cache_root = os.path.abspath(cache_root)
        cache_dir = self._get_cache_dir(normalized_cache_root)
        if not os.path.isdir(cache_dir):
            return

        catalog = self._load_catalog(normalized_cache_root)
        live_shards = {
            os.path.abspath(str(entry.get("path") or ""))
            for entry in catalog.values()
            if str(entry.get("path") or "").strip()
        }
        written = self._written_shards_by_cache_root.get(normalized_cache_root)
        if written:
            live_shards.update(written)

        for entry in os.scandir(cache_dir):
            if not entry.is_file() or not entry.name.lower().endswith(".safetensors"):
                continue
            shard_path = os.path.abspath(entry.path)
            if shard_path in live_shards:
                continue
            sidecar_path = build_safetensors_sidecar_path(shard_path)
            try:
                os.remove(shard_path)
            except OSError as ex:
                logger.warning(f"failed to remove stale Anima text-cache shard {shard_path}: {ex}")
            try:
                if os.path.exists(sidecar_path):
                    os.remove(sidecar_path)
            except OSError as ex:
                logger.warning(f"failed to remove stale Anima text-cache sidecar {sidecar_path}: {ex}")

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = self._load_npz_archive(npz_path)
            if "prompt_embeds" not in npz:
                return False
            if "attn_mask" not in npz:
                return False
            if "t5_input_ids" not in npz:
                return False
            if "t5_attn_mask" not in npz:
                return False
            if "caption_dropout_rate" not in npz:
                return False
            if "cache_format_version" not in npz:
                self._log_cache_upgrade_notice(npz_path)
                return False
            if int(np.array(npz["cache_format_version"]).item()) != self.CACHE_FORMAT_VERSION:
                self._log_cache_upgrade_notice(npz_path)
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def is_disk_cached_outputs_expected_for_info(self, npz_path: str, info=None) -> bool:
        if not self.cache_to_disk:
            return False
        if info is not None:
            cache_root = self._resolve_cache_root_for_info(info)
            cache_key = self._build_cache_key(info)
            catalog = self._load_catalog(cache_root)
            entry = catalog.get(cache_key)
            if entry is not None:
                shard_path = str(entry.get("path") or "")
                entry_key = str(entry.get("entry_key") or "")
                if shard_path and entry_key and os.path.exists(shard_path):
                    try:
                        source_stat = os.stat(info.absolute_path)
                    except OSError:
                        source_stat = None
                    if source_stat is not None:
                        if int(entry.get("source_mtime_ns", 0) or 0) == int(source_stat.st_mtime_ns):
                            if int(entry.get("source_size", 0) or 0) == int(source_stat.st_size):
                                info.text_encoder_outputs_disk_cache_ref = LatentsDiskCacheRef(
                                    format="safetensors",
                                    path=shard_path,
                                    entry_key=entry_key,
                                )
                                return True
        return self.is_disk_cached_outputs_expected(npz_path)

    @classmethod
    def _log_cache_upgrade_notice(cls, npz_path: str) -> None:
        if cls._cache_upgrade_notice_logged:
            return
        cls._cache_upgrade_notice_logged = True
        logger.info(
            f"Detected legacy Anima text cache format at {npz_path}. The cache will be rebuilt once to enable variable-length text caching."
        )
        logger.info(
            f"检测到旧版 Anima 文本缓存格式：{npz_path}。系统会自动重建一次缓存，以启用变长文本缓存加速。"
        )

    def load_outputs_npz(self, npz_path_or_ref) -> List[np.ndarray]:
        if isinstance(npz_path_or_ref, LatentsDiskCacheRef) and npz_path_or_ref.format == "safetensors":
            entry_key = str(npz_path_or_ref.entry_key or "")
            if not entry_key:
                raise ValueError(f"Anima text-cache entry key is missing for {npz_path_or_ref.path}")
            with safe_open_torch_cpu(npz_path_or_ref.path) as handle:
                prompt_embeds = handle.get_tensor(f"prompt_embeds::{entry_key}").numpy()
                attn_mask = handle.get_tensor(f"attn_mask::{entry_key}").numpy()
                t5_input_ids = handle.get_tensor(f"t5_input_ids::{entry_key}").numpy()
                t5_attn_mask = handle.get_tensor(f"t5_attn_mask::{entry_key}").numpy()
                caption_dropout_rate = handle.get_tensor(f"caption_dropout_rate::{entry_key}").numpy().reshape(())
            return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, caption_dropout_rate]

        npz_path = str(npz_path_or_ref)
        data = self._load_npz_archive(npz_path)
        prompt_embeds = data["prompt_embeds"]
        prompt_embeds_dtype = str(np.asarray(data.get("prompt_embeds_dtype", "float32")).item()) if "prompt_embeds_dtype" in data else ""
        if prompt_embeds_dtype == "float16":
            prompt_embeds = prompt_embeds.astype(np.float16, copy=False)
        elif prompt_embeds_dtype == "bfloat16":
            prompt_embeds = prompt_embeds.astype(np.float32, copy=False)
        attn_mask = data["attn_mask"]
        t5_input_ids = data["t5_input_ids"]
        t5_attn_mask = data["t5_attn_mask"]
        caption_dropout_rate = data["caption_dropout_rate"]
        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, caption_dropout_rate]

    @staticmethod
    def _get_effective_length(mask: np.ndarray) -> int:
        if mask.ndim == 0:
            return 1

        flat_mask = np.asarray(mask).reshape(-1)
        nonzero = np.flatnonzero(flat_mask)
        if nonzero.size == 0:
            return 1 if flat_mask.shape[0] > 0 else 0
        return int(nonzero[-1]) + 1

    @classmethod
    def _trim_cached_outputs(
        cls,
        prompt_embeds: np.ndarray,
        attn_mask: np.ndarray,
        t5_input_ids: np.ndarray,
        t5_attn_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        prompt_length = min(prompt_embeds.shape[0], cls._get_effective_length(attn_mask))
        t5_length = min(t5_input_ids.shape[0], cls._get_effective_length(t5_attn_mask))

        prompt_length = max(prompt_length, 1)
        t5_length = max(t5_length, 1)

        trimmed_prompt_embeds = np.ascontiguousarray(prompt_embeds[:prompt_length])
        trimmed_attn_mask = np.ascontiguousarray(attn_mask[:prompt_length])
        trimmed_t5_input_ids = np.ascontiguousarray(t5_input_ids[:t5_length])
        trimmed_t5_attn_mask = np.ascontiguousarray(t5_attn_mask[:t5_length])
        return trimmed_prompt_embeds, trimmed_attn_mask, trimmed_t5_input_ids, trimmed_t5_attn_mask

    def _flush_pending_entries(self) -> None:
        if not self._pending_entries or self._pending_cache_root is None:
            return

        cache_root = os.path.abspath(self._pending_cache_root)
        entries = list(self._pending_entries)
        self._pending_entries = []
        self._pending_cache_root = None

        cache_dir = self._get_cache_dir(cache_root)
        os.makedirs(cache_dir, exist_ok=True)

        sequence_no = self._sequence_by_cache_root.get(cache_root, 0) + 1
        self._sequence_by_cache_root[cache_root] = sequence_no
        shard_stem = build_safetensors_shard_stem(
            (0, 0),
            flip_aug=False,
            alpha_mask=False,
            sequence_no=sequence_no,
            image_count=len(entries),
            unique_suffix=f"anima_text_{sequence_no:04d}_{os.getpid()}",
        ).replace("__bucket_0x0", "__anima_text")
        shard_path = os.path.join(cache_dir, shard_stem + ".safetensors")
        sidecar_path = build_safetensors_sidecar_path(shard_path)

        tensors: dict[str, torch.Tensor] = {}
        manifest_entries: list[dict[str, object]] = []

        for entry_index, entry in enumerate(entries):
            entry_key = f"{entry_index:08d}"
            prompt_embeds = np.asarray(entry["prompt_embeds"])
            attn_mask = np.asarray(entry["attn_mask"])
            t5_input_ids = np.asarray(entry["t5_input_ids"])
            t5_attn_mask = np.asarray(entry["t5_attn_mask"])
            caption_dropout_rate = float(entry["caption_dropout_rate"])
            info = entry["info"]
            cache_key = self._build_cache_key(info)
            try:
                source_stat = os.stat(info.absolute_path)
                source_mtime_ns = int(source_stat.st_mtime_ns)
                source_size = int(source_stat.st_size)
            except OSError:
                source_mtime_ns = 0
                source_size = 0

            tensors[f"prompt_embeds::{entry_key}"] = torch.from_numpy(prompt_embeds)
            tensors[f"attn_mask::{entry_key}"] = torch.from_numpy(attn_mask.astype(np.int32, copy=False))
            tensors[f"t5_input_ids::{entry_key}"] = torch.from_numpy(t5_input_ids.astype(np.int32, copy=False))
            tensors[f"t5_attn_mask::{entry_key}"] = torch.from_numpy(t5_attn_mask.astype(np.int32, copy=False))
            tensors[f"caption_dropout_rate::{entry_key}"] = torch.tensor(caption_dropout_rate, dtype=torch.float32)

            info.text_encoder_outputs_disk_cache_ref = LatentsDiskCacheRef(format="safetensors", path=shard_path, entry_key=entry_key)
            manifest_entries.append(
                {
                    "cache_key": cache_key,
                    "entry_key": entry_key,
                    "source_mtime_ns": source_mtime_ns,
                    "source_size": source_size,
                }
            )
            self._load_catalog(cache_root)[cache_key] = {
                "path": shard_path,
                "entry_key": entry_key,
                "source_mtime_ns": source_mtime_ns,
                "source_size": source_size,
            }

        mem_eff_save_file(
            tensors,
            shard_path,
            metadata={
                "format": "mikazuki_anima_text_safetensors_shard",
                "format_version": "1",
                "namespace": self.ANIMA_TEXT_ENCODER_NAMESPACE,
                "sequence_no": str(sequence_no),
                "entry_count": str(len(entries)),
            },
        )
        save_safetensors_shard_manifest(
            sidecar_path,
            {
                "format": "mikazuki_anima_text_safetensors_shard",
                "format_version": 1,
                "namespace": self.ANIMA_TEXT_ENCODER_NAMESPACE,
                "shard_file": os.path.basename(shard_path),
                "sequence_no": sequence_no,
                "image_count": len(entries),
                "entries": manifest_entries,
            },
        )
        self._register_written_shard(cache_root, shard_path)

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ):
        anima_text_encoding_strategy: AnimaTextEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]

        tokens_and_masks = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = anima_text_encoding_strategy.encode_tokens(
                tokenize_strategy, models, tokens_and_masks
            )

        # Convert to numpy for caching. Disk cache prefers float16 to keep Anima TE cache size under control.
        if self.cache_to_disk:
            prompt_embeds = prompt_embeds.detach().to(dtype=torch.float16).cpu().numpy().astype(np.float16, copy=False)
        else:
            if prompt_embeds.dtype == torch.bfloat16:
                prompt_embeds = prompt_embeds.float()
            prompt_embeds = prompt_embeds.detach().cpu().numpy()
        attn_mask = attn_mask.cpu().numpy()
        t5_input_ids = t5_input_ids.cpu().numpy().astype(np.int32)
        t5_attn_mask = t5_attn_mask.cpu().numpy().astype(np.int32)

        for i, info in enumerate(infos):
            prompt_embeds_i, attn_mask_i, t5_input_ids_i, t5_attn_mask_i = self._trim_cached_outputs(
                prompt_embeds[i],
                attn_mask[i],
                t5_input_ids[i],
                t5_attn_mask[i],
            )
            caption_dropout_rate = torch.tensor(info.caption_dropout_rate, dtype=torch.float32)

            if self.cache_to_disk:
                cache_root = self._resolve_cache_root_for_info(info)
                if self._pending_cache_root is not None and self._pending_cache_root != cache_root:
                    self._flush_pending_entries()
                if self._pending_cache_root is None:
                    self._pending_cache_root = cache_root
                self._pending_entries.append(
                    {
                        "info": info,
                        "prompt_embeds": prompt_embeds_i,
                        "attn_mask": attn_mask_i,
                        "t5_input_ids": t5_input_ids_i,
                        "t5_attn_mask": t5_attn_mask_i,
                        "caption_dropout_rate": float(caption_dropout_rate.item()),
                    }
                )
                if len(self._pending_entries) >= self._max_pending_entries:
                    self._flush_pending_entries()
            else:
                info.text_encoder_outputs = (prompt_embeds_i, attn_mask_i, t5_input_ids_i, t5_attn_mask_i, caption_dropout_rate)

    def finalize_caching(self) -> None:
        self._flush_pending_entries()
        for cache_root in list(self._safetensors_catalog_cache.keys()):
            self._prune_stale_shards(cache_root)


class AnimaLatentsCachingStrategy(LatentsCachingStrategy):
    """Latent caching strategy for Anima using WanVAE.

    WanVAE produces 16-channel latents with spatial downscale 8x.
    Latent shape for images: (B, 16, 1, H/8, W/8)
    """

    ANIMA_LATENTS_NPZ_SUFFIX = "_anima.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return self.ANIMA_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.ANIMA_LATENTS_NPZ_SUFFIX

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, cache_ref_or_path, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        if not isinstance(cache_ref_or_path, (str, os.PathLike)):
            return super().load_latents_from_disk(cache_ref_or_path, bucket_reso)
        return self._default_load_latents_from_disk(8, os.fspath(cache_ref_or_path), bucket_reso)

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        prepared_batch = self.prepare_batch_latents(image_infos, alpha_mask, random_crop)
        self.cache_batch_latents_prepared(vae, image_infos, prepared_batch, flip_aug, alpha_mask, random_crop)

    def cache_batch_latents_prepared(self, vae, image_infos: List, prepared_batch, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        """Cache batch of latents using Qwen Image VAE.

        vae is expected to be the Qwen Image VAE (AutoencoderKLQwenImage).
        The encoding function handles the mean/std normalization.
        """
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage = vae
        vae_device = vae.device
        vae_dtype = vae.dtype

        def encode_by_vae(img_tensor):
            """Encode image tensor to latents.

            img_tensor: (B, C, H, W) in [-1, 1] range (already normalized by IMAGE_TRANSFORMS)
            Qwen Image VAE accepts inputs in (B, C, H, W) or (B, C, 1, H, W) shape.
            Returns latents in (B, 16, 1, H/8, W/8) shape on CPU.
            """
            latents = vae.encode_pixels_to_latents(img_tensor)  # Keep 4D for input/output
            return latents.to("cpu")

        self._default_cache_batch_latents_prepared(
            encode_by_vae,
            vae_device,
            vae_dtype,
            image_infos,
            prepared_batch,
            flip_aug,
            multi_resolution=True,
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae_device)
