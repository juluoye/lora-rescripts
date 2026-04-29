from __future__ import annotations

import logging

import diffusers
import torch
from einops import rearrange

from library.original_unet import FlashAttentionFunction

logger = logging.getLogger(__name__)


def replace_unet_modules(unet, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        logger.info("Enable memory efficient attention for U-Net")
        unet.set_use_memory_efficient_attention(False, True)
    elif xformers:
        logger.info("Enable xformers for U-Net")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.info("Enable SDPA for U-Net")
        unet.set_use_memory_efficient_attention(False, False)
        unet.set_use_sdpa(True)


def replace_vae_modules(vae: diffusers.models.AutoencoderKL, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        replace_vae_attn_to_memory_efficient()
    elif xformers:
        vae.set_use_memory_efficient_attention_xformers(True)
    elif sdpa:
        replace_vae_attn_to_sdpa()


def replace_vae_attn_to_memory_efficient():
    logger.info("VAE Attention.forward has been replaced to FlashAttention (not xformers)")
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, hidden_states, **kwargs):
        q_bucket_size = 512
        k_bucket_size = 1024

        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        query_proj = self.to_q(hidden_states)
        key_proj = self.to_k(hidden_states)
        value_proj = self.to_v(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (query_proj, key_proj, value_proj)
        )

        out = flash_func.apply(query_proj, key_proj, value_proj, None, False, q_bucket_size, k_bucket_size)
        out = rearrange(out, "b h n d -> b n (h d)")

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    def forward_flash_attn_0_14(self, hidden_states, **kwargs):
        if not hasattr(self, "to_q"):
            self.to_q = self.query
            self.to_k = self.key
            self.to_v = self.value
            self.to_out = [self.proj_attn, torch.nn.Identity()]
            self.heads = self.num_heads
        return forward_flash_attn(self, hidden_states, **kwargs)

    if diffusers.__version__ < "0.15.0":
        diffusers.models.attention.AttentionBlock.forward = forward_flash_attn_0_14
    else:
        diffusers.models.attention_processor.Attention.forward = forward_flash_attn


def replace_vae_attn_to_xformers():
    logger.info("VAE: Attention.forward has been replaced to xformers")
    import xformers.ops

    def forward_xformers(self, hidden_states, **kwargs):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        query_proj = self.to_q(hidden_states)
        key_proj = self.to_k(hidden_states)
        value_proj = self.to_v(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (query_proj, key_proj, value_proj)
        )

        query_proj = query_proj.contiguous()
        key_proj = key_proj.contiguous()
        value_proj = value_proj.contiguous()
        out = xformers.ops.memory_efficient_attention(query_proj, key_proj, value_proj, attn_bias=None)
        out = rearrange(out, "b h n d -> b n (h d)")

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    def forward_xformers_0_14(self, hidden_states, **kwargs):
        if not hasattr(self, "to_q"):
            self.to_q = self.query
            self.to_k = self.key
            self.to_v = self.value
            self.to_out = [self.proj_attn, torch.nn.Identity()]
            self.heads = self.num_heads
        return forward_xformers(self, hidden_states, **kwargs)

    if diffusers.__version__ < "0.15.0":
        diffusers.models.attention.AttentionBlock.forward = forward_xformers_0_14
    else:
        diffusers.models.attention_processor.Attention.forward = forward_xformers


def replace_vae_attn_to_sdpa():
    logger.info("VAE: Attention.forward has been replaced to sdpa")

    def forward_sdpa(self, hidden_states, **kwargs):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        query_proj = self.to_q(hidden_states)
        key_proj = self.to_k(hidden_states)
        value_proj = self.to_v(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.heads), (query_proj, key_proj, value_proj)
        )

        out = torch.nn.functional.scaled_dot_product_attention(
            query_proj, key_proj, value_proj, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        out = rearrange(out, "b n h d -> b n (h d)")

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    def forward_sdpa_0_14(self, hidden_states, **kwargs):
        if not hasattr(self, "to_q"):
            self.to_q = self.query
            self.to_k = self.key
            self.to_v = self.value
            self.to_out = [self.proj_attn, torch.nn.Identity()]
            self.heads = self.num_heads
        return forward_sdpa(self, hidden_states, **kwargs)

    if diffusers.__version__ < "0.15.0":
        diffusers.models.attention.AttentionBlock.forward = forward_sdpa_0_14
    else:
        diffusers.models.attention_processor.Attention.forward = forward_sdpa


__all__ = [
    "replace_unet_modules",
    "replace_vae_attn_to_memory_efficient",
    "replace_vae_attn_to_sdpa",
    "replace_vae_attn_to_xformers",
    "replace_vae_modules",
]
