from __future__ import annotations

from mikazuki.app.training_network_overrides import (
    apply_anima_ui_overrides,
    apply_flux_tlora_ui_overrides,
    apply_stable_tlora_ui_overrides,
    normalize_conflicting_network_target_flags,
)
from mikazuki.app.training_sdxl_overrides import (
    apply_sdxl_block_swap_ui_overrides,
    apply_sdxl_low_vram_ui_overrides,
    build_sdxl_clip_skip_warning,
)


def apply_training_ui_overrides(config: dict) -> list[str]:
    warnings = apply_sdxl_low_vram_ui_overrides(config)
    warnings.extend(apply_sdxl_block_swap_ui_overrides(config))
    apply_anima_ui_overrides(config)
    apply_flux_tlora_ui_overrides(config)
    apply_stable_tlora_ui_overrides(config)
    warnings.extend(normalize_conflicting_network_target_flags(config))
    return warnings
