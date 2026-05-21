from __future__ import annotations

from mikazuki.app.training_prompt_utils import parse_boolish


def normalize_conflicting_network_target_flags(config: dict) -> list[str]:
    if "network_train_unet_only" not in config or "network_train_text_encoder_only" not in config:
        return []

    train_unet_only = parse_boolish(config.get("network_train_unet_only"))
    train_text_encoder_only = parse_boolish(config.get("network_train_text_encoder_only"))
    if not train_unet_only or not train_text_encoder_only:
        return []

    config["network_train_unet_only"] = False
    config["network_train_text_encoder_only"] = False

    warnings = [
        "检测到“仅训练 DiT/U-Net”和“仅训练文本编码器”被同时勾选。"
        "这通常表示你想训练两者，因此本次已自动改为“同时训练 DiT/U-Net 和文本编码器”。"
    ]

    if parse_boolish(config.get("cache_text_encoder_outputs")):
        config["cache_text_encoder_outputs"] = False
        if "cache_text_encoder_outputs_to_disk" in config:
            config["cache_text_encoder_outputs_to_disk"] = False
        warnings.append(
            "由于已自动切换为同时训练文本编码器，文本编码器输出缓存也已自动关闭。"
        )

    return warnings


TLORA_STALE_NETWORK_ARG_PREFIXES = (
    "tlora_min_rank=",
    "tlora_rank_schedule=",
    "tlora_orthogonal_init=",
)

PISSA_STALE_NETWORK_ARG_PREFIXES = (
    "pissa_init=",
    "pissa_method=",
    "pissa_niter=",
    "pissa_oversample=",
    "pissa_apply_conv2d=",
    "pissa_export_mode=",
)

ANIMA_DORA_STALE_NETWORK_ARG_PREFIXES = (
    "dora_wd=",
    "bypass_mode=",
)

ANIMA_ADAPTER_ROUTE_TYPES = {
    "anima-lora",
    "anima-ileco",
    "anima-addift",
    "anima-multi-addift",
}


def normalize_network_args(*values) -> list[str]:
    items = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                item_str = str(item).strip()
                if item_str:
                    items.append(item_str)
        else:
            item_str = str(value).strip()
            if item_str:
                items.append(item_str)
    return items


def filter_network_args(args_list, stale_prefixes) -> list[str]:
    return [item for item in args_list if not str(item).startswith(tuple(stale_prefixes))]


def upsert_network_arg(args_list, key, value) -> list[str]:
    prefix = f"{key}="
    filtered = [item for item in args_list if not str(item).startswith(prefix)]
    if value is not None and str(value).strip() != "":
        filtered.append(f"{key}={value}")
    return filtered


def get_network_arg_value(args_list, key):
    prefix = f"{key}="
    for item in reversed(args_list):
        item_str = str(item).strip()
        if item_str.startswith(prefix):
            return item_str.split("=", 1)[1].strip()
    return None


def normalize_lokr_export_mode(value) -> str:
    mode = str(value or "native").strip().lower().replace("-", "_")
    return mode if mode in {"native", "lora_compatible"} else "native"


def normalize_bool_config_or_arg(config: dict, network_args: list[str], key: str, alias: str | None = None) -> bool:
    if key in config:
        return parse_boolish(config.get(key))
    if alias and alias in config:
        return parse_boolish(config.get(alias))
    existing = get_network_arg_value(network_args, key)
    if existing is None and alias:
        existing = get_network_arg_value(network_args, alias)
    return parse_boolish(existing) if existing is not None else False


def pop_network_args(config: dict) -> list[str]:
    return normalize_network_args(config.get("network_args"), config.pop("network_args_custom", None))


def assign_network_args(config: dict, network_args: list[str]) -> None:
    if network_args:
        config["network_args"] = network_args
    else:
        config.pop("network_args", None)


def apply_tlora_rank_overrides(config: dict, network_args: list[str]) -> list[str]:
    network_args = filter_network_args(network_args, TLORA_STALE_NETWORK_ARG_PREFIXES)

    try:
        network_dim = int(config.get("network_dim", 0) or 0)
    except (TypeError, ValueError):
        network_dim = 0

    try:
        min_rank = int(config.get("tlora_min_rank", 1) or 1)
    except (TypeError, ValueError):
        min_rank = 1

    if network_dim > 0:
        min_rank = max(1, min(min_rank, network_dim))
    else:
        min_rank = max(1, min_rank)

    config["tlora_min_rank"] = min_rank
    network_args = upsert_network_arg(network_args, "tlora_min_rank", min_rank)

    rank_schedule = str(config.get("tlora_rank_schedule", "cosine") or "cosine").strip().lower() or "cosine"
    if rank_schedule not in {"linear", "cosine"}:
        rank_schedule = "cosine"
    config["tlora_rank_schedule"] = rank_schedule
    network_args = upsert_network_arg(network_args, "tlora_rank_schedule", rank_schedule)

    orthogonal_init = parse_boolish(config.get("tlora_orthogonal_init", False))
    config["tlora_orthogonal_init"] = orthogonal_init
    network_args = upsert_network_arg(network_args, "tlora_orthogonal_init", "True" if orthogonal_init else "False")

    return network_args


def apply_pissa_overrides(config: dict, network_args: list[str]) -> list[str]:
    network_args = filter_network_args(network_args, PISSA_STALE_NETWORK_ARG_PREFIXES)

    pissa_init = parse_boolish(config.get("pissa_init", False))
    config["pissa_init"] = pissa_init
    if not pissa_init:
        return network_args

    pissa_method = str(config.get("pissa_method", "rsvd") or "rsvd").strip().lower() or "rsvd"
    if pissa_method not in {"rsvd", "svd"}:
        pissa_method = "rsvd"
    config["pissa_method"] = pissa_method

    try:
        pissa_niter = int(config.get("pissa_niter", 2) or 2)
    except (TypeError, ValueError):
        pissa_niter = 2
    pissa_niter = max(0, pissa_niter)
    config["pissa_niter"] = pissa_niter

    try:
        pissa_oversample = int(config.get("pissa_oversample", 8) or 8)
    except (TypeError, ValueError):
        pissa_oversample = 8
    pissa_oversample = max(0, pissa_oversample)
    config["pissa_oversample"] = pissa_oversample

    pissa_apply_conv2d = parse_boolish(config.get("pissa_apply_conv2d", False))
    config["pissa_apply_conv2d"] = pissa_apply_conv2d

    pissa_export_mode_raw = config.get("pissa_export_mode", "LoRA无损兼容导出")
    pissa_export_mode_text = str(pissa_export_mode_raw or "LoRA无损兼容导出").strip()
    pissa_export_mode = "approx" if "快速" in pissa_export_mode_text else "lossless"
    config["pissa_export_mode"] = pissa_export_mode_text

    network_args = upsert_network_arg(network_args, "pissa_init", "True")
    network_args = upsert_network_arg(network_args, "pissa_method", pissa_method)
    network_args = upsert_network_arg(network_args, "pissa_niter", pissa_niter)
    network_args = upsert_network_arg(network_args, "pissa_oversample", pissa_oversample)
    network_args = upsert_network_arg(network_args, "pissa_apply_conv2d", "True" if pissa_apply_conv2d else "False")
    network_args = upsert_network_arg(network_args, "pissa_export_mode", pissa_export_mode)
    return network_args


def apply_anima_ui_overrides(config: dict) -> None:
    model_train_type = str(config.get("model_train_type", "")).strip().lower()
    if not model_train_type.startswith("anima"):
        return

    sample_scheduler = str(config.get("sample_scheduler", "") or "").strip().lower()
    if not sample_scheduler:
        config["sample_scheduler"] = "simple"
    elif sample_scheduler != "simple":
        config["sample_scheduler"] = "simple"

    raw_sample_sampler = str(config.get("sample_sampler", "") or "").strip().lower()
    sample_sampler_aliases = {
        "euler_a": "euler",
        "k_euler_a": "k_euler",
    }
    normalized_sample_sampler = sample_sampler_aliases.get(raw_sample_sampler, raw_sample_sampler)
    if not normalized_sample_sampler:
        normalized_sample_sampler = "euler"
    elif normalized_sample_sampler not in {"euler", "k_euler"}:
        normalized_sample_sampler = "euler"
    config["sample_sampler"] = normalized_sample_sampler

    if model_train_type not in ANIMA_ADAPTER_ROUTE_TYPES:
        return

    lora_type = str(config.pop("lora_type", "")).strip().lower()
    network_args = pop_network_args(config)
    raw_train_norm = config.pop("train_norm", None)
    raw_dora_wd = config.pop("dora_wd", None)
    raw_bypass_mode = config.pop("bypass_mode", None)
    if raw_train_norm is None:
        existing_train_norm = get_network_arg_value(network_args, "train_norm")
        train_norm_enabled = parse_boolish(existing_train_norm) if existing_train_norm is not None else None
    else:
        train_norm_enabled = parse_boolish(raw_train_norm)

    existing_dora_wd = get_network_arg_value(network_args, "dora_wd")
    if raw_dora_wd is None:
        dora_enabled = parse_boolish(existing_dora_wd) if existing_dora_wd is not None else False
    else:
        dora_enabled = parse_boolish(raw_dora_wd)

    existing_bypass_mode = get_network_arg_value(network_args, "bypass_mode")
    if raw_bypass_mode is None:
        bypass_mode = parse_boolish(existing_bypass_mode) if existing_bypass_mode is not None else False
    else:
        bypass_mode = parse_boolish(raw_bypass_mode)

    if not lora_type:
        legacy_network_module = str(config.get("network_module", "")).strip().lower()
        legacy_adapter_type = str(get_network_arg_value(network_args, "anima_adapter_type") or "").strip().lower()
        if legacy_network_module == "networks.tlora_anima":
            lora_type = "tlora"
        elif legacy_adapter_type in {"lora", "lora_fa", "vera", "tlora", "lokr"}:
            lora_type = legacy_adapter_type
        elif legacy_network_module == "lycoris.kohya":
            lora_type = "lokr"
        elif str(get_network_arg_value(network_args, "algo") or "").strip().lower() == "lokr":
            lora_type = "lokr"
        else:
            lora_type = "lora"

    if lora_type:
        config["lora_type"] = lora_type
        config.pop("lycoris_algo", None)

        if lora_type == "lokr":
            config["network_module"] = "networks.lora_anima"
            config["anima_adapter_type"] = "lokr"
            config["dora_wd"] = False
            config["bypass_mode"] = False
            existing_lokr_export_mode = get_network_arg_value(network_args, "lokr_export_mode")
            lokr_export_mode = normalize_lokr_export_mode(config.get("lokr_export_mode", existing_lokr_export_mode))
            full_matrix_enabled = normalize_bool_config_or_arg(config, network_args, "full_matrix", "lokr_full_matrix")
            decompose_both_enabled = normalize_bool_config_or_arg(config, network_args, "decompose_both", "lokr_decompose_both")
            unbalanced_factorization_enabled = normalize_bool_config_or_arg(
                config, network_args, "unbalanced_factorization"
            )
            config["lokr_export_mode"] = lokr_export_mode
            config["full_matrix"] = full_matrix_enabled
            config["decompose_both"] = decompose_both_enabled
            config["unbalanced_factorization"] = unbalanced_factorization_enabled
            existing_lokr_factor = get_network_arg_value(network_args, "lokr_factor")
            legacy_factor = get_network_arg_value(network_args, "factor")
            if "lokr_factor" not in config:
                if existing_lokr_factor not in (None, ""):
                    config["lokr_factor"] = existing_lokr_factor
                elif legacy_factor not in (None, ""):
                    config["lokr_factor"] = legacy_factor
            lokr_factor = int(config.get("lokr_factor", 8) or 8)
            config["lokr_factor"] = lokr_factor
            network_args = upsert_network_arg(network_args, "anima_adapter_type", "lokr")
            network_args = upsert_network_arg(network_args, "lokr_factor", lokr_factor)
            network_args = upsert_network_arg(network_args, "lokr_export_mode", lokr_export_mode)
            network_args = upsert_network_arg(network_args, "full_matrix", "True" if full_matrix_enabled else None)
            network_args = upsert_network_arg(network_args, "decompose_both", "True" if decompose_both_enabled else None)
            network_args = upsert_network_arg(
                network_args, "unbalanced_factorization", "True" if unbalanced_factorization_enabled else None
            )
            if "dropout" in config:
                config["network_dropout"] = config.get("dropout")
            elif get_network_arg_value(network_args, "dropout") not in (None, ""):
                try:
                    config["network_dropout"] = float(get_network_arg_value(network_args, "dropout"))
                except (TypeError, ValueError):
                    config["network_dropout"] = 0
            elif "network_dropout" not in config:
                config["network_dropout"] = 0
            for key in ("conv_dim", "conv_alpha"):
                config.pop(key, None)
            stale_prefixes = (
                "algo=",
                "factor=",
                "conv_dim=",
                "conv_alpha=",
                "train_norm=",
                "dropout=",
                "tlora_min_rank=",
                "tlora_rank_schedule=",
                "tlora_orthogonal_init=",
                "lokr_full_matrix=",
                "lokr_decompose_both=",
                *ANIMA_DORA_STALE_NETWORK_ARG_PREFIXES,
                *PISSA_STALE_NETWORK_ARG_PREFIXES,
            )
            network_args = filter_network_args(network_args, stale_prefixes)
            config["pissa_init"] = False
        elif lora_type == "tlora":
            config["network_module"] = "networks.tlora_anima"
            config["anima_adapter_type"] = "tlora"
            config["dora_wd"] = False
            config["bypass_mode"] = False
            stale_prefixes = (
                "algo=",
                "factor=",
                "conv_dim=",
                "conv_alpha=",
                "train_norm=",
                "dropout=",
                "lokr_factor=",
                "lokr_export_mode=",
                "full_matrix=",
                "lokr_full_matrix=",
                "decompose_both=",
                "lokr_decompose_both=",
                "unbalanced_factorization=",
                "tlora_min_rank=",
                "tlora_rank_schedule=",
                "tlora_orthogonal_init=",
                *ANIMA_DORA_STALE_NETWORK_ARG_PREFIXES,
                *PISSA_STALE_NETWORK_ARG_PREFIXES,
            )
            network_args = filter_network_args(network_args, stale_prefixes)
            network_args = upsert_network_arg(network_args, "anima_adapter_type", "tlora")
            network_args = apply_tlora_rank_overrides(config, network_args)
            config["pissa_init"] = False

            for key in (
                "lokr_factor",
                "lokr_export_mode",
                "full_matrix",
                "lokr_full_matrix",
                "decompose_both",
                "lokr_decompose_both",
                "unbalanced_factorization",
                "conv_dim",
                "conv_alpha",
                "dropout",
            ):
                config.pop(key, None)
        elif lora_type == "lora_fa":
            config["network_module"] = "networks.lora_anima"
            config["anima_adapter_type"] = "lora_fa"
            config["dora_wd"] = False
            config["bypass_mode"] = False
            network_args = upsert_network_arg(network_args, "anima_adapter_type", "lora_fa")
            network_args = [
                item
                for item in network_args
                if not str(item).startswith(
                    (
                        "lokr_factor=",
                        "lokr_export_mode=",
                        "full_matrix=",
                        "lokr_full_matrix=",
                        "decompose_both=",
                        "lokr_decompose_both=",
                        "unbalanced_factorization=",
                        "tlora_min_rank=",
                        "tlora_rank_schedule=",
                        "tlora_orthogonal_init=",
                        *ANIMA_DORA_STALE_NETWORK_ARG_PREFIXES,
                    )
                )
            ]
            network_args = filter_network_args(network_args, PISSA_STALE_NETWORK_ARG_PREFIXES)
            config["pissa_init"] = False
            for key in (
                "lokr_factor",
                "lokr_export_mode",
                "full_matrix",
                "lokr_full_matrix",
                "decompose_both",
                "lokr_decompose_both",
                "unbalanced_factorization",
                "conv_dim",
                "conv_alpha",
                "dropout",
            ):
                config.pop(key, None)
        elif lora_type == "vera":
            config["network_module"] = "networks.lora_anima"
            config["anima_adapter_type"] = "vera"
            config["dora_wd"] = False
            config["bypass_mode"] = False
            network_args = upsert_network_arg(network_args, "anima_adapter_type", "vera")
            network_args = [
                item
                for item in network_args
                if not str(item).startswith(
                    (
                        "lokr_factor=",
                        "lokr_export_mode=",
                        "full_matrix=",
                        "lokr_full_matrix=",
                        "decompose_both=",
                        "lokr_decompose_both=",
                        "unbalanced_factorization=",
                        "tlora_min_rank=",
                        "tlora_rank_schedule=",
                        "tlora_orthogonal_init=",
                        *ANIMA_DORA_STALE_NETWORK_ARG_PREFIXES,
                    )
                )
            ]
            network_args = filter_network_args(network_args, PISSA_STALE_NETWORK_ARG_PREFIXES)
            config["pissa_init"] = False
            for key in (
                "lokr_factor",
                "lokr_export_mode",
                "full_matrix",
                "lokr_full_matrix",
                "decompose_both",
                "lokr_decompose_both",
                "unbalanced_factorization",
                "conv_dim",
                "conv_alpha",
                "dropout",
            ):
                config.pop(key, None)
        else:
            config["network_module"] = "networks.lora_anima"
            config["anima_adapter_type"] = "lora"
            if dora_enabled:
                bypass_mode = False
            config["dora_wd"] = dora_enabled
            config["bypass_mode"] = bypass_mode
            network_args = upsert_network_arg(network_args, "anima_adapter_type", "lora")
            network_args = [
                item
                for item in network_args
                if not str(item).startswith(
                    (
                        "lokr_factor=",
                        "lokr_export_mode=",
                        "full_matrix=",
                        "lokr_full_matrix=",
                        "decompose_both=",
                        "lokr_decompose_both=",
                        "unbalanced_factorization=",
                        "tlora_min_rank=",
                        "tlora_rank_schedule=",
                        "tlora_orthogonal_init=",
                    )
                )
            ]
            network_args = filter_network_args(network_args, ANIMA_DORA_STALE_NETWORK_ARG_PREFIXES)
            if dora_enabled:
                network_args = filter_network_args(network_args, PISSA_STALE_NETWORK_ARG_PREFIXES)
                config["pissa_init"] = False
            else:
                network_args = apply_pissa_overrides(config, network_args)
            network_args = upsert_network_arg(network_args, "dora_wd", "True" if dora_enabled else None)
            network_args = upsert_network_arg(network_args, "bypass_mode", "True" if bypass_mode else "False")
            for key in (
                "lokr_factor",
                "lokr_export_mode",
                "full_matrix",
                "lokr_full_matrix",
                "decompose_both",
                "lokr_decompose_both",
                "unbalanced_factorization",
                "conv_dim",
                "conv_alpha",
                "dropout",
            ):
                config.pop(key, None)

        if train_norm_enabled is not None:
            network_args = upsert_network_arg(network_args, "train_norm", "True" if train_norm_enabled else "False")

    assign_network_args(config, network_args)

    if "prefer_json_caption" in config:
        custom_attributes = config.get("custom_attributes")
        if not isinstance(custom_attributes, dict):
            custom_attributes = {}
        custom_attributes["prefer_json_caption"] = parse_boolish(config.pop("prefer_json_caption"))
        config["custom_attributes"] = custom_attributes


def apply_flux_tlora_ui_overrides(config: dict) -> None:
    model_train_type = str(config.get("model_train_type", "")).strip().lower()
    if model_train_type != "flux-lora":
        return

    network_args = pop_network_args(config)
    network_module = str(config.get("network_module", "") or "").strip().lower()
    network_args = filter_network_args(network_args, TLORA_STALE_NETWORK_ARG_PREFIXES)

    if network_module == "networks.tlora_flux":
        network_args = apply_tlora_rank_overrides(config, network_args)

    assign_network_args(config, network_args)

    sample_scheduler = str(config.get("sample_scheduler", "") or "").strip()
    if sample_scheduler == "":
        config["sample_scheduler"] = "simple"


def apply_stable_tlora_ui_overrides(config: dict) -> None:
    model_train_type = str(config.get("model_train_type", "")).strip().lower()
    if model_train_type not in {"sd-lora", "sdxl-lora"}:
        return

    network_args = pop_network_args(config)
    network_module = str(config.get("network_module", "") or "").strip().lower()
    network_args = filter_network_args(network_args, TLORA_STALE_NETWORK_ARG_PREFIXES)
    network_args = filter_network_args(network_args, PISSA_STALE_NETWORK_ARG_PREFIXES)

    if network_module == "networks.tlora":
        network_args = apply_tlora_rank_overrides(config, network_args)
        config["pissa_init"] = False
    elif network_module == "networks.lora":
        network_args = apply_pissa_overrides(config, network_args)
    else:
        config["pissa_init"] = False

    assign_network_args(config, network_args)
