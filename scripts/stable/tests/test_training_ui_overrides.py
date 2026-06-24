from mikazuki.app.training_ui_overrides import apply_training_ui_overrides


def test_apply_training_ui_overrides_keeps_anima_finetune_sampler_defaults_without_lora_rewrite():
    config = {
        "model_train_type": "anima-finetune",
        "sample_scheduler": "ddim",
        "sample_sampler": "euler_a",
        "network_module": "some.custom.module",
        "network_args": ["anima_adapter_type=lokr", "bypass_mode=True"],
    }

    warnings = apply_training_ui_overrides(config)

    assert warnings == []
    assert config["sample_scheduler"] == "simple"
    assert config["sample_sampler"] == "euler"
    assert config["network_module"] == "some.custom.module"
    assert config["network_args"] == ["anima_adapter_type=lokr", "bypass_mode=True"]
    assert "lora_type" not in config
    assert "dora_wd" not in config
    assert "bypass_mode" not in config


def test_apply_training_ui_overrides_normalizes_anima_concept_edit_lora_defaults():
    config = {
        "model_train_type": "anima-addift",
        "sample_scheduler": "",
        "sample_sampler": "k_euler_a",
        "lora_type": "lora",
        "dora_wd": True,
        "bypass_mode": True,
        "network_args_custom": ["train_norm=True", "bypass_mode=True"],
    }

    warnings = apply_training_ui_overrides(config)

    assert warnings == []
    assert config["sample_scheduler"] == "simple"
    assert config["sample_sampler"] == "k_euler"
    assert config["lora_type"] == "lora"
    assert config["network_module"] == "networks.lora_anima"
    assert config["anima_adapter_type"] == "lora"
    assert config["dora_wd"] is True
    assert config["bypass_mode"] is False
    assert "network_args" in config
    assert "dora_wd=True" in config["network_args"]
    assert "bypass_mode=False" in config["network_args"]
    assert "train_norm=True" in config["network_args"]


def test_apply_training_ui_overrides_preserves_anima_lokr_native_options():
    config = {
        "model_train_type": "anima-lora",
        "lora_type": "lokr",
        "lokr_factor": 4,
        "lokr_export_mode": "native",
        "full_matrix": True,
        "decompose_both": True,
        "unbalanced_factorization": True,
        "network_args_custom": ["train_norm=True"],
    }

    warnings = apply_training_ui_overrides(config)

    assert warnings == []
    assert config["network_module"] == "networks.lora_anima"
    assert config["anima_adapter_type"] == "lokr"
    assert config["lokr_export_mode"] == "native"
    assert config["full_matrix"] is True
    assert config["decompose_both"] is True
    assert config["unbalanced_factorization"] is True
    assert "anima_adapter_type=lokr" in config["network_args"]
    assert "lokr_factor=4" in config["network_args"]
    assert "lokr_export_mode=native" in config["network_args"]
    assert "full_matrix=True" in config["network_args"]
    assert "decompose_both=True" in config["network_args"]
    assert "unbalanced_factorization=True" in config["network_args"]
    assert "train_norm=True" in config["network_args"]


def test_apply_training_ui_overrides_cleans_lokr_options_when_switching_to_lora():
    config = {
        "model_train_type": "anima-lora",
        "lora_type": "lora",
        "lokr_factor": 4,
        "lokr_export_mode": "native",
        "full_matrix": True,
        "decompose_both": True,
        "unbalanced_factorization": True,
        "network_args": [
            "anima_adapter_type=lokr",
            "lokr_factor=4",
            "lokr_export_mode=native",
            "full_matrix=True",
            "decompose_both=True",
            "unbalanced_factorization=True",
        ],
    }

    warnings = apply_training_ui_overrides(config)

    assert warnings == []
    assert config["anima_adapter_type"] == "lora"
    for key in ("lokr_factor", "lokr_export_mode", "full_matrix", "decompose_both", "unbalanced_factorization"):
        assert key not in config
    assert all("lokr_" not in item for item in config["network_args"])
    assert all(not item.startswith(("full_matrix=", "decompose_both=", "unbalanced_factorization=")) for item in config["network_args"])


def test_apply_training_ui_overrides_maps_anima_loha_to_lycoris_with_anima_preset():
    config = {
        "model_train_type": "anima-lora",
        "lora_type": "loha",
        "anima_main_block_template": "主干 block + adln（self_attn + cross_attn + mlp + adln）",
        "dropout": 0.1,
        "rank_dropout": 0.2,
        "module_dropout": 0.3,
        "train_norm": True,
        "dora_wd": True,
        "bypass_mode": True,
        "wd_on_output": False,
    }

    warnings = apply_training_ui_overrides(config)

    assert warnings == []
    assert config["network_module"] == "lycoris.kohya"
    assert config["lycoris_algo"] == "loha"
    assert "algo=loha" in config["network_args"]
    assert "preset=./config/lycoris_presets/anima_main_block_with_adln.toml" in config["network_args"]
    assert "dropout=0.1" in config["network_args"]
    assert "rank_dropout=0.2" in config["network_args"]
    assert "module_dropout=0.3" in config["network_args"]
    assert "train_norm=True" in config["network_args"]
    assert "dora_wd=True" in config["network_args"]
    assert "wd_on_output=False" in config["network_args"]
    assert "bypass_mode=True" in config["network_args"]


def test_apply_training_ui_overrides_maps_anima_boft_to_lycoris_with_main_block_preset():
    config = {
        "model_train_type": "anima-lora",
        "lora_type": "boft",
        "anima_main_block_template": "仅 mlp",
        "constraint": 0.75,
        "rescaled": True,
        "network_args_custom": ["preset=./config/lycoris_presets/custom_override.toml"],
    }

    warnings = apply_training_ui_overrides(config)

    assert warnings == []
    assert config["network_module"] == "lycoris.kohya"
    assert config["lycoris_algo"] == "boft"
    assert "algo=boft" in config["network_args"]
    assert "preset=./config/lycoris_presets/anima_mlp_only.toml" in config["network_args"]
    assert "preset=./config/lycoris_presets/custom_override.toml" in config["network_args"]
    assert "constraint=0.75" in config["network_args"]
    assert "rescaled=True" in config["network_args"]


def test_apply_training_ui_overrides_maps_anima_glora_to_lycoris_with_anima_preset():
    config = {
        "model_train_type": "anima-lora",
        "lora_type": "glora",
        "anima_main_block_template": "仅 attention（self_attn + cross_attn）",
        "dropout": 0.05,
        "rank_dropout": 0.15,
        "module_dropout": 0.25,
        "use_scalar": True,
        "rs_lora": True,
        "network_args_custom": ["algo=glora"],
    }

    warnings = apply_training_ui_overrides(config)

    assert warnings == []
    assert config["network_module"] == "lycoris.kohya"
    assert config["lycoris_algo"] == "glora"
    assert "algo=glora" in config["network_args"]
    assert "preset=./config/lycoris_presets/anima_attention_only.toml" in config["network_args"]
    assert "dropout=0.05" in config["network_args"]
    assert "rank_dropout=0.15" in config["network_args"]
    assert "module_dropout=0.25" in config["network_args"]
    assert "use_scalar=True" in config["network_args"]
    assert "rs_lora=True" in config["network_args"]


def test_apply_training_ui_overrides_maps_anima_glokr_to_lycoris_with_anima_preset():
    config = {
        "model_train_type": "anima-lora",
        "lora_type": "glokr",
        "anima_main_block_template": "主干 block（self_attn + cross_attn + mlp）",
        "dropout": 0.07,
        "rank_dropout": 0.17,
        "module_dropout": 0.27,
        "use_scalar": True,
        "rs_lora": True,
        "full_matrix": True,
        "decompose_both": True,
        "unbalanced_factorization": True,
        "lokr_factor": 6,
    }

    warnings = apply_training_ui_overrides(config)

    assert warnings == []
    assert config["network_module"] == "lycoris.kohya"
    assert config["lycoris_algo"] == "glokr"
    assert "algo=glokr" in config["network_args"]
    assert "preset=./config/lycoris_presets/anima_main_block.toml" in config["network_args"]
    assert "dropout=0.07" in config["network_args"]
    assert "rank_dropout=0.17" in config["network_args"]
    assert "module_dropout=0.27" in config["network_args"]
    assert "use_scalar=True" in config["network_args"]
    assert "rs_lora=True" in config["network_args"]
    assert "full_matrix=True" in config["network_args"]
    assert "decompose_both=True" in config["network_args"]
    assert "unbalanced_factorization=True" in config["network_args"]
    assert "lokr_factor=6" in config["network_args"]


def test_apply_training_ui_overrides_normalizes_legacy_anima_glokr_full_matrix_sentinel():
    config = {
        "model_train_type": "anima-lora",
        "lora_type": "glokr",
        "network_dim": 114514,
        "network_alpha": 114514,
        "network_module": "networks.lora_anima",
        "network_args": ["anima_adapter_type=glokr"],
    }

    warnings = apply_training_ui_overrides(config)

    assert warnings == []
    assert config["network_module"] == "lycoris.kohya"
    assert config["lycoris_algo"] == "glokr"
    assert config["network_dim"] == 1
    assert config["network_alpha"] == 1
    assert "algo=glokr" in config["network_args"]
    assert "full_matrix=True" in config["network_args"]
