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
