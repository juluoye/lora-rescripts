from pathlib import Path

from mikazuki.utils.direct_trainers import validate_concept_edit_runtime_config


def test_validate_concept_edit_runtime_config_requires_addift_images(tmp_path: Path):
    config = {
        "model_train_type": "sd-addift",
        "original_prompt": "open eyes",
        "target_prompt": "closed eyes",
        "original_image_path": str(tmp_path / "missing.png"),
        "target_image_path": str(tmp_path / "target.png"),
    }
    message = validate_concept_edit_runtime_config(config)
    assert message is not None
    assert "原始图像不存在" in message


def test_validate_concept_edit_runtime_config_accepts_multi_addift_pair_dir(tmp_path: Path):
    original = tmp_path / "sample.png"
    target = tmp_path / "sample_target.png"
    original.write_bytes(b"fake")
    target.write_bytes(b"fake")

    config = {
        "model_train_type": "sd-multi-addift",
        "original_prompt": "base prompt",
        "target_prompt": "edited prompt",
        "concept_edit_data_dir": str(tmp_path),
        "diff_target_name": "_target",
    }
    message = validate_concept_edit_runtime_config(config)
    assert message is None


def test_validate_concept_edit_runtime_config_requires_unet_only(tmp_path: Path):
    original = tmp_path / "sample.png"
    target = tmp_path / "sample_target.png"
    original.write_bytes(b"fake")
    target.write_bytes(b"fake")

    config = {
        "model_train_type": "sd-addift",
        "original_prompt": "open eyes",
        "target_prompt": "closed eyes",
        "original_image_path": str(original),
        "target_image_path": str(target),
        "network_train_unet_only": False,
        "network_train_text_encoder_only": False,
    }
    message = validate_concept_edit_runtime_config(config)
    assert message is not None
    assert "U-Net / DiT only" in message


def test_validate_concept_edit_runtime_config_accepts_anima_multi_addift_pair_dir(tmp_path: Path):
    original = tmp_path / "sample.png"
    target = tmp_path / "sample_target.png"
    original.write_bytes(b"fake")
    target.write_bytes(b"fake")

    config = {
        "model_train_type": "anima-multi-addift",
        "original_prompt": "base prompt",
        "target_prompt": "edited prompt",
        "concept_edit_data_dir": str(tmp_path),
        "diff_target_name": "_target",
        "network_train_unet_only": True,
    }
    message = validate_concept_edit_runtime_config(config)
    assert message is None
