from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = Path(r"H:\lulynx-trainer\backend")
TMP_ROOT = REPO_ROOT / "tmp" / "native_backend_smoke"
DATASET_ROOT = TMP_ROOT / "dataset"
OUTPUT_ROOT = TMP_ROOT / "outputs"
SOURCE_SAMPLE_DIR = REPO_ROOT / "sucai" / "6_lulu"
FALLBACK_SAMPLE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)


def _ensure_backend_imports() -> None:
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@dataclass
class SmokeResult:
    name: str
    stage: str
    ok: bool
    details: dict[str, Any]


def _write_result(result: SmokeResult) -> None:
    payload = {
        "name": result.name,
        "stage": result.stage,
        "ok": result.ok,
        "details": result.details,
    }
    print(json.dumps(payload, ensure_ascii=False, default=_json_default))


def prepare_smoke_dataset(sample_dir: Path | None = None) -> Path:
    target_dir = DATASET_ROOT / "1_smoke"
    target_dir.mkdir(parents=True, exist_ok=True)

    source_dir = sample_dir or SOURCE_SAMPLE_DIR
    image_src = source_dir / "2.png"
    caption_src = source_dir / "2.txt"
    image_dst = target_dir / "2.png"
    caption_dst = target_dir / "2.txt"

    try:
        if image_src.exists() and caption_src.exists():
            shutil.copy2(image_src, image_dst)
            shutil.copy2(caption_src, caption_dst)
            return target_dir
    except PermissionError:
        pass

    image_dst.write_bytes(base64.b64decode(FALLBACK_SAMPLE_PNG_BASE64))
    caption_dst.write_text("smoke sample", encoding="utf-8")
    return target_dir


def build_frontend_config(
    *,
    model_type: str,
    base_model_path: str = "",
    anima_model_path: str = "",
    anima_qwen3_path: str = "",
    vae_path: str = "",
    newbie_diffusers_path: str = "",
    output_name: str,
    train_data_dir: str,
    output_dir: str,
) -> dict[str, Any]:
    return {
        "training_type": "lora",
        "trainer_engine": "lulynx",
        "model_type": model_type,
        "pretrained_model_name_or_path": base_model_path,
        "anima_model_path": anima_model_path,
        "anima_qwen3_path": anima_qwen3_path,
        "newbie_diffusers_path": newbie_diffusers_path,
        "vae_path": vae_path,
        "train_data_dir": train_data_dir,
        "output_dir": output_dir,
        "output_name": output_name,
        "logging_dir": str(OUTPUT_ROOT / "logs" / output_name),
        "network_module": "lora",
        "network_dim": 4,
        "network_alpha": 4,
        "network_train_unet_only": True,
        "network_train_text_encoder_only": False,
        "optimizer_type": "AdamW",
        "learning_rate": 1e-4,
        "lr_scheduler": "constant",
        "max_train_steps": 1,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 0,
        "resolution": "1024,1024",
        "enable_bucket": False,
        "mixed_precision": "bf16",
        "save_precision": "bf16",
        "gradient_checkpointing": True,
        "cache_latents": False,
        "cache_latents_to_disk": False,
        "cache_text_encoder_outputs": False,
        "save_every_n_epochs": 999,
        "sample_every_n_steps": 0,
        "sample_every_n_epochs": 0,
        "xformers": False,
        "sdpa": True,
        "attention_backend": "sdpa",
        "seed": 1337,
        "caption_extension": ".txt",
    }


def _load_config(frontend_dict: dict[str, Any]):
    _ensure_backend_imports()
    from core.lulynx_trainer import ConfigAdapter

    return ConfigAdapter.from_frontend_dict(frontend_dict)


def _trainer_prepare(frontend_dict: dict[str, Any]) -> dict[str, Any]:
    _ensure_backend_imports()
    from core.lulynx_trainer import LulynxTrainer

    config = _load_config(frontend_dict)
    valid, errors, warnings = config.validate()
    details: dict[str, Any] = {
        "config_valid": valid,
        "config_errors": errors,
        "config_warnings": warnings,
        "model_type": str(config.model_type),
    }
    if not valid:
        raise RuntimeError(f"config validation failed: {errors}")

    trainer = LulynxTrainer(config=config)
    trainer.prepare()
    details["prepared"] = True
    if trainer.model is not None:
        details["loaded_model_arch"] = trainer.model.model_arch
        details["has_te2"] = trainer.model.text_encoder_2 is not None
        if hasattr(trainer.model, "anima_secondary_encoder_kind"):
            details["anima_secondary_encoder_kind"] = getattr(
                trainer.model, "anima_secondary_encoder_kind"
            )
        if hasattr(trainer.model, "newbie_scaffold_mode"):
            details["newbie_scaffold_mode"] = getattr(
                trainer.model, "newbie_scaffold_mode"
            )
    return details


def _loader_only_anima_qwen(frontend_dict: dict[str, Any]) -> dict[str, Any]:
    _ensure_backend_imports()
    from core.lulynx_trainer.anima_loader import load_anima_model

    model, report = load_anima_model(
        model_path=frontend_dict["anima_model_path"],
        qwen3_path=frontend_dict["anima_qwen3_path"],
        vae_path=frontend_dict["vae_path"],
        attn_mode="sdpa",
        device="cuda",
    )
    return {
        "loaded_model_arch": model.model_arch,
        "secondary_kind": getattr(model, "anima_secondary_encoder_kind", ""),
        "report_summary": report.summary(),
        "qwen3_loaded": report.qwen3_loaded,
        "limitations": [item.value for item in report.limitations],
    }


def try_find_sd15_model() -> Path | None:
    candidates = [
        REPO_ROOT / "models" / "sd15.safetensors",
        REPO_ROOT / "models" / "anything-v4.5-pruned.safetensors",
        REPO_ROOT / "models" / "v1-5-pruned-emaonly.safetensors",
        REPO_ROOT / "models" / "stable-diffusion-v1-5.safetensors",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_case(name: str, sample_dir: Path | None = None) -> SmokeResult:
    dataset_dir = prepare_smoke_dataset(sample_dir)
    (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)

    anima_model = REPO_ROOT / "models" / "diffusion_models" / "anima-preview2.safetensors"
    anima_vae = REPO_ROOT / "models" / "vae" / "qwen_image_vae.safetensors"
    anima_qwen = REPO_ROOT / "models" / "text_encoders" / "qwen_3_06b_base.safetensors"
    newbie_model = REPO_ROOT / "models" / "newbie"

    try:
        if name == "anima_prepare_clip_only":
            details = _trainer_prepare(
                build_frontend_config(
                    model_type="anima",
                    anima_model_path=str(anima_model),
                    vae_path=str(anima_vae),
                    output_name=name,
                    train_data_dir=str(dataset_dir),
                    output_dir=str(OUTPUT_ROOT / name),
                )
            )
            return SmokeResult(name=name, stage="prepare", ok=True, details=details)

        if name == "anima_loader_qwen_file":
            details = _loader_only_anima_qwen(
                build_frontend_config(
                    model_type="anima",
                    anima_model_path=str(anima_model),
                    anima_qwen3_path=str(anima_qwen),
                    vae_path=str(anima_vae),
                    output_name=name,
                    train_data_dir=str(dataset_dir),
                    output_dir=str(OUTPUT_ROOT / name),
                )
            )
            return SmokeResult(name=name, stage="loader", ok=True, details=details)

        if name == "newbie_prepare_current_layout":
            details = _trainer_prepare(
                build_frontend_config(
                    model_type="newbie",
                    newbie_diffusers_path=str(newbie_model),
                    output_name=name,
                    train_data_dir=str(dataset_dir),
                    output_dir=str(OUTPUT_ROOT / name),
                )
            )
            return SmokeResult(name=name, stage="prepare", ok=True, details=details)

        if name == "sd15_prepare_if_present":
            sd15_path = try_find_sd15_model()
            if sd15_path is None:
                return SmokeResult(
                    name=name,
                    stage="prepare",
                    ok=False,
                    details={"skipped": True, "reason": "no local sd15 checkpoint found"},
                )
            details = _trainer_prepare(
                build_frontend_config(
                    model_type="sd15",
                    base_model_path=str(sd15_path),
                    output_name=name,
                    train_data_dir=str(dataset_dir),
                    output_dir=str(OUTPUT_ROOT / name),
                )
            )
            return SmokeResult(name=name, stage="prepare", ok=True, details=details)

        raise ValueError(f"unknown smoke case: {name}")
    except Exception as exc:
        return SmokeResult(
            name=name,
            stage="error",
            ok=False,
            details={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[
            "anima_prepare_clip_only",
            "anima_loader_qwen_file",
            "newbie_prepare_current_layout",
            "sd15_prepare_if_present",
        ],
    )
    parser.add_argument(
        "--sample-dir",
        default="",
        help="Optional directory containing 2.png and 2.txt for smoke dataset seeding.",
    )
    args = parser.parse_args()

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    results = [run_case(name, sample_dir) for name in args.cases]
    for result in results:
        _write_result(result)

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
