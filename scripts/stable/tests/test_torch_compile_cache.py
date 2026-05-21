import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mikazuki.utils.torch_compile_cache import apply_torch_compile_cache_env, build_torch_compile_cache_context


def test_torch_compile_cache_context_builds_stable_root(tmp_path):
    config = {
        "torch_compile": True,
        "dynamo_backend": "inductor",
        "model_train_type": "anima-lora",
        "pretrained_model_name_or_path": r"D:\AI\models\anima-base-v1.0.safetensors",
        "mixed_precision": "bf16",
        "full_bf16": True,
    }
    env = {"MIKAZUKI_FLASHATTENTION_STARTUP": "1"}

    context = build_torch_compile_cache_context(config, env, repo_root=tmp_path)

    assert context is not None
    assert context.cache_root.is_relative_to(tmp_path / "cache" / "torch_compile")
    assert context.inductor_cache_dir.name == "inductor"
    assert context.triton_cache_dir.name == "triton"
    assert context.runtime_name == "flashattention"
    assert context.backend == "inductor"
    assert context.precision == "full-bf16"


def test_torch_compile_cache_env_writes_manifest(tmp_path):
    config = {
        "torch_compile": True,
        "dynamo_backend": "inductor",
        "model_train_type": "sdxl-lora",
        "pretrained_model_name_or_path": r"D:\AI\models\sdxl-base.safetensors",
        "mixed_precision": "fp16",
    }
    env = {}

    context = apply_torch_compile_cache_env(env, config, repo_root=tmp_path)

    assert context is not None
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(context.inductor_cache_dir)
    assert env["TRITON_CACHE_DIR"] == str(context.triton_cache_dir)
    assert env["TRITON_HOME"] == str(context.cache_root)
    assert env["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert env["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "1"

    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert manifest["enabled"] is True
    assert manifest["backend"] == "inductor"
    assert manifest["model_name"] == "sdxl-base"


def test_torch_compile_cache_skips_non_inductor_backend(tmp_path):
    config = {
        "torch_compile": True,
        "dynamo_backend": "aot_eager",
        "model_train_type": "anima-lora",
        "pretrained_model_name_or_path": r"D:\AI\models\anima-base-v1.0.safetensors",
    }
    env = {}

    context = apply_torch_compile_cache_env(env, config, repo_root=tmp_path)

    assert context is None
    assert env == {}
