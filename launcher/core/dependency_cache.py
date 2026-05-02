"""Runtime dependency cache plans, status inspection, and prefetch helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from launcher.config import RUNTIMES, RUNTIME_MAP, RuntimeDef, get_repo_root
from launcher.core.subprocess_utils import hidden_subprocess_kwargs


_FLASHATTENTION_RELEASE_TAG = "v0.7.13"
_FLASHATTENTION_VERSION = "2.8.3"
_BLACKWELL_XFORMERS_WHEEL_URL = (
    "https://huggingface.co/czmahi/xformers-windows-torch2.8-cu128-py312/resolve/main/"
    "latest-torch2.8-python3.12-xformers-comfyui-windows/"
    "xformers-0.0.31%2B8fc8ec5a.d20250503-cp312-cp312-win_amd64.whl"
)

_CHINA_MIRROR_PRESETS: Dict[str, Dict[str, str]] = {
    "aliyun": {
        "pip_index_url": "https://mirrors.aliyun.com/pypi/simple/",
        "pip_find_links": "https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html",
        "hf_endpoint": "https://hf-mirror.com",
    },
    "tsinghua": {
        "pip_index_url": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "pip_find_links": "https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html",
        "hf_endpoint": "https://hf-mirror.com",
    },
    "ustc": {
        "pip_index_url": "https://pypi.mirrors.ustc.edu.cn/simple",
        "pip_find_links": "https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html",
        "hf_endpoint": "https://hf-mirror.com",
    },
    "baidu": {
        "pip_index_url": "https://mirror.baidu.com/pypi/simple",
        "pip_find_links": "https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html",
        "hf_endpoint": "https://hf-mirror.com",
    },
}


@dataclass(frozen=True)
class DependencyPlanItem:
    item_id: str
    label_zh: str
    label_en: str
    kind: str  # "pip" | "url"
    pip_args: tuple[str, ...] = ()
    url: str = ""
    note_zh: str = ""
    note_en: str = ""


def get_dependency_cache_root(repo_root: Optional[Path] = None) -> Path:
    if repo_root is None:
        repo_root = get_repo_root()
    return repo_root / "cache" / "dependencies"


def get_runtime_dependency_cache_dir(runtime_id: str, repo_root: Optional[Path] = None) -> Path:
    return get_dependency_cache_root(repo_root) / runtime_id


def _china_mirror_env(repo_root: Path) -> Dict[str, str]:
    config_path = repo_root / "config" / "china_mirror.json"
    preset_id = "aliyun"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            saved_id = str(payload.get("selected_id") or "").strip().lower()
            if saved_id in _CHINA_MIRROR_PRESETS:
                preset_id = saved_id
        except Exception:
            preset_id = "aliyun"

    preset = _CHINA_MIRROR_PRESETS.get(preset_id, _CHINA_MIRROR_PRESETS["aliyun"])
    env = {
        "MIKAZUKI_CN_MIRROR": "1",
        "MIKAZUKI_CN_MIRROR_PRESET": preset_id,
        "PIP_INDEX_URL": preset["pip_index_url"],
        "PIP_FIND_LINKS": preset["pip_find_links"],
        "HF_ENDPOINT": preset["hf_endpoint"],
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "GIT_TERMINAL_PROMPT": "false",
    }
    git_config = repo_root / "assets" / "gitconfig-cn"
    if git_config.exists():
        env["GIT_CONFIG_GLOBAL"] = str(git_config)
    return env


def build_dependency_cache_env(repo_root: Path, *, cn_mirror: bool) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HOME", "huggingface")
    env["PYTHONUTF8"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if cn_mirror:
        env.update(_china_mirror_env(repo_root))
    else:
        for key in ("MIKAZUKI_CN_MIRROR", "MIKAZUKI_CN_MIRROR_PRESET", "PIP_INDEX_URL", "PIP_FIND_LINKS", "HF_ENDPOINT", "GIT_CONFIG_GLOBAL"):
            env.pop(key, None)
    return env


def apply_proxy_settings(env: Dict[str, str], proxy_settings: Optional[Dict[str, str]]) -> Dict[str, str]:
    result = dict(env)
    proxy_settings = proxy_settings or {}
    for source_key, env_keys in (
        ("http_proxy", ("HTTP_PROXY", "http_proxy")),
        ("https_proxy", ("HTTPS_PROXY", "https_proxy")),
        ("all_proxy", ("ALL_PROXY", "all_proxy")),
    ):
        value = str(proxy_settings.get(source_key) or "").strip()
        if value:
            for env_key in env_keys:
                result[env_key] = value
        else:
            for env_key in env_keys:
                result.pop(env_key, None)
    return result


def _rewrite_url_for_cn_mirror(url: str, env: Dict[str, str]) -> str:
    hf_endpoint = str(env.get("HF_ENDPOINT") or "").rstrip("/")
    if hf_endpoint and url.startswith("https://huggingface.co/"):
        return hf_endpoint + url[len("https://huggingface.co") :]
    return url


def _flashattention_default_url() -> str:
    file_name = f"flash_attn-{_FLASHATTENTION_VERSION}+cu128torch2.10-cp312-cp312-win_amd64.whl"
    return (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/"
        f"{_FLASHATTENTION_RELEASE_TAG}/{urllib.parse.quote(file_name)}"
    )


def build_runtime_dependency_plan(runtime_id: str) -> List[DependencyPlanItem]:
    common_requirements = DependencyPlanItem(
        item_id="requirements",
        label_zh="项目依赖",
        label_en="Project requirements",
        kind="pip",
        pip_args=("--prefer-binary", "-r", "requirements.txt"),
        note_zh="缓存 requirements.txt 及其依赖轮子",
        note_en="Cache wheels for requirements.txt and its dependencies",
    )
    standard_torch = DependencyPlanItem(
        item_id="torch_stack",
        label_zh="PyTorch 与 TorchVision",
        label_en="PyTorch and TorchVision",
        kind="pip",
        pip_args=(
            "torch==2.10.0+cu128",
            "torchvision==0.25.0+cu128",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
        ),
    )
    standard_xformers = DependencyPlanItem(
        item_id="xformers",
        label_zh="xformers",
        label_en="xformers",
        kind="pip",
        pip_args=(
            "xformers>=0.0.34",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ),
        note_zh="可选，但建议提前缓存",
        note_en="Optional, but recommended to cache ahead of time",
    )
    sage_torch = DependencyPlanItem(
        item_id="torch_stack",
        label_zh="PyTorch 与 TorchVision",
        label_en="PyTorch and TorchVision",
        kind="pip",
        pip_args=(
            "torch==2.10.0+cu128",
            "torchvision==0.25.0+cu128",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
        ),
    )
    sage2_torch = DependencyPlanItem(
        item_id="torch_stack",
        label_zh="PyTorch 与 TorchVision",
        label_en="PyTorch and TorchVision",
        kind="pip",
        pip_args=(
            "torch==2.6.0+cu124",
            "torchvision==0.21.0+cu124",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ),
    )
    triton_runtime = DependencyPlanItem(
        item_id="triton_runtime",
        label_zh="Triton Runtime",
        label_en="Triton runtime",
        kind="pip",
        pip_args=("triton-windows==3.5.1.post24",),
    )
    sage_pkg = DependencyPlanItem(
        item_id="sageattention",
        label_zh="SageAttention 1.x",
        label_en="SageAttention 1.x",
        kind="pip",
        pip_args=("sageattention==1.0.6",),
    )
    sage2_pkg = DependencyPlanItem(
        item_id="sageattention",
        label_zh="SageAttention 2.x",
        label_en="SageAttention 2.x",
        kind="pip",
        pip_args=("sageattention==2.2.0",),
    )
    flash_wheel = DependencyPlanItem(
        item_id="flashattention_wheel",
        label_zh="FlashAttention 预编译轮子",
        label_en="FlashAttention prebuilt wheel",
        kind="url",
        url=_flashattention_default_url(),
    )
    blackwell_xformers = DependencyPlanItem(
        item_id="blackwell_xformers",
        label_zh="Blackwell xformers 轮子",
        label_en="Blackwell xformers wheel",
        kind="url",
        url=_BLACKWELL_XFORMERS_WHEEL_URL,
    )

    plan_map: Dict[str, List[DependencyPlanItem]] = {
        "standard": [standard_torch, standard_xformers, common_requirements],
        "flashattention": [standard_torch, common_requirements, flash_wheel],
        "sageattention": [sage_torch, common_requirements, triton_runtime, sage_pkg],
        "sageattention2": [sage2_torch, common_requirements, triton_runtime, sage2_pkg],
        "blackwell": [standard_torch, common_requirements, blackwell_xformers],
        "sageattention-blackwell": [standard_torch, common_requirements, blackwell_xformers, triton_runtime, sage_pkg],
        "intel-xpu": [common_requirements],
        "intel-xpu-sage": [common_requirements],
        "rocm-amd": [common_requirements],
    }
    return list(plan_map.get(runtime_id, [common_requirements]))


def _iter_cache_files(item_dir: Path) -> Iterable[Path]:
    if not item_dir.exists():
        return []
    return [path for path in item_dir.rglob("*") if path.is_file()]


def _dir_size(item_dir: Path) -> int:
    return sum(path.stat().st_size for path in _iter_cache_files(item_dir))


def _dir_file_count(item_dir: Path) -> int:
    return sum(1 for _ in _iter_cache_files(item_dir))


def _dir_updated_at(item_dir: Path) -> Optional[str]:
    latest_mtime: Optional[float] = None
    for path in _iter_cache_files(item_dir):
        mtime = path.stat().st_mtime
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
    if latest_mtime is None:
        return None
    return datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()


def get_runtime_dependency_cache_state(
    repo_root: Path,
    runtime_def: RuntimeDef,
) -> Dict[str, Any]:
    cache_dir = get_runtime_dependency_cache_dir(runtime_def.id, repo_root)
    plan = build_runtime_dependency_plan(runtime_def.id)
    items: List[Dict[str, Any]] = []
    total_bytes = 0
    cached_items = 0

    for item in plan:
        item_dir = cache_dir / item.item_id
        item_bytes = _dir_size(item_dir)
        file_count = _dir_file_count(item_dir)
        cached = file_count > 0 and item_bytes > 0
        if cached:
            cached_items += 1
        total_bytes += item_bytes
        items.append(
            {
                "item_id": item.item_id,
                "label_zh": item.label_zh,
                "label_en": item.label_en,
                "kind": item.kind,
                "note_zh": item.note_zh,
                "note_en": item.note_en,
                "cached": cached,
                "file_count": file_count,
                "bytes": item_bytes,
                "updated_at": _dir_updated_at(item_dir),
                "cache_dir": str(item_dir),
            }
        )

    return {
        "runtime_id": runtime_def.id,
        "cache_dir": str(cache_dir),
        "cache_exists": cache_dir.exists(),
        "ready": cached_items == len(plan) and len(plan) > 0,
        "total_items": len(plan),
        "cached_items": cached_items,
        "total_bytes": total_bytes,
        "items": items,
    }


def get_all_dependency_cache_states(repo_root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    if repo_root is None:
        repo_root = get_repo_root()
    return {
        runtime_def.id: get_runtime_dependency_cache_state(repo_root, runtime_def)
        for runtime_def in RUNTIMES
    }


def _download_url_to_path(
    url: str,
    destination: Path,
    *,
    env: Dict[str, str],
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_base: Optional[Dict[str, Any]] = None,
) -> None:
    request = urllib.request.Request(_rewrite_url_for_cn_mirror(url, env))
    chunk_size = 1024 * 1024
    downloaded = 0
    started_at = time.monotonic()
    last_sample_at = started_at
    last_sample_bytes = 0

    if log_callback:
        log_callback(f"Downloading: {request.full_url}")

    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        length_header = response.headers.get("Content-Length")
        total_bytes = int(length_header) if length_header and length_header.isdigit() else None
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if progress_callback and progress_base and (now - last_sample_at >= 0.75):
                speed = (downloaded - last_sample_bytes) / max(0.001, now - last_sample_at)
                payload = dict(progress_base)
                payload.update(
                    {
                        "item_downloaded_bytes": downloaded,
                        "item_total_bytes": total_bytes,
                        "item_speed_bytes_per_sec": speed,
                        "cache_bytes_on_disk": downloaded,
                    }
                )
                progress_callback(payload)
                last_sample_at = now
                last_sample_bytes = downloaded

    if progress_callback and progress_base:
        elapsed = max(0.001, time.monotonic() - started_at)
        payload = dict(progress_base)
        payload.update(
            {
                "item_downloaded_bytes": downloaded,
                "item_total_bytes": downloaded,
                "item_speed_bytes_per_sec": downloaded / elapsed,
                "cache_bytes_on_disk": downloaded,
            }
        )
        progress_callback(payload)


def _run_pip_download(
    python_path: Path,
    item_dir: Path,
    item: DependencyPlanItem,
    *,
    repo_root: Path,
    env: Dict[str, str],
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_base: Optional[Dict[str, Any]] = None,
) -> None:
    argv = [
        str(python_path),
        "-m",
        "pip",
        "download",
        "--dest",
        str(item_dir),
        "--progress-bar",
        "off",
        "--retries",
        "8",
        "--resume-retries",
        "8",
        "--timeout",
        "1000",
        *item.pip_args,
    ]
    if log_callback:
        log_callback(f"Running: {' '.join(argv)}")

    process = subprocess.Popen(
        argv,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **hidden_subprocess_kwargs(),
    )

    def _reader() -> None:
        if not process.stdout:
            return
        for line in process.stdout:
            if log_callback:
                log_callback(line.rstrip("\r\n"))

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    started_at = time.monotonic()
    last_sample_at = started_at
    last_sample_bytes = 0

    try:
        while process.poll() is None:
            time.sleep(0.75)
            if progress_callback and progress_base:
                downloaded = _dir_size(item_dir)
                now = time.monotonic()
                speed = (downloaded - last_sample_bytes) / max(0.001, now - last_sample_at)
                payload = dict(progress_base)
                payload.update(
                    {
                        "item_downloaded_bytes": downloaded,
                        "item_total_bytes": None,
                        "item_speed_bytes_per_sec": max(0.0, speed),
                        "cache_bytes_on_disk": downloaded,
                    }
                )
                progress_callback(payload)
                last_sample_at = now
                last_sample_bytes = downloaded
    finally:
        process.wait()
        reader.join(timeout=2.0)

    if process.returncode != 0:
        raise RuntimeError(f"pip download failed for {item.label_en} with exit code {process.returncode}.")

    if progress_callback and progress_base:
        downloaded = _dir_size(item_dir)
        elapsed = max(0.001, time.monotonic() - started_at)
        payload = dict(progress_base)
        payload.update(
            {
                "item_downloaded_bytes": downloaded,
                "item_total_bytes": downloaded,
                "item_speed_bytes_per_sec": downloaded / elapsed,
                "cache_bytes_on_disk": downloaded,
            }
        )
        progress_callback(payload)


def prefetch_runtime_dependencies(
    repo_root: Path,
    runtime_id: str,
    python_path: Path,
    *,
    cn_mirror: bool,
    proxy_settings: Optional[Dict[str, str]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    runtime_def = RUNTIME_MAP[runtime_id]
    cache_dir = get_runtime_dependency_cache_dir(runtime_id, repo_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = apply_proxy_settings(build_dependency_cache_env(repo_root, cn_mirror=cn_mirror), proxy_settings)
    plan = build_runtime_dependency_plan(runtime_id)

    if log_callback:
        log_callback(f"[Launcher] Dependency cache target: {cache_dir}")
        log_callback(f"[Launcher] Runtime dependency items: {len(plan)}")

    total_items = len(plan)
    for index, item in enumerate(plan, start=1):
        item_dir = cache_dir / item.item_id
        if item_dir.exists():
            shutil.rmtree(item_dir)
        item_dir.mkdir(parents=True, exist_ok=True)

        progress_base = {
            "runtime_id": runtime_id,
            "item_id": item.item_id,
            "item_label_zh": item.label_zh,
            "item_label_en": item.label_en,
            "item_kind": item.kind,
            "item_index": index,
            "completed_items": index - 1,
            "total_items": total_items,
        }
        if progress_callback:
            progress_callback(
                {
                    **progress_base,
                    "state": "running",
                    "item_downloaded_bytes": 0,
                    "item_total_bytes": None,
                    "item_speed_bytes_per_sec": 0.0,
                    "cache_bytes_on_disk": 0,
                }
            )

        if item.kind == "pip":
            _run_pip_download(
                python_path,
                item_dir,
                item,
                repo_root=repo_root,
                env=env,
                log_callback=log_callback,
                progress_callback=progress_callback,
                progress_base={**progress_base, "state": "running"},
            )
        elif item.kind == "url":
            parsed = urllib.parse.urlparse(item.url)
            file_name = Path(urllib.parse.unquote(parsed.path)).name
            if not file_name:
                raise RuntimeError(f"Could not infer filename from URL: {item.url}")
            _download_url_to_path(
                item.url,
                item_dir / file_name,
                env=env,
                log_callback=log_callback,
                progress_callback=progress_callback,
                progress_base={**progress_base, "state": "running"},
            )
        else:
            raise RuntimeError(f"Unsupported dependency cache item kind: {item.kind}")

        if progress_callback:
            progress_callback(
                {
                    **progress_base,
                    "state": "running",
                    "completed_items": index,
                    "item_downloaded_bytes": _dir_size(item_dir),
                    "item_total_bytes": _dir_size(item_dir),
                    "item_speed_bytes_per_sec": 0.0,
                    "cache_bytes_on_disk": _dir_size(item_dir),
                }
            )

    final_state = get_runtime_dependency_cache_state(repo_root, runtime_def)
    if progress_callback:
        progress_callback(
            {
                "runtime_id": runtime_id,
                "item_id": None,
                "item_label_zh": None,
                "item_label_en": None,
                "item_kind": None,
                "item_index": total_items,
                "completed_items": total_items,
                "total_items": total_items,
                "state": "succeeded",
                "item_downloaded_bytes": 0,
                "item_total_bytes": 0,
                "item_speed_bytes_per_sec": 0.0,
                "cache_bytes_on_disk": final_state["total_bytes"],
            }
        )
    return final_state


def clear_runtime_dependency_cache(repo_root: Path, runtime_id: str) -> Dict[str, Any]:
    cache_dir = get_runtime_dependency_cache_dir(runtime_id, repo_root)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    return get_runtime_dependency_cache_state(repo_root, RUNTIME_MAP[runtime_id])
