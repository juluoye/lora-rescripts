"""Shared runtime task helpers for launch/install flows."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from launcher.config import (
    SAFE_MODE_CLEAR_VARS,
    STANDARD_ENV_CLEAR_VARS,
    STANDARD_ENV_VARS,
    RuntimeDef,
    get_repo_root,
)
from launcher.core.dependency_cache import get_dependency_cache_root, get_runtime_dependency_cache_dir
from launcher.core.proxy_utils import normalize_proxy_settings
from launcher.core.subprocess_utils import hidden_subprocess_kwargs


@dataclass
class LaunchOptions:
    """User-configurable launch options."""

    runtime_id: str = "standard"
    safe_mode: bool = False
    cn_mirror: bool = False
    attention_policy: str = "default"  # "default", "prefer_sage", "prefer_flash", "force_sdpa"
    host: str = "127.0.0.1"
    port: int = 28000
    listen: bool = False
    disable_tensorboard: bool = False
    disable_tageditor: bool = False
    disable_auto_mirror: bool = False
    dev_mode: bool = False
    localization: str = ""
    apply_proxy_to_trainer: bool = False
    http_proxy: str = ""
    https_proxy: str = ""
    all_proxy: str = ""


def build_launch_env(
    runtime_def: RuntimeDef,
    options: LaunchOptions,
) -> Dict[str, str]:
    """Build the environment dictionary for launching gui.py."""

    env = os.environ.copy()

    if options.safe_mode:
        for var in SAFE_MODE_CLEAR_VARS:
            env.pop(var, None)
        env["PYTHONNOUSERSITE"] = "1"

    for var in STANDARD_ENV_CLEAR_VARS:
        env.pop(var, None)

    for key, value in STANDARD_ENV_VARS.items():
        env[key] = value

    if runtime_def.preferred_runtime:
        env["MIKAZUKI_PREFERRED_RUNTIME"] = runtime_def.preferred_runtime
    else:
        env.pop("MIKAZUKI_PREFERRED_RUNTIME", None)

    for key, value in runtime_def.env_vars.items():
        env[key] = value

    runtime_default_attention_policy = runtime_def.env_vars.get("MIKAZUKI_STARTUP_ATTENTION_POLICY")

    if options.attention_policy == "force_sdpa":
        env["MIKAZUKI_STARTUP_ATTENTION_POLICY"] = "force_sdpa"
    elif runtime_default_attention_policy == "runtime_guarded":
        env["MIKAZUKI_STARTUP_ATTENTION_POLICY"] = runtime_default_attention_policy
    elif options.attention_policy == "prefer_flash":
        env["MIKAZUKI_STARTUP_ATTENTION_POLICY"] = "prefer_flash"
    elif options.attention_policy == "prefer_sage":
        env["MIKAZUKI_STARTUP_ATTENTION_POLICY"] = "prefer_sage"
    else:
        if "MIKAZUKI_STARTUP_ATTENTION_POLICY" not in runtime_def.env_vars:
            env.pop("MIKAZUKI_STARTUP_ATTENTION_POLICY", None)

    if options.cn_mirror:
        env["MIKAZUKI_CN_MIRROR"] = "1"
    else:
        env.pop("MIKAZUKI_CN_MIRROR", None)

    normalized_proxy = normalize_proxy_settings(
        {
            "http_proxy": options.http_proxy,
            "https_proxy": options.https_proxy,
            "all_proxy": options.all_proxy,
        }
    )
    for env_key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        env.pop(env_key, None)
    if options.apply_proxy_to_trainer:
        if normalized_proxy.get("http_proxy"):
            env["HTTP_PROXY"] = normalized_proxy["http_proxy"]
            env["http_proxy"] = normalized_proxy["http_proxy"]
        if normalized_proxy.get("https_proxy"):
            env["HTTPS_PROXY"] = normalized_proxy["https_proxy"]
            env["https_proxy"] = normalized_proxy["https_proxy"]
        if normalized_proxy.get("all_proxy"):
            env["ALL_PROXY"] = normalized_proxy["all_proxy"]
            env["all_proxy"] = normalized_proxy["all_proxy"]
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = "127.0.0.1,localhost"

    return env


def build_launch_args(options: LaunchOptions) -> List[str]:
    """Build the command-line arguments for gui.py."""

    args = ["gui.py"]

    if options.host and options.host != "127.0.0.1":
        args.extend(["--host", options.host])

    if options.port != 28000:
        args.extend(["--port", str(options.port)])

    if options.listen:
        args.append("--listen")

    if options.disable_tensorboard:
        args.append("--disable-tensorboard")

    if options.disable_tageditor:
        args.append("--disable-tageditor")

    if options.disable_auto_mirror:
        args.append("--disable-auto-mirror")

    if options.dev_mode:
        args.append("--dev")

    if options.localization:
        args.extend(["--localization", options.localization])

    return args


def build_launch_command(python_path: Path, options: LaunchOptions) -> List[str]:
    """Build the full launch command."""

    return [str(python_path)] + build_launch_args(options)


def spawn_launch_process(
    command: List[str],
    env: Dict[str, str],
    repo_root: Optional[Path] = None,
) -> subprocess.Popen:
    """Spawn the launcher process for gui.py."""

    if repo_root is None:
        repo_root = get_repo_root()

    return subprocess.Popen(
        command,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **hidden_subprocess_kwargs(
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        ),
    )


def build_install_env(
    runtime_id: Optional[str] = None,
    cn_mirror: bool = False,
    proxy_settings: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build environment for runtime install scripts."""

    env = os.environ.copy()
    if cn_mirror:
        env["MIKAZUKI_CN_MIRROR"] = "1"
    else:
        env.pop("MIKAZUKI_CN_MIRROR", None)
    cache_root = get_dependency_cache_root()
    env["MIKAZUKI_DEPENDENCY_CACHE_ROOT"] = str(cache_root)
    if runtime_id:
        env["MIKAZUKI_DEPENDENCY_CACHE_DIR"] = str(get_runtime_dependency_cache_dir(runtime_id))
    else:
        env.pop("MIKAZUKI_DEPENDENCY_CACHE_DIR", None)
    proxy_settings = proxy_settings or {}
    for source_key, env_keys in (
        ("http_proxy", ("HTTP_PROXY", "http_proxy")),
        ("https_proxy", ("HTTPS_PROXY", "https_proxy")),
        ("all_proxy", ("ALL_PROXY", "all_proxy")),
    ):
        value = str(proxy_settings.get(source_key) or "").strip()
        if value:
            for env_key in env_keys:
                env[env_key] = value
        else:
            for env_key in env_keys:
                env.pop(env_key, None)
    return env


def build_install_commands(
    runtime_def: RuntimeDef,
    repo_root: Optional[Path] = None,
) -> List[List[str]]:
    """Build PowerShell commands for runtime install scripts."""

    if repo_root is None:
        repo_root = get_repo_root()

    commands: List[List[str]] = []
    for script_name in runtime_def.install_scripts:
        script_path = repo_root / script_name
        commands.append(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ]
        )
    return commands


def _emit_stream_output(
    line: str,
    *,
    progress: bool,
    log_callback: Optional[Callable[[str], None]] = None,
    output_callback: Optional[Callable[[str, bool], None]] = None,
) -> None:
    if output_callback is not None:
        output_callback(line, progress)
    elif log_callback is not None:
        log_callback(line)


def _read_streamed_output(
    stream,
    *,
    log_callback: Optional[Callable[[str], None]] = None,
    output_callback: Optional[Callable[[str, bool], None]] = None,
    encoding: str = "utf-8",
) -> None:
    if stream is None:
        return

    try:
        fd = stream.fileno()
    except Exception:
        return

    def _decode(raw: bytes) -> str:
        try:
            return raw.decode(encoding, errors="replace")
        except Exception:
            return raw.decode("utf-8", errors="replace")

    buffer = b""
    while True:
        try:
            chunk = os.read(fd, 8192)
        except OSError:
            break
        if not chunk:
            break
        buffer += chunk

        while True:
            cr_idx = buffer.find(b"\r")
            lf_idx = buffer.find(b"\n")
            if cr_idx == -1 and lf_idx == -1:
                break

            if cr_idx == -1:
                idx = lf_idx
            elif lf_idx == -1:
                idx = cr_idx
            else:
                idx = min(cr_idx, lf_idx)

            is_progress = buffer[idx : idx + 1] == b"\r"
            delimiter_length = 1
            if is_progress and idx + 1 < len(buffer) and buffer[idx + 1 : idx + 2] == b"\n":
                is_progress = False
                delimiter_length = 2

            raw_line = buffer[:idx]
            buffer = buffer[idx + delimiter_length :]

            line = _decode(raw_line).rstrip()
            if line:
                _emit_stream_output(
                    line,
                    progress=is_progress,
                    log_callback=log_callback,
                    output_callback=output_callback,
                )

    if buffer:
        line = _decode(buffer).rstrip()
        if line:
            _emit_stream_output(
                line,
                progress=False,
                log_callback=log_callback,
                output_callback=output_callback,
            )


def run_streamed_command(
    command: List[str],
    env: Dict[str, str],
    cwd: Path,
    log_callback: Optional[Callable[[str], None]] = None,
    output_callback: Optional[Callable[[str, bool], None]] = None,
) -> bool:
    """Run a command while streaming merged stdout/stderr."""

    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            **hidden_subprocess_kwargs(),
        )

        if process.stdout:
            _read_streamed_output(
                process.stdout,
                log_callback=log_callback,
                output_callback=output_callback,
            )

        process.wait()
        _emit_stream_output(
            f"Exit code: {process.returncode}",
            progress=False,
            log_callback=log_callback,
            output_callback=output_callback,
        )
        return process.returncode == 0
    except FileNotFoundError:
        _emit_stream_output(
            f"Error: executable not found: {command[0]}",
            progress=False,
            log_callback=log_callback,
            output_callback=output_callback,
        )
        return False
    except Exception as exc:
        _emit_stream_output(
            f"Error running command: {exc}",
            progress=False,
            log_callback=log_callback,
            output_callback=output_callback,
        )
        return False
