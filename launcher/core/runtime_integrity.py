"""Quick runtime integrity checks for embedded Python environments."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from launcher.core.subprocess_utils import hidden_subprocess_kwargs


@dataclass(frozen=True)
class RuntimeIntegrity:
    """Result of a lightweight runtime integrity probe."""

    integrity_ok: bool
    bootstrap_ready: bool
    issue_code: Optional[str]
    message_zh: Optional[str]
    message_en: Optional[str]


def _issue(code: str, message_zh: str, message_en: str, *, integrity_ok: bool, bootstrap_ready: bool) -> RuntimeIntegrity:
    return RuntimeIntegrity(
        integrity_ok=integrity_ok,
        bootstrap_ready=bootstrap_ready,
        issue_code=code,
        message_zh=message_zh,
        message_en=message_en,
    )


def _ok(bootstrap_ready: bool) -> RuntimeIntegrity:
    if bootstrap_ready:
        return RuntimeIntegrity(
            integrity_ok=True,
            bootstrap_ready=True,
            issue_code=None,
            message_zh="运行时骨架完整，pip/bootstrap 也已就绪。",
            message_en="The runtime skeleton is complete and pip/bootstrap is ready.",
        )

    return RuntimeIntegrity(
        integrity_ok=True,
        bootstrap_ready=False,
        issue_code="pip_unavailable",
        message_zh="运行时骨架完整，但 pip/bootstrap 尚未就绪。先执行一次初始化即可。",
        message_en="The runtime skeleton is complete, but pip/bootstrap is not ready yet. Run Initialize first.",
    )


def _probe_embedded_python(python_path: Path, timeout_seconds: float = 5.0) -> RuntimeIntegrity:
    probe_script = (
        "import importlib.util, json\n"
        "result = {\n"
        "  'encodings_ok': False,\n"
        "  'pip_ready': False,\n"
        "  'encodings_error': '',\n"
        "}\n"
        "try:\n"
        "  import encodings  # noqa: F401\n"
        "  result['encodings_ok'] = True\n"
        "except Exception as exc:\n"
        "  result['encodings_error'] = str(exc)\n"
        "if result['encodings_ok']:\n"
        "  result['pip_ready'] = importlib.util.find_spec('pip') is not None\n"
        "print(json.dumps(result, ensure_ascii=True))\n"
    )

    try:
        completed = subprocess.run(
            [str(python_path), "-c", probe_script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            **hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return _issue(
            "probe_timeout",
            "运行时完整性检查超时，当前 Python 环境可能已经损坏或卡住。",
            "Runtime integrity probing timed out. This Python environment may be damaged or hanging.",
            integrity_ok=False,
            bootstrap_ready=False,
        )
    except OSError as exc:
        return _issue(
            "python_launch_failed",
            f"无法启动这个运行时里的 Python：{exc}",
            f"The launcher could not start Python from this runtime: {exc}",
            integrity_ok=False,
            bootstrap_ready=False,
        )

    output = (completed.stdout or "").strip().splitlines()
    payload = output[-1] if output else ""
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        detail = stderr or payload or f"exit code {completed.returncode}"
        return _issue(
            "python_probe_failed",
            f"运行时里的 Python 无法正常启动标准库探针：{detail}",
            f"Python inside this runtime could not start the standard-library probe: {detail}",
            integrity_ok=False,
            bootstrap_ready=False,
        )

    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        return _issue(
            "probe_output_invalid",
            "运行时完整性检查返回了无法解析的结果，当前环境可能不完整。",
            "Runtime integrity probing returned an unreadable result. This environment may be incomplete.",
            integrity_ok=False,
            bootstrap_ready=False,
        )

    if not result.get("encodings_ok"):
        detail = str(result.get("encodings_error") or "encodings import failed")
        return _issue(
            "encodings_unavailable",
            f"这个运行时缺少可用的标准库启动链（encodings 载入失败）：{detail}",
            f"This runtime is missing a usable standard-library bootstrap chain (encodings import failed): {detail}",
            integrity_ok=False,
            bootstrap_ready=False,
        )

    return _ok(bool(result.get("pip_ready")))


def assess_runtime_integrity(env_dir: Path, python_path: Path) -> RuntimeIntegrity:
    """Assess whether an embedded runtime skeleton is complete enough to use."""

    pth_files = list(env_dir.glob("python*._pth"))
    if not pth_files:
        return _issue(
            "missing_pth",
            "运行时目录里缺少 python*._pth，嵌入式 Python 路径白名单文件不完整。",
            "The runtime directory is missing python*._pth, so the embedded Python path allowlist is incomplete.",
            integrity_ok=False,
            bootstrap_ready=False,
        )

    zip_files = list(env_dir.glob("python*.zip"))
    if not zip_files:
        return _issue(
            "missing_stdlib_zip",
            "运行时目录里缺少 python*.zip，标准库压缩包不存在。",
            "The runtime directory is missing python*.zip, so the standard-library archive is absent.",
            integrity_ok=False,
            bootstrap_ready=False,
        )

    return _probe_embedded_python(python_path)
