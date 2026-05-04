"""Build script — package the launcher as a single EXE using PyInstaller."""

import shutil
import sys
import os
import subprocess
from pathlib import Path

import PyInstaller.__main__

LAUNCHER_DIR = Path(__file__).parent
PROJECT_ROOT = LAUNCHER_DIR.parent
PREFERRED_PYTHON = PROJECT_ROOT / "env" / "python_launcher" / "python.exe"

EXCLUDED_MODULES = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "diffusers",
    "xformers",
    "tensorflow",
    "tensorboard",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "sklearn",
    "onnxruntime",
    "cv2",
]


def _resolve_upx_dir() -> Path | None:
    explicit = os.environ.get("UPX_DIR", "").strip()
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if (candidate / "upx.exe").exists() or (candidate / "upx").exists():
            return candidate

    upx_on_path = shutil.which("upx")
    if upx_on_path:
        return Path(upx_on_path).resolve().parent

    common_candidates = [
        Path("C:/upx"),
        Path("C:/tools/upx"),
        PROJECT_ROOT / "tools" / "upx",
        PROJECT_ROOT / "tools" / "UPX",
        PROJECT_ROOT / "upx",
    ]
    for candidate in common_candidates:
        if (candidate / "upx.exe").exists() or (candidate / "upx").exists():
            return candidate.resolve()
    return None


def _should_reexec_with_preferred_python() -> bool:
    if os.environ.get("LAUNCHER_BUILD_REEXEC") == "1":
        return False
    if not PREFERRED_PYTHON.exists():
        return False
    try:
        current = Path(sys.executable).resolve()
        preferred = PREFERRED_PYTHON.resolve()
    except Exception:
        return False
    return current != preferred


def _reexec_with_preferred_python() -> int:
    command = [str(PREFERRED_PYTHON), str(Path(__file__).resolve())]
    env = os.environ.copy()
    env["LAUNCHER_BUILD_REEXEC"] = "1"
    if env.get("UPX_DIR"):
        print(f"[Launcher Build] Preserving UPX_DIR={env['UPX_DIR']}")
    print(f"[Launcher Build] Re-launching with preferred Python: {PREFERRED_PYTHON}")
    completed = subprocess.run(command, env=env)
    return int(completed.returncode)


def build():
    if _should_reexec_with_preferred_python():
        print(f"[Launcher Build] Current Python: {sys.executable}")
        print(
            "[Launcher Build] Repository-local launcher runtime detected. "
            "Using env/python_launcher/python.exe for a smaller and more consistent EXE."
        )
        raise SystemExit(_reexec_with_preferred_python())

    print(f"[Launcher Build] Python: {sys.executable}")
    icon_path = LAUNCHER_DIR / "assets" / "favicon-launcher.ico"
    if not icon_path.exists():
        fallback_icon = LAUNCHER_DIR / "assets" / "favicon-old.ico"
        if fallback_icon.exists():
            icon_path = fallback_icon
    icon_args = [f"--icon={icon_path}"] if icon_path.exists() else []

    # Web dist directory (built React SPA)
    web_dist = LAUNCHER_DIR / "web" / "dist"
    upx_dir = _resolve_upx_dir()

    # Collect all launcher submodules as hidden imports
    hidden_imports = [
        "--hidden-import=webview",
        "--hidden-import=launcher",
        "--hidden-import=launcher.main",
        "--hidden-import=launcher.config",
        "--hidden-import=launcher.i18n",
        "--hidden-import=launcher.api",
        "--hidden-import=launcher.window",
        "--hidden-import=launcher.core",
        "--hidden-import=launcher.core.launcher",
        "--hidden-import=launcher.core.installer",
        "--hidden-import=launcher.core.runtime_detector",
        "--hidden-import=launcher.core.settings",
        "--hidden-import=launcher.core.plugins",
        "--hidden-import=launcher.core.gpu",
        "--hidden-import=launcher.core.preflight",
        "--hidden-import=launcher.core.recommendation",
        "--hidden-import=launcher.core.api_result",
        "--hidden-import=launcher.core.compatibility",
        "--hidden-import=launcher.core.diagnostics",
        "--hidden-import=launcher.core.task_history_store",
        "--hidden-import=launcher.core.runtime_coordinator",
        "--hidden-import=launcher.core.runtime_catalog",
        "--hidden-import=launcher.core.runtime_tasks",
        "--hidden-import=launcher.core.task_executor",
        "--hidden-import=launcher.core.task_state",
        "--hidden-import=launcher.core.task_plans",
        "--hidden-import=launcher.core.update_checker",
        "--hidden-import=launcher.core.versioning",
    ]
    exclude_args = [f"--exclude-module={name}" for name in EXCLUDED_MODULES]

    args = [
        str(LAUNCHER_DIR / "main.py"),
        "--name=SD-reScripts-Launcher",
        "--onefile",
        "--windowed",
        f"--add-data={LAUNCHER_DIR / 'i18n'};launcher/i18n",
        f"--add-data={LAUNCHER_DIR / 'assets'};launcher/assets",
        f"--paths={PROJECT_ROOT}",
    ] + hidden_imports + exclude_args + icon_args + [
        "--clean",
        "--noconfirm",
        f"--distpath={PROJECT_ROOT / 'dist'}",
        f"--workpath={PROJECT_ROOT / 'build'}",
        f"--specpath={PROJECT_ROOT}",
    ]

    if upx_dir is not None:
        args.append(f"--upx-dir={upx_dir}")
        print(f"[Launcher Build] UPX enabled: {upx_dir}")
    else:
        print(
            "[Launcher Build] UPX not found. Building an uncompressed onefile EXE; "
            "the launcher size may be much larger than previous compressed builds."
        )
        print(
            "[Launcher Build] To restore smaller EXE size, install UPX and either add it to PATH "
            "or set UPX_DIR to the folder containing upx.exe."
        )

    # Include web dist if it exists
    if web_dist.exists():
        args.append(f"--add-data={web_dist};launcher/web/dist")

    PyInstaller.__main__.run(args)

    # Copy the EXE to project root for convenience
    exe_src = PROJECT_ROOT / "dist" / "SD-reScripts-Launcher.exe"
    exe_dst = PROJECT_ROOT / "SD-reScripts-Launcher.exe"
    if exe_src.exists():
        try:
            shutil.copy2(str(exe_src), str(exe_dst))
            print(f"\nCopied: {exe_src} -> {exe_dst}")
            print(f"Ready to use: double-click {exe_dst}")
        except PermissionError:
            print(f"\nBuild succeeded, but {exe_dst} is currently in use and could not be overwritten.")
            print(f"Close the running launcher, then copy this file manually: {exe_src}")
    else:
        print(f"\nWarning: {exe_src} not found. Build may have failed.")


if __name__ == "__main__":
    build()
