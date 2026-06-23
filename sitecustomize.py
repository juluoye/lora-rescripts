from __future__ import annotations

import os
import subprocess
import sys


def _bootstrap_experimental_runtime_guards() -> None:
    try:
        from mikazuki.utils.runtime_mode import infer_attention_runtime_mode, is_amd_rocm_runtime, is_intel_xpu_runtime

        runtime_mode = infer_attention_runtime_mode()
        if not (is_amd_rocm_runtime(runtime_mode) or is_intel_xpu_runtime(runtime_mode)):
            return

        from mikazuki.utils.runtime_import_guards import install_experimental_runtime_import_guards

        install_experimental_runtime_import_guards()
    except Exception as exc:
        if str(os.environ.get("MIKAZUKI_DEBUG_SITECUSTOMIZE", "") or "").strip() == "1":
            print(f"[sitecustomize] experimental runtime bootstrap failed: {exc}", file=sys.stderr, flush=True)


def _patch_windows_subprocess_text_decoding() -> None:
    if sys.platform != "win32":
        return

    original_popen_init = getattr(subprocess.Popen, "__init__", None)
    if original_popen_init is None or getattr(subprocess.Popen, "_mikazuki_text_errors_patched", False):
        return

    def _patched_popen_init(self, *args, **kwargs):
        text_mode = bool(kwargs.get("text")) or bool(kwargs.get("universal_newlines"))
        if text_mode and kwargs.get("errors") is None:
            # Windows helper processes may still emit non-UTF-8 bytes even when
            # the parent expects text mode. Replace undecodable bytes instead of
            # letting subprocess background reader threads crash the launch flow.
            kwargs["errors"] = "replace"
        return original_popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _patched_popen_init
    subprocess.Popen._mikazuki_text_errors_patched = True


_patch_windows_subprocess_text_decoding()
_bootstrap_experimental_runtime_guards()
