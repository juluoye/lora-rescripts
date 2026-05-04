"""Lazy exports for the mikazuki.app package.

This package is imported by the launcher for config access, so importing it
must not eagerly require FastAPI or the backend application stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        return import_module(".application", __name__).app
    raise AttributeError(name)
