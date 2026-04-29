from __future__ import annotations

import argparse
import re
from typing import Optional


_ZH_HELP_FALLBACK = "中文说明：请参考前面的英文描述。"
_JA_HELP_FALLBACK = "日本語説明：前の英語の説明を参照してください。"


def _classify_non_english_help_part(part: str) -> str:
    if re.search(r"[\u3040-\u30ff]", part):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", part):
        return "zh"
    return "other"


def normalize_help_text(help_text: Optional[str]) -> Optional[str]:
    if not isinstance(help_text, str) or " / " not in help_text:
        return help_text

    parts = [part.strip() for part in help_text.split(" / ") if part.strip()]
    if len(parts) < 2:
        return help_text

    english = parts[0]
    zh = None
    ja = None
    extras = []

    for part in parts[1:]:
        language = _classify_non_english_help_part(part)
        if language == "zh" and zh is None:
            zh = part
        elif language == "ja" and ja is None:
            ja = part
        else:
            extras.append(part)

    if zh is None and ja is None and extras:
        ja = extras.pop(0)

    if zh is None and extras:
        zh = extras.pop(0)

    if ja is None and extras:
        ja = extras.pop(0)

    if zh is None:
        zh = _ZH_HELP_FALLBACK
    if ja is None:
        ja = _JA_HELP_FALLBACK

    return " / ".join([english, zh, ja, *extras])


def build_add(parser: argparse.ArgumentParser):
    def add(*args, **kwargs):
        if "help" in kwargs:
            kwargs["help"] = normalize_help_text(kwargs["help"])
        return parser.add_argument(*args, **kwargs)

    return add
