# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Clean-room dataset tag editor backend service.

Provides a stateless API surface for future UIs:
- browse/filter dataset images and captions
- compute common tags and tag frequency summaries
- save single/batch caption edits
- apply batch caption operations
- interrogate a single image with WD14/Gemini
- move/delete image+caption pairs safely
"""

from __future__ import annotations

import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image

from core.security import validate_path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
DEFAULT_CAPTION_EXTENSION = ".txt"
SUPPORTED_CAPTION_EXTENSIONS = (".txt", ".caption")


@dataclass
class DatasetCaptionItem:
    image_path: Path
    relative_path: str
    caption_path: Path
    caption_exists: bool
    caption_text: str
    mtime: float

    @property
    def tags(self) -> List[str]:
        return split_tags(self.caption_text)


def split_tags(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part and part.strip()]


def join_tags(tags: Sequence[str]) -> str:
    return ", ".join(tag for tag in tags if tag)


def dedupe_tags(tags: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = tag.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result


def normalize_caption_extension(raw: str = "") -> str:
    value = str(raw or "").strip()
    if not value:
        return DEFAULT_CAPTION_EXTENSION
    if not value.startswith("."):
        value = "." + value
    return value


def _relative_to(path: Path, parent: Path) -> str:
    return str(path.relative_to(parent)).replace("\\", "/")


def _find_existing_caption_path(image_path: Path) -> tuple[Path, bool]:
    candidates = (
        image_path.parent / f"{image_path.name}.txt",
        image_path.with_suffix(".txt"),
        image_path.with_suffix(".caption"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, True
    return image_path.with_suffix(DEFAULT_CAPTION_EXTENSION), False


def _resolve_caption_path(image_path: Path, preferred_extension: str = "") -> tuple[Path, bool]:
    existing_path, exists = _find_existing_caption_path(image_path)
    if exists:
        return existing_path, True
    ext = normalize_caption_extension(preferred_extension)
    return image_path.with_suffix(ext), False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


class TagEditorService:
    """Stateless clean-room service for tag-editor style dataset operations."""

    def load_dataset(
        self,
        directory: str,
        *,
        recursive: bool = True,
        caption_extension: str = "",
        limit: int = 200,
        offset: int = 0,
        sort_by: str = "name",
        sort_order: str = "asc",
        filename_query: str = "",
        caption_query: str = "",
        positive_tags: Optional[Sequence[str]] = None,
        positive_logic: str = "OR",
        negative_tags: Optional[Sequence[str]] = None,
        negative_logic: str = "OR",
        selected_paths: Optional[Sequence[str]] = None,
        selection_mode: str = "all",
        has_caption: str = "any",
        top_tags_limit: int = 50,
    ) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        items = self._scan_dataset(dataset_dir, recursive=recursive, caption_extension=caption_extension)
        filtered = self._filter_items(
            items,
            filename_query=filename_query,
            caption_query=caption_query,
            positive_tags=positive_tags or [],
            positive_logic=positive_logic,
            negative_tags=negative_tags or [],
            negative_logic=negative_logic,
            selected_paths=selected_paths or [],
            selection_mode=selection_mode,
            has_caption=has_caption,
        )
        sorted_items = self._sort_items(filtered, sort_by=sort_by, sort_order=sort_order)
        page = sorted_items[offset : offset + limit]

        tag_counts = self._tag_counts(filtered)
        common_tags = self._common_tags(filtered)
        return {
            "directory": str(dataset_dir),
            "recursive": recursive,
            "total": len(items),
            "filtered_total": len(filtered),
            "offset": offset,
            "limit": limit,
            "items": [self._serialize_item(item) for item in page],
            "common_tags": common_tags,
            "top_tags": [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(max(0, top_tags_limit))],
            "summary": {
                "captioned_count": sum(1 for item in filtered if item.caption_text),
                "uncaptioned_count": sum(1 for item in filtered if not item.caption_text),
                "selected_count": len(filtered),
            },
            "capabilities": self.capabilities(),
        }

    def save_caption(
        self,
        *,
        image_path: str,
        caption: str,
        caption_extension: str = "",
    ) -> Dict[str, Any]:
        image = validate_path(image_path, must_exist=True, allow_files=True, allow_dirs=False)
        caption_path, _ = _resolve_caption_path(image, caption_extension)
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text(str(caption or "").strip(), encoding="utf-8")
        return {
            "image_path": str(image),
            "caption_path": str(caption_path),
            "caption_length": len(str(caption or "").strip()),
        }

    def save_captions_batch(
        self,
        updates: Sequence[Dict[str, Any]],
        *,
        caption_extension: str = "",
    ) -> Dict[str, Any]:
        success = 0
        failed = 0
        errors: List[str] = []
        for update in updates:
            try:
                self.save_caption(
                    image_path=str(update.get("image_path", "") or ""),
                    caption=str(update.get("caption", "") or ""),
                    caption_extension=str(update.get("caption_extension", "") or caption_extension),
                )
                success += 1
            except Exception as exc:
                failed += 1
                errors.append(f"{update.get('image_path', '')}: {exc}")
        return {"success": success, "failed": failed, "errors": errors}

    def apply_batch_action(
        self,
        directory: str,
        *,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        image_paths: Optional[Sequence[str]] = None,
        recursive: bool = True,
        caption_extension: str = "",
        create_backup: bool = False,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        items = self._resolve_target_items(
            dataset_dir,
            image_paths=image_paths or [],
            recursive=recursive,
            caption_extension=caption_extension,
            filter_payload=filter_payload or {},
        )
        params = dict(params or {})
        backup_name = ""
        if create_backup and items:
            backup_name = self._create_backup_for_items(dataset_dir, items)

        modified = 0
        unchanged = 0
        samples: List[Dict[str, Any]] = []
        tag_counts = self._tag_counts(items)
        for item in items:
            before = item.caption_text
            after = self._apply_action(before, action=action, params=params, tag_counts=tag_counts)
            if before == after:
                unchanged += 1
                continue
            item.caption_path.parent.mkdir(parents=True, exist_ok=True)
            item.caption_path.write_text(after, encoding="utf-8")
            modified += 1
            if len(samples) < 20:
                samples.append(
                    {
                        "image_path": str(item.image_path),
                        "caption_path": str(item.caption_path),
                        "before": before,
                        "after": after,
                    }
                )
        return {
            "target_count": len(items),
            "modified_count": modified,
            "unchanged_count": unchanged,
            "backup_name": backup_name,
            "samples": samples,
        }

    def interrogate_image(
        self,
        *,
        image_path: str,
        method: str = "wd14",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        image = validate_path(image_path, must_exist=True, allow_files=True, allow_dirs=False)
        config = dict(config or {})
        with Image.open(image) as opened:
            pil_image = opened.convert("RGB")
            pil_image.load()

        if method == "gemini":
            from core.gemini_tagger import GeminiTagger

            api_key = str(config.get("api_key", "") or "")
            if not api_key:
                raise ValueError("gemini api_key is required")
            tagger = GeminiTagger(
                api_key=api_key,
                base_url=str(config.get("base_url", "") or None) or None,
                proxy=str(config.get("proxy", "") or None) or None,
                model=str(config.get("model", "gemini-1.5-flash") or "gemini-1.5-flash"),
                safety_none=bool(config.get("safety_none", True)),
            )
            if config.get("prompt"):
                tagger.set_prompt(str(config.get("prompt", "")))
            if config.get("prefix_tags"):
                tagger.set_prefix_tags(str(config.get("prefix_tags", "")))
            caption = tagger.tag_image(pil_image)
            tags = split_tags(caption or "")
            return {"method": method, "caption": caption or "", "tags": tags, "scores": {}}

        from core.wd14_tagger import WD14Tagger

        tagger = WD14Tagger(model_name=str(config.get("model", "wd-convnext-v3") or "wd-convnext-v3"))
        ratings, tags = tagger.tag_image(
            pil_image,
            threshold=float(config.get("threshold", 0.35) or 0.35),
            character_threshold=float(config.get("character_threshold", 0.85) or 0.85),
            exclude_tags=list(config.get("exclude_tags", []) or []),
            replace_underscore=bool(config.get("replace_underscore", True)),
        )
        try:
            tagger.unload()
        except Exception:
            pass
        return {
            "method": method,
            "caption": join_tags(list(tags.keys())),
            "tags": list(tags.keys()),
            "scores": tags,
            "ratings": ratings,
        }

    def move_files(
        self,
        *,
        directory: str,
        image_paths: Sequence[str],
        target_dir: str,
        keep_relative_structure: bool = True,
    ) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        target_root = validate_path(target_dir, must_exist=False, allow_dirs=True, allow_files=False)
        target_root.mkdir(parents=True, exist_ok=True)
        moved = 0
        skipped = 0
        for raw_path in image_paths:
            image = validate_path(raw_path, must_exist=True, allow_files=True, allow_dirs=False)
            if keep_relative_structure:
                try:
                    relative_parent = image.parent.relative_to(dataset_dir)
                except ValueError:
                    relative_parent = Path()
            else:
                relative_parent = Path()
            destination_dir = target_root / relative_parent
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_image = destination_dir / image.name
            caption_path, caption_exists = _find_existing_caption_path(image)
            shutil.move(str(image), str(destination_image))
            if caption_exists and caption_path.exists():
                shutil.move(str(caption_path), str(destination_dir / caption_path.name))
            moved += 1
        return {"moved_count": moved, "skipped_count": skipped, "target_dir": str(target_root)}

    def delete_files(
        self,
        *,
        directory: str,
        image_paths: Sequence[str],
    ) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        trash_root = dataset_dir / ".trash" / datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_root.mkdir(parents=True, exist_ok=True)
        return self.move_files(
            directory=str(dataset_dir),
            image_paths=image_paths,
            target_dir=str(trash_root),
            keep_relative_structure=True,
        ) | {"trash_dir": str(trash_root)}

    def capabilities(self) -> Dict[str, Any]:
        return {
            "mode": "cleanroom",
            "filter_logic": ["AND", "OR"],
            "selection_modes": ["all", "inclusive", "exclusive"],
            "has_caption_modes": ["any", "yes", "no"],
            "sort_by": ["name", "mtime", "tag_count", "caption_length"],
            "batch_actions": [
                "append_tags",
                "prepend_tags",
                "replace_tags",
                "remove_tags",
                "search_replace_tags",
                "search_replace_caption",
                "sort_tags",
                "dedupe_tags",
                "set_caption",
            ],
            "caption_extensions": list(SUPPORTED_CAPTION_EXTENSIONS),
            "safe_delete_mode": "trash",
        }

    def _scan_dataset(self, dataset_dir: Path, *, recursive: bool, caption_extension: str) -> List[DatasetCaptionItem]:
        iterator = dataset_dir.rglob("*") if recursive else dataset_dir.glob("*")
        items: List[DatasetCaptionItem] = []
        for candidate in iterator:
            if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            caption_path, exists = _resolve_caption_path(candidate, caption_extension)
            items.append(
                DatasetCaptionItem(
                    image_path=candidate,
                    relative_path=_relative_to(candidate, dataset_dir),
                    caption_path=caption_path,
                    caption_exists=exists,
                    caption_text=_read_text(caption_path) if exists else "",
                    mtime=candidate.stat().st_mtime,
                )
            )
        return items

    def _filter_items(
        self,
        items: Sequence[DatasetCaptionItem],
        *,
        filename_query: str,
        caption_query: str,
        positive_tags: Sequence[str],
        positive_logic: str,
        negative_tags: Sequence[str],
        negative_logic: str,
        selected_paths: Sequence[str],
        selection_mode: str,
        has_caption: str,
    ) -> List[DatasetCaptionItem]:
        positive = {tag.strip().lower() for tag in positive_tags if tag and tag.strip()}
        negative = {tag.strip().lower() for tag in negative_tags if tag and tag.strip()}
        selected = {str(path).replace("\\", "/").lower() for path in selected_paths if path}
        filename_query = str(filename_query or "").strip().lower()
        caption_query = str(caption_query or "").strip().lower()
        results: List[DatasetCaptionItem] = []

        for item in items:
            filename_lower = item.relative_path.lower()
            caption_lower = item.caption_text.lower()
            tags_lower = {tag.lower() for tag in item.tags}

            if filename_query and filename_query not in filename_lower:
                continue
            if caption_query and caption_query not in caption_lower:
                continue
            if has_caption == "yes" and not item.caption_text:
                continue
            if has_caption == "no" and item.caption_text:
                continue
            if positive and not self._match_tag_logic(tags_lower, positive, positive_logic):
                continue
            if negative and self._match_tag_logic(tags_lower, negative, negative_logic):
                continue
            if selected:
                selected_match = filename_lower in selected or str(item.image_path).replace("\\", "/").lower() in selected
                if selection_mode == "inclusive" and not selected_match:
                    continue
                if selection_mode == "exclusive" and selected_match:
                    continue
            results.append(item)
        return results

    def _match_tag_logic(self, tags_lower: set[str], filter_tags: set[str], logic: str) -> bool:
        logic_value = str(logic or "OR").strip().upper()
        if not filter_tags:
            return True
        if logic_value == "AND":
            return filter_tags.issubset(tags_lower)
        return bool(tags_lower & filter_tags)

    def _sort_items(self, items: Sequence[DatasetCaptionItem], *, sort_by: str, sort_order: str) -> List[DatasetCaptionItem]:
        reverse = str(sort_order or "asc").strip().lower() == "desc"
        mode = str(sort_by or "name").strip().lower()
        if mode == "mtime":
            key = lambda item: (item.mtime, item.relative_path.lower())
        elif mode == "tag_count":
            key = lambda item: (len(item.tags), item.relative_path.lower())
        elif mode == "caption_length":
            key = lambda item: (len(item.caption_text), item.relative_path.lower())
        else:
            key = lambda item: item.relative_path.lower()
        return sorted(items, key=key, reverse=reverse)

    def _serialize_item(self, item: DatasetCaptionItem) -> Dict[str, Any]:
        return {
            "image_path": str(item.image_path),
            "relative_path": item.relative_path,
            "caption_path": str(item.caption_path),
            "caption": item.caption_text,
            "tags": item.tags,
            "has_caption": bool(item.caption_text),
            "caption_exists": item.caption_exists,
            "mtime": item.mtime,
            "tag_count": len(item.tags),
        }

    def _tag_counts(self, items: Sequence[DatasetCaptionItem]) -> Counter[str]:
        counter: Counter[str] = Counter()
        for item in items:
            counter.update(item.tags)
        return counter

    def _common_tags(self, items: Sequence[DatasetCaptionItem]) -> List[str]:
        if not items:
            return []
        common = {tag.lower(): tag for tag in items[0].tags}
        current = set(common.keys())
        for item in items[1:]:
            current &= {tag.lower() for tag in item.tags}
        resolved: List[str] = []
        for lowered in current:
            resolved.append(common.get(lowered, lowered))
        return sorted(resolved, key=str.lower)

    def _resolve_target_items(
        self,
        dataset_dir: Path,
        *,
        image_paths: Sequence[str],
        recursive: bool,
        caption_extension: str,
        filter_payload: Dict[str, Any],
    ) -> List[DatasetCaptionItem]:
        if image_paths:
            results: List[DatasetCaptionItem] = []
            for raw_path in image_paths:
                image = validate_path(raw_path, must_exist=True, allow_files=True, allow_dirs=False)
                caption_path, exists = _resolve_caption_path(image, caption_extension)
                results.append(
                    DatasetCaptionItem(
                        image_path=image,
                        relative_path=_relative_to(image, dataset_dir) if dataset_dir in image.parents or image.parent == dataset_dir else image.name,
                        caption_path=caption_path,
                        caption_exists=exists,
                        caption_text=_read_text(caption_path) if exists else "",
                        mtime=image.stat().st_mtime,
                    )
                )
            return results

        listing = self.load_dataset(
            str(dataset_dir),
            recursive=recursive,
            caption_extension=caption_extension,
            limit=1_000_000,
            offset=0,
            sort_by=str(filter_payload.get("sort_by", "name") or "name"),
            sort_order=str(filter_payload.get("sort_order", "asc") or "asc"),
            filename_query=str(filter_payload.get("filename_query", "") or ""),
            caption_query=str(filter_payload.get("caption_query", "") or ""),
            positive_tags=list(filter_payload.get("positive_tags", []) or []),
            positive_logic=str(filter_payload.get("positive_logic", "OR") or "OR"),
            negative_tags=list(filter_payload.get("negative_tags", []) or []),
            negative_logic=str(filter_payload.get("negative_logic", "OR") or "OR"),
            selected_paths=list(filter_payload.get("selected_paths", []) or []),
            selection_mode=str(filter_payload.get("selection_mode", "all") or "all"),
            has_caption=str(filter_payload.get("has_caption", "any") or "any"),
        )
        return [
            DatasetCaptionItem(
                image_path=Path(item["image_path"]),
                relative_path=str(item["relative_path"]),
                caption_path=Path(item["caption_path"]),
                caption_exists=bool(item["caption_exists"]),
                caption_text=str(item["caption"] or ""),
                mtime=float(item["mtime"]),
            )
            for item in listing["items"]
        ]

    def _create_backup_for_items(self, dataset_dir: Path, items: Sequence[DatasetCaptionItem]) -> str:
        backup_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = dataset_dir / ".backups" / backup_name
        backup_root.mkdir(parents=True, exist_ok=True)
        for item in items:
            if not item.caption_path.exists():
                continue
            try:
                relative = item.caption_path.relative_to(dataset_dir)
            except ValueError:
                relative = Path(item.caption_path.name)
            destination = backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.caption_path, destination)
        return backup_name

    def _apply_action(
        self,
        caption: str,
        *,
        action: str,
        params: Dict[str, Any],
        tag_counts: Counter[str],
    ) -> str:
        current_tags = split_tags(caption)
        action_name = str(action or "").strip().lower()
        if action_name == "set_caption":
            return str(params.get("caption", "") or "").strip()
        if action_name == "append_tags":
            return join_tags(current_tags + split_tags(str(params.get("tags", "") or "")))
        if action_name == "prepend_tags":
            return join_tags(split_tags(str(params.get("tags", "") or "")) + current_tags)
        if action_name == "replace_tags":
            search_tags = split_tags(str(params.get("search_tags", "") or ""))
            replace_tags = split_tags(str(params.get("replace_tags", "") or ""))
            replaced = current_tags[:]
            tags_to_remove: set[str] = set()
            mapping: Dict[str, str] = {}
            for idx, search in enumerate(search_tags):
                if idx < len(replace_tags):
                    replacement = replace_tags[idx]
                    if replacement:
                        mapping[search.lower()] = replacement
                    else:
                        tags_to_remove.add(search.lower())
                else:
                    tags_to_remove.add(search.lower())
            out: List[str] = []
            for tag in replaced:
                lowered = tag.lower()
                if lowered in tags_to_remove:
                    continue
                out.append(mapping.get(lowered, tag))
            if len(replace_tags) > len(search_tags):
                out.extend(replace_tags[len(search_tags) :])
            return join_tags(out)
        if action_name == "remove_tags":
            remove = {tag.lower() for tag in split_tags(str(params.get("tags", "") or ""))}
            return join_tags([tag for tag in current_tags if tag.lower() not in remove])
        if action_name == "search_replace_tags":
            return join_tags(
                self._search_replace_tags(
                    current_tags,
                    search_text=str(params.get("search_text", "") or ""),
                    replace_text=str(params.get("replace_text", "") or ""),
                    use_regex=bool(params.get("use_regex", False)),
                )
            )
        if action_name == "search_replace_caption":
            text = caption
            search_text = str(params.get("search_text", "") or "")
            replace_text = str(params.get("replace_text", "") or "")
            if not search_text:
                return text
            if bool(params.get("use_regex", False)):
                try:
                    return re.sub(search_text, replace_text, text)
                except re.error:
                    return text
            return text.replace(search_text, replace_text)
        if action_name == "sort_tags":
            mode = str(params.get("sort_by", "alpha") or "alpha").strip().lower()
            order = str(params.get("sort_order", "asc") or "asc").strip().lower()
            reverse = order == "desc"
            if mode == "frequency":
                sorted_tags = sorted(current_tags, key=lambda tag: (tag_counts.get(tag, 0), tag.lower()), reverse=reverse)
            elif mode == "length":
                sorted_tags = sorted(current_tags, key=lambda tag: (len(tag), tag.lower()), reverse=reverse)
            else:
                sorted_tags = sorted(current_tags, key=str.lower, reverse=reverse)
            return join_tags(sorted_tags)
        if action_name == "dedupe_tags":
            return join_tags(dedupe_tags(current_tags))
        raise ValueError(f"Unsupported batch action: {action}")

    def _search_replace_tags(
        self,
        tags: Sequence[str],
        *,
        search_text: str,
        replace_text: str,
        use_regex: bool,
    ) -> List[str]:
        if not search_text:
            return list(tags)
        results: List[str] = []
        for tag in tags:
            if use_regex:
                try:
                    results.append(re.sub(search_text, replace_text, tag))
                except re.error:
                    return list(tags)
            else:
                results.append(tag.replace(search_text, replace_text))
        return results
