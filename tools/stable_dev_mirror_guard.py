from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = "stable-dev-mirror-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("stable_dev_mirror_manifest.json")
TRIAGE_SCHEMA_VERSION = "stable-dev-drift-triage-v2"
DEFAULT_TRIAGE_PATH = Path(__file__).with_name("stable_dev_drift_triage.json")
VALID_TRIAGE_ACTIONS = (
    "keep_dev_owned",
    "keep_stable_owned",
    "restore_mirror_from_stable",
)
VALID_TRIAGE_VALIDATION_MODES = ("manual", "trainer_registry")
VALID_TRIAGE_OWNERS = ("stable", "dev")
TRIAGE_ACTION_TO_OWNER = {
    "keep_dev_owned": "dev",
    "keep_stable_owned": "stable",
    "restore_mirror_from_stable": "stable",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {payload.get('schema_version')!r}")
    return payload


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_triage(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != TRIAGE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported triage schema: {payload.get('schema_version')!r}")
    return payload


def _suffixes_from_manifest(payload: dict) -> tuple[str, ...]:
    suffixes = payload.get("include_suffixes") or [".py"]
    normalized = []
    for item in suffixes:
        text = str(item).strip()
        if not text:
            continue
        normalized.append(text if text.startswith(".") else f".{text}")
    return tuple(normalized or [".py"])


def _collect_files(root: Path, suffixes: Iterable[str]) -> dict[str, Path]:
    suffix_set = {item.lower() for item in suffixes}
    results: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffix_set:
            continue
        results[path.relative_to(root).as_posix()] = path
    return results


def _compare_roots(source_root: Path, target_root: Path, suffixes: Iterable[str]) -> dict[str, list[str]]:
    source_files = _collect_files(source_root, suffixes)
    target_files = _collect_files(target_root, suffixes)

    identical: list[str] = []
    drift: list[str] = []
    only_source: list[str] = []
    only_target: list[str] = []

    for relative_path in sorted(set(source_files) | set(target_files)):
        source_path = source_files.get(relative_path)
        target_path = target_files.get(relative_path)
        if source_path and target_path:
            if _sha256(source_path) == _sha256(target_path):
                identical.append(relative_path)
            else:
                drift.append(relative_path)
        elif source_path:
            only_source.append(relative_path)
        else:
            only_target.append(relative_path)

    return {
        "identical": identical,
        "drift": drift,
        "only_source": only_source,
        "only_target": only_target,
    }


def _matches_any(relative_path: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in patterns)


def _selected_paths(paths: Iterable[str], patterns: list[str] | None) -> list[str]:
    return [path for path in sorted(paths) if _matches_any(path, patterns)]


def _normalize_repo_relative_path(value: str | Path) -> str:
    normalized = Path(str(value or "")).as_posix().strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _split_scripts_owner_and_relative(value: str | Path) -> tuple[str | None, str]:
    normalized = _normalize_repo_relative_path(value)
    for owner in VALID_TRIAGE_OWNERS:
        prefix = f"scripts/{owner}/"
        if normalized.startswith(prefix):
            return owner, normalized[len(prefix) :]
    return None, normalized


def _load_trainer_registry_module():
    repo_root = _repo_root()
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    from mikazuki.utils import trainer_registry

    return trainer_registry


def _validate_trainer_registry_entry(
    trainer_registry,
    path: str,
    expected_owner: str,
    expected_training_types: list[str],
) -> list[str]:
    errors: list[str] = []
    seen_training_types: set[str] = set()

    for raw_training_type in expected_training_types:
        training_type = str(raw_training_type or "").strip().lower()
        if not training_type:
            errors.append(f"{path}: expected_training_types contains a blank value")
            continue
        if training_type in seen_training_types:
            errors.append(f"{path}: duplicate expected training type {training_type!r}")
            continue
        seen_training_types.add(training_type)

        definition = trainer_registry.TRAINER_REGISTRY.get(training_type)
        if definition is None:
            errors.append(f"{path}: unknown training type {training_type!r}")
            continue

        candidate_paths: list[str] = []
        for candidate in (
            trainer_registry.get_trainer_file_for_training_type(training_type),
            definition.trainer_file,
        ):
            candidate_text = str(candidate or "").strip()
            if candidate_text and candidate_text not in candidate_paths:
                candidate_paths.append(candidate_text)

        resolved_targets: list[str] = []
        for candidate_path in candidate_paths:
            owner, relative_path = _split_scripts_owner_and_relative(candidate_path)
            resolved_targets.append(f"{owner or '?'}:{relative_path}")
            if owner == expected_owner and relative_path == path:
                break
        else:
            rendered_targets = ", ".join(resolved_targets) or "<missing>"
            errors.append(
                f"{path}: training type {training_type!r} resolves to {rendered_targets}, "
                f"expected scripts/{expected_owner}/{path}"
            )

    return errors


def _resolve_roots(payload: dict) -> tuple[Path, Path]:
    root = _repo_root()
    source_root = (root / str(payload["source_root"])).resolve()
    target_root = (root / str(payload["target_root"])).resolve()
    return source_root, target_root


def _extract_python_public_surface(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    public_functions: list[str] = []
    exported_names: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            public_functions.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        value = None
                    if isinstance(value, (list, tuple)):
                        exported_names = [str(item) for item in value]
                    break

    return {
        "public_functions": sorted(dict.fromkeys(public_functions)),
        "__all__": list(dict.fromkeys(exported_names)),
    }


def command_exports(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    relative_paths = args.path or [
        "library/train_config_util.py",
        "library/train_util.py",
        "library/gen_img_diffusers_input_util.py",
        "library/gen_img_diffusers_batch_util.py",
        "library/gen_img_sdxl_input_util.py",
        "library/gen_img_sdxl_batch_util.py",
        "library/gen_img_sdxl_main_loop_util.py",
    ]
    check_all = not args.functions_only
    check_functions = not args.all_only

    issues: list[str] = []
    for relative_path in relative_paths:
        stable_path = repo_root / "scripts" / "stable" / relative_path
        dev_path = repo_root / "scripts" / "dev" / relative_path

        if not stable_path.exists():
            issues.append(f"{relative_path}: missing scripts/stable copy")
            continue
        if not dev_path.exists():
            issues.append(f"{relative_path}: missing scripts/dev copy")
            continue

        stable_surface = _extract_python_public_surface(stable_path)
        dev_surface = _extract_python_public_surface(dev_path)

        if check_all and stable_surface["__all__"] != dev_surface["__all__"]:
            issues.append(
                f"{relative_path}: __all__ mismatch\n"
                f"  stable={stable_surface['__all__']}\n"
                f"  dev={dev_surface['__all__']}"
            )

        if check_functions and stable_surface["public_functions"] != dev_surface["public_functions"]:
            issues.append(
                f"{relative_path}: public function mismatch\n"
                f"  stable={stable_surface['public_functions']}\n"
                f"  dev={dev_surface['public_functions']}"
            )

    if issues:
        print("[export_surface_issues]")
        for item in issues:
            print(item)
        print(f"FAILED: found {len(issues)} export surface issue(s).")
        return 1

    print(f"OK: checked {len(relative_paths)} mirrored module export surfaces.")
    return 0


def command_stats(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    source_root, target_root = _resolve_roots(manifest)
    suffixes = _suffixes_from_manifest(manifest)
    result = _compare_roots(source_root, target_root, suffixes)

    print(f"source_root: {source_root}")
    print(f"target_root: {target_root}")
    print(f"include_suffixes: {', '.join(suffixes)}")
    print(f"identical: {len(result['identical'])}")
    print(f"drift: {len(result['drift'])}")
    print(f"only_source: {len(result['only_source'])}")
    print(f"only_target: {len(result['only_target'])}")

    if args.show:
        for bucket_name in ("drift", "only_source", "only_target"):
            items = result[bucket_name][: max(0, args.limit)]
            if not items:
                continue
            print(f"\n[{bucket_name}]")
            for item in items:
                print(item)
    return 0


def command_scan(args: argparse.Namespace) -> int:
    source_root = (_repo_root() / "scripts" / "stable").resolve()
    target_root = (_repo_root() / "scripts" / "dev").resolve()
    suffixes = (".py",)
    result = _compare_roots(source_root, target_root, suffixes)

    print(f"identical mirror candidates: {len(result['identical'])}")
    print(f"same-path drift files: {len(result['drift'])}")
    print(f"stable-only files: {len(result['only_source'])}")
    print(f"dev-only files: {len(result['only_target'])}")

    if args.write_manifest:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "source_root": "scripts/stable",
            "target_root": "scripts/dev",
            "include_suffixes": [".py"],
            "mirrors": result["identical"],
        }
        _write_manifest(args.manifest, payload)
        print(f"wrote manifest: {args.manifest}")

    if args.show:
        for item in result["identical"][: max(0, args.limit)]:
            print(item)
    return 0


def command_check(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    source_root, target_root = _resolve_roots(manifest)
    mirror_paths = _selected_paths(manifest.get("mirrors", []), args.match)

    missing_source: list[str] = []
    missing_target: list[str] = []
    drift: list[str] = []

    for relative_path in mirror_paths:
        source_path = source_root / relative_path
        target_path = target_root / relative_path
        if not source_path.exists():
            missing_source.append(relative_path)
            continue
        if not target_path.exists():
            missing_target.append(relative_path)
            continue
        if _sha256(source_path) != _sha256(target_path):
            drift.append(relative_path)

    if not missing_source and not missing_target and not drift:
        print(f"OK: {len(mirror_paths)} mirrored files are in sync.")
        return 0

    if missing_source:
        print("[missing_source]")
        for item in missing_source:
            print(item)
    if missing_target:
        print("[missing_target]")
        for item in missing_target:
            print(item)
    if drift:
        print("[drift]")
        for item in drift:
            print(item)

    total_issues = len(missing_source) + len(missing_target) + len(drift)
    print(f"FAILED: found {total_issues} mirror issues across {len(mirror_paths)} tracked files.")
    return 1


def command_sync(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    source_root, target_root = _resolve_roots(manifest)
    mirror_paths = _selected_paths(manifest.get("mirrors", []), args.match)
    copied = 0

    for relative_path in mirror_paths:
        source_path = source_root / relative_path
        target_path = target_root / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file: {source_path}")
        if not args.dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        copied += 1
        action = "would copy" if args.dry_run else "copied"
        print(f"{action}: {relative_path}")

    if args.dry_run:
        print(f"DRY RUN: {copied} files would be synced from stable to dev.")
    else:
        print(f"DONE: synced {copied} files from stable to dev.")
    return 0


def command_triage(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    triage = _load_triage(args.triage)
    source_root, target_root = _resolve_roots(manifest)
    suffixes = _suffixes_from_manifest(manifest)
    result = _compare_roots(source_root, target_root, suffixes)
    live_drift = sorted(result["drift"])
    live_drift_set = set(live_drift)

    raw_entries = triage.get("entries") or []
    if not isinstance(raw_entries, list):
        raise ValueError("Triage payload must contain a list under `entries`.")

    path_to_entry: dict[str, dict] = {}
    duplicate_paths: list[str] = []
    invalid_entries: list[str] = []
    validation_errors: list[str] = []
    trainer_registry = None

    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            invalid_entries.append(f"entry[{index}] is not an object")
            continue

        path = str(entry.get("path", "") or "").strip()
        action = str(entry.get("recommended_action", "") or "").strip()
        reason = str(entry.get("reason", "") or "").strip()
        validation_mode = str(entry.get("validation_mode", "") or "").strip()
        expected_owner = str(entry.get("expected_owner", "") or "").strip()
        raw_expected_training_types = entry.get("expected_training_types")

        if not path:
            invalid_entries.append(f"entry[{index}] is missing path")
            continue
        if action not in VALID_TRIAGE_ACTIONS:
            invalid_entries.append(f"{path}: invalid action {action!r}")
            continue
        if not reason:
            invalid_entries.append(f"{path}: missing reason")
            continue
        if validation_mode not in VALID_TRIAGE_VALIDATION_MODES:
            invalid_entries.append(f"{path}: invalid validation_mode {validation_mode!r}")
            continue
        if expected_owner not in VALID_TRIAGE_OWNERS:
            invalid_entries.append(f"{path}: invalid expected_owner {expected_owner!r}")
            continue
        implied_owner = TRIAGE_ACTION_TO_OWNER.get(action)
        if implied_owner and expected_owner != implied_owner:
            invalid_entries.append(
                f"{path}: expected_owner {expected_owner!r} does not match recommended_action {action!r}"
            )
            continue
        if path in path_to_entry:
            duplicate_paths.append(path)
            continue

        expected_training_types: list[str] = []
        if validation_mode == "trainer_registry":
            if action == "restore_mirror_from_stable":
                invalid_entries.append(
                    f"{path}: trainer_registry validation cannot be paired with {action!r}"
                )
                continue
            if not isinstance(raw_expected_training_types, list) or not raw_expected_training_types:
                invalid_entries.append(f"{path}: trainer_registry validation requires expected_training_types")
                continue
            expected_training_types = [str(item or "").strip() for item in raw_expected_training_types]
            if not all(expected_training_types):
                invalid_entries.append(f"{path}: expected_training_types contains blank values")
                continue
            if trainer_registry is None:
                trainer_registry = _load_trainer_registry_module()
            validation_errors.extend(
                _validate_trainer_registry_entry(
                    trainer_registry=trainer_registry,
                    path=path,
                    expected_owner=expected_owner,
                    expected_training_types=expected_training_types,
                )
            )
        elif raw_expected_training_types not in (None, []):
            invalid_entries.append(f"{path}: manual validation entries should not define expected_training_types")
            continue

        path_to_entry[path] = {
            "path": path,
            "recommended_action": action,
            "reason": reason,
            "validation_mode": validation_mode,
            "expected_owner": expected_owner,
            "expected_training_types": expected_training_types,
        }

    triaged_live_paths = sorted(live_drift_set & set(path_to_entry))
    untriaged_live_paths = sorted(live_drift_set - set(path_to_entry))
    stale_triage_paths = sorted(set(path_to_entry) - live_drift_set)

    counts = {action: 0 for action in VALID_TRIAGE_ACTIONS}
    validation_mode_counts = {mode: 0 for mode in VALID_TRIAGE_VALIDATION_MODES}
    for path in triaged_live_paths:
        counts[path_to_entry[path]["recommended_action"]] += 1
        validation_mode_counts[path_to_entry[path]["validation_mode"]] += 1

    print(f"source_root: {source_root}")
    print(f"target_root: {target_root}")
    print(f"live_drift: {len(live_drift)}")
    for action in VALID_TRIAGE_ACTIONS:
        print(f"{action}: {counts[action]}")
    for validation_mode in VALID_TRIAGE_VALIDATION_MODES:
        print(f"{validation_mode}: {validation_mode_counts[validation_mode]}")
    print(f"untriaged_live_drift: {len(untriaged_live_paths)}")
    print(f"stale_triage_entries: {len(stale_triage_paths)}")
    print(f"validation_errors: {len(validation_errors)}")

    if args.show:
        for action in VALID_TRIAGE_ACTIONS:
            grouped_paths = [
                path
                for path in triaged_live_paths
                if path_to_entry[path]["recommended_action"] == action
            ]
            if not grouped_paths:
                continue
            print(f"\n[{action}]")
            for path in grouped_paths[: max(0, args.limit)]:
                entry = path_to_entry[path]
                print(
                    f"{path} :: owner={entry['expected_owner']} :: mode={entry['validation_mode']} :: {entry['reason']}"
                )

    if invalid_entries:
        print("\n[invalid_entries]")
        for item in invalid_entries[: max(0, args.limit)]:
            print(item)

    if validation_errors:
        print("\n[validation_errors]")
        for item in validation_errors[: max(0, args.limit)]:
            print(item)

    if duplicate_paths:
        print("\n[duplicate_paths]")
        for item in duplicate_paths[: max(0, args.limit)]:
            print(item)

    if untriaged_live_paths:
        print("\n[untriaged_live_drift]")
        for item in untriaged_live_paths[: max(0, args.limit)]:
            print(item)

    if stale_triage_paths:
        print("\n[stale_triage_entries]")
        for item in stale_triage_paths[: max(0, args.limit)]:
            print(item)

    if invalid_entries or validation_errors or duplicate_paths or untriaged_live_paths or stale_triage_paths:
        issue_count = (
            len(invalid_entries)
            + len(validation_errors)
            + len(duplicate_paths)
            + len(untriaged_live_paths)
            + len(stale_triage_paths)
        )
        print(f"FAILED: found {issue_count} triage issue(s).")
        return 1

    print(f"OK: all {len(live_drift)} live drift files are triaged.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guard and maintain mirrored Python files between scripts/stable and scripts/dev.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser("stats", help="Show live same-path stats between scripts/stable and scripts/dev.")
    stats_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    stats_parser.add_argument("--show", action="store_true", help="Print drift and one-sided samples.")
    stats_parser.add_argument("--limit", type=int, default=20, help="Maximum sample lines to print per bucket.")
    stats_parser.set_defaults(func=command_stats)

    scan_parser = subparsers.add_parser("scan", help="Discover current same-path mirror candidates.")
    scan_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    scan_parser.add_argument("--write-manifest", action="store_true", help="Overwrite the manifest with current identical same-path files.")
    scan_parser.add_argument("--show", action="store_true", help="Print candidate mirror paths.")
    scan_parser.add_argument("--limit", type=int, default=50, help="Maximum candidate paths to print.")
    scan_parser.set_defaults(func=command_scan)

    check_parser = subparsers.add_parser("check", help="Check tracked mirror files for drift or missing copies.")
    check_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    check_parser.add_argument("--match", action="append", default=[], help="Optional fnmatch pattern to limit tracked paths.")
    check_parser.set_defaults(func=command_check)

    triage_parser = subparsers.add_parser("triage", help="Validate and summarize the manual drift triage for same-path drift files.")
    triage_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    triage_parser.add_argument("--triage", type=Path, default=DEFAULT_TRIAGE_PATH)
    triage_parser.add_argument("--show", action="store_true", help="Print triaged live drift grouped by action.")
    triage_parser.add_argument("--limit", type=int, default=200, help="Maximum lines to print per action or issue bucket.")
    triage_parser.set_defaults(func=command_triage)

    sync_parser = subparsers.add_parser("sync", help="Copy tracked mirror files from scripts/stable to scripts/dev.")
    sync_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    sync_parser.add_argument("--match", action="append", default=[], help="Optional fnmatch pattern to limit tracked paths.")
    sync_parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without changing files.")
    sync_parser.set_defaults(func=command_sync)

    exports_parser = subparsers.add_parser(
        "exports",
        help="Check mirrored Python modules for matching __all__ and top-level public function names.",
    )
    exports_parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Repo-relative path under scripts/stable|dev to check, for example library/train_util.py",
    )
    exports_parser.add_argument(
        "--all-only",
        action="store_true",
        help="Only compare __all__ exports, not top-level public function names.",
    )
    exports_parser.add_argument(
        "--functions-only",
        action="store_true",
        help="Only compare top-level public function names, not __all__ exports.",
    )
    exports_parser.set_defaults(func=command_exports)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
