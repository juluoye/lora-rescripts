from __future__ import annotations

import argparse
import fnmatch
import subprocess
from collections import defaultdict
from pathlib import Path


DEFAULT_PATTERNS = (
    "build/**",
    "dist/**",
    "env/**",
    "python/**",
    "python_*/**",
    "python-*/**",
    "py310/**",
    "python_blackwell/**",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _git_ls_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    payload = result.stdout.decode("utf-8", errors="replace")
    return [item for item in payload.split("\0") if item]


def _match_patterns(paths: list[str], patterns: tuple[str, ...]) -> dict[str, list[str]]:
    matched: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                matched[pattern].append(path)
                break
    return matched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report tracked files that still live under generated/runtime noise directories.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Extra fnmatch pattern to audit. Defaults cover build/dist/env/python style directories.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print sample matching paths for each bucket.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum sample paths to print per bucket when --show is used.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    patterns = tuple(DEFAULT_PATTERNS + tuple(args.pattern))
    tracked_paths = _git_ls_files(repo_root)
    matched = _match_patterns(tracked_paths, patterns)

    total = 0
    for pattern in patterns:
        count = len(matched.get(pattern, []))
        if count == 0:
            continue
        total += count
        print(f"{pattern}: {count}")
        if args.show:
            for item in matched[pattern][: max(0, args.limit)]:
                print(f"  {item}")

    print(f"total_tracked_noise_files: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
