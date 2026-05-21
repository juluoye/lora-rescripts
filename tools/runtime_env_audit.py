from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


SUSPICIOUS_NAME_KEYWORDS = (
    "mikazuki",
    "sd-scripts",
    "lora-rescripts",
    "kohya",
)

METADATA_FILES = (
    "direct_url.json",
    "INSTALLER",
)


@dataclass
class Finding:
    level: str
    kind: str
    path: str
    message: str


@dataclass
class RuntimeAuditReport:
    runtime_root: str
    python_exe: str | None
    site_packages_dirs: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        if any(item.level == "high" for item in self.findings):
            return "high"
        if any(item.level == "medium" for item in self.findings):
            return "medium"
        if any(item.level == "low" for item in self.findings):
            return "low"
        return "none"

    def to_dict(self) -> dict:
        return {
            "runtime_root": self.runtime_root,
            "python_exe": self.python_exe,
            "site_packages_dirs": self.site_packages_dirs,
            "risk_level": self.risk_level,
            "findings": [asdict(item) for item in self.findings],
        }


def _iter_site_packages(runtime_root: Path) -> Iterable[Path]:
    candidates = (
        runtime_root / "Lib" / "site-packages",
        runtime_root / "lib" / "site-packages",
        runtime_root / "lib64" / "site-packages",
    )
    for path in candidates:
        if path.is_dir():
            yield path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _is_path_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _classify_reference(reference: str, runtime_root: Path) -> tuple[str, str] | None:
    text = reference.strip()
    if not text or text.startswith("#"):
        return None

    lowered = text.lower()
    if lowered.startswith(("git+", "http://", "https://")):
        return (
            "medium",
            f"Reference points to a VCS or remote URL: {text}",
        )

    raw_path = Path(text)
    if not raw_path.is_absolute():
        raw_path = (runtime_root / raw_path).resolve()
    else:
        raw_path = raw_path.resolve()

    if not _is_path_inside(raw_path, runtime_root):
        return (
            "high",
            f"Reference escapes runtime root: {raw_path}",
        )

    return None


def audit_runtime(runtime_root: Path) -> RuntimeAuditReport:
    python_exe = None
    for candidate in (
        runtime_root / "python.exe",
        runtime_root / "Scripts" / "python.exe",
        runtime_root / "bin" / "python",
    ):
        if candidate.exists():
            python_exe = str(candidate)
            break

    report = RuntimeAuditReport(
        runtime_root=str(runtime_root),
        python_exe=python_exe,
    )

    for site_packages in _iter_site_packages(runtime_root):
        report.site_packages_dirs.append(str(site_packages))

        for pth in sorted(site_packages.glob("*.pth")):
            for line in _read_text(pth).splitlines():
                finding = _classify_reference(line, runtime_root)
                if finding:
                    level, message = finding
                    report.findings.append(
                        Finding(
                            level=level,
                            kind="pth-reference",
                            path=str(pth),
                            message=message,
                        )
                    )

        for egg_link in sorted(site_packages.glob("*.egg-link")):
            for line in _read_text(egg_link).splitlines():
                finding = _classify_reference(line, runtime_root)
                if finding:
                    level, message = finding
                    report.findings.append(
                        Finding(
                            level=level,
                            kind="egg-link",
                            path=str(egg_link),
                            message=message,
                        )
                    )

        for meta_file_name in METADATA_FILES:
            for meta_file in sorted(site_packages.rglob(meta_file_name)):
                text = _read_text(meta_file)
                lowered = text.lower()
                if meta_file_name == "direct_url.json":
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        report.findings.append(
                            Finding(
                                level="medium",
                                kind="metadata-parse",
                                path=str(meta_file),
                                message="Failed to parse direct_url.json",
                            )
                        )
                        continue
                    url = str(payload.get("url", "")).strip()
                    if url:
                        finding = _classify_reference(url, runtime_root)
                        if finding:
                            level, message = finding
                            report.findings.append(
                                Finding(
                                    level=level,
                                    kind="direct-url",
                                    path=str(meta_file),
                                    message=message,
                                )
                            )
                elif any(keyword in lowered for keyword in SUSPICIOUS_NAME_KEYWORDS):
                    report.findings.append(
                        Finding(
                            level="medium",
                            kind="metadata-keyword",
                            path=str(meta_file),
                            message="Metadata text contains project-specific keywords",
                        )
                    )

        for entry in sorted(site_packages.iterdir()):
            name_lower = entry.name.lower()
            if any(keyword in name_lower for keyword in SUSPICIOUS_NAME_KEYWORDS):
                report.findings.append(
                    Finding(
                        level="medium",
                        kind="package-name",
                        path=str(entry),
                        message=f"Package name matches suspicious keyword: {entry.name}",
                    )
                )

    if not report.site_packages_dirs:
        report.findings.append(
            Finding(
                level="low",
                kind="missing-site-packages",
                path=str(runtime_root),
                message="No site-packages directory found",
            )
        )

    if python_exe is None:
        report.findings.append(
            Finding(
                level="low",
                kind="missing-python",
                path=str(runtime_root),
                message="No python executable found under runtime root",
            )
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Heuristic runtime contamination audit for local Python runtimes. "
            "This helps detect editable installs, external path references, and "
            "project-specific leftovers before using a runtime as a clean-room base."
        )
    )
    parser.add_argument("runtime_root", help="Runtime root directory to inspect")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).resolve()
    if not runtime_root.exists():
        print(f"Runtime root not found: {runtime_root}", file=sys.stderr)
        return 2

    report = audit_runtime(runtime_root)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"Runtime: {report.runtime_root}")
    print(f"Python: {report.python_exe or 'missing'}")
    print(f"Risk: {report.risk_level}")
    if report.site_packages_dirs:
        print("site-packages:")
        for path in report.site_packages_dirs:
            print(f"  - {path}")

    if not report.findings:
        print("Findings: none")
        return 0

    print("Findings:")
    for finding in report.findings:
        print(
            f"  - [{finding.level}] {finding.kind}: {finding.message} "
            f"({finding.path})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
