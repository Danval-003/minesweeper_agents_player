#!/usr/bin/env python3
"""Utility to bump minesweeper-env-rl versions in pyproject/__init__."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

PYPROJECT_PATTERN = re.compile(r'(version\s*=\s*")(?P<version>\d+\.\d+\.\d+)(")')
INIT_PATTERN = re.compile(r'(__version__\s*=\s*")(?P<version>\d+\.\d+\.\d+)(")')


@dataclass
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str) -> "Version":
        parts = raw.strip().split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid semantic version: {raw}")
        major, minor, patch = (int(part) for part in parts)
        return cls(major, minor, patch)

    def bump(self, release: str) -> "Version":
        if release == "major":
            return Version(self.major + 1, 0, 0)
        if release == "minor":
            return Version(self.major, self.minor + 1, 0)
        if release == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        raise ValueError(f"Unknown release type: {release}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _update_file(
    path: pathlib.Path,
    pattern: re.Pattern[str],
    new_version: str,
    append_template: str | None = None,
) -> None:
    text = path.read_text(encoding="utf-8")
    match = pattern.search(text)
    if not match:
        if append_template is None:
            raise RuntimeError(f"Could not find version declaration inside {path}")
        suffix = "" if text.endswith("\n") else "\n"
        text = text + suffix + append_template.format(version=new_version) + "\n"
        path.write_text(text, encoding="utf-8")
        return

    start, end = match.span("version")
    updated = text[:start] + new_version + text[end:]
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump minesweeper_env_rl package version.")
    parser.add_argument("--pyproject", type=pathlib.Path, default=pathlib.Path("minesweeper_env_rl/pyproject.toml"))
    parser.add_argument("--init-file", type=pathlib.Path, default=pathlib.Path("minesweeper_env_rl/__init__.py"))
    parser.add_argument("--release-type", required=True, choices=("major", "minor", "patch"))
    args = parser.parse_args()

    pyproject_text = args.pyproject.read_text(encoding="utf-8")
    pyproject_match = PYPROJECT_PATTERN.search(pyproject_text)
    if not pyproject_match:
        raise SystemExit("version key not found in pyproject.toml")

    current_version = Version.parse(pyproject_match.group("version"))
    bumped = current_version.bump(args.release_type)

    _update_file(args.pyproject, PYPROJECT_PATTERN, str(bumped))
    if args.init_file.exists():
        _update_file(
            args.init_file,
            INIT_PATTERN,
            str(bumped),
            append_template='__version__ = "{version}"',
        )

    print(bumped)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - utility script
        print(f"bump_version.py failed: {exc}", file=sys.stderr)
        raise
