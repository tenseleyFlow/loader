"""Tests for compatibility export boundaries."""

from __future__ import annotations

import re
from pathlib import Path


def test_runtime_code_does_not_import_agent_compatibility_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "loader"
    pattern = re.compile(
        r"(from\s+loader\.agent\.(reasoning|safeguards)\s+import)"
        r"|(import\s+loader\.agent\.(reasoning|safeguards))"
        r"|(from\s+\.\s*(reasoning|safeguards)\s+import)"
    )
    violations: list[str] = []

    for path in src_root.rglob("*.py"):
        relative_path = path.relative_to(src_root)
        if relative_path in {
            Path("agent/reasoning.py"),
            Path("agent/safeguards.py"),
        }:
            continue
        text = path.read_text()
        if pattern.search(text):
            violations.append(str(relative_path))

    assert violations == []
