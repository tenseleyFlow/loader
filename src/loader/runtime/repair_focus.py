"""Shared helpers for extracting and enforcing active repair focus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from os import sep
from pathlib import Path

from ..llm.base import Message


@dataclass(frozen=True)
class ActiveRepairContext:
    """Concrete repair focus extracted from recent verification feedback."""

    artifact_path: str
    repair_lines: list[str]
    allowed_paths: tuple[str, ...]
    allowed_roots: tuple[str, ...]


def extract_active_repair_context(
    messages: list[Message],
) -> ActiveRepairContext | None:
    """Return the most recent concrete repair target from session history."""

    for message in reversed(messages):
        content = str(getattr(message, "content", "") or "")
        if "Repair focus:" not in content:
            continue

        repair_lines: list[str] = []
        artifact_path = ""
        absolute_paths: list[str] = []
        capture = False
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not capture:
                if line == "Repair focus:":
                    capture = True
                continue
            if not line:
                if repair_lines:
                    break
                continue
            if not line.startswith("- "):
                if repair_lines:
                    break
                continue

            repair_lines.append(line)
            if not artifact_path:
                match = re.search(r"Immediate next step: edit `([^`]+)`", line)
                if match:
                    artifact_path = normalize_repair_path(match.group(1))

            for candidate in re.findall(r"`([^`]+)`", line):
                if not candidate.startswith(("/", "~")):
                    continue
                normalized = normalize_repair_path(candidate)
                if normalized not in absolute_paths:
                    absolute_paths.append(normalized)

        if repair_lines:
            if artifact_path:
                if artifact_path not in absolute_paths:
                    absolute_paths.insert(0, artifact_path)
            allowed_paths = tuple(
                sorted(
                    absolute_paths,
                    key=lambda item: (not Path(item).exists(), item),
                )
            )
            allowed_roots = _collapse_roots(_path_roots(set(absolute_paths)))
            return ActiveRepairContext(
                artifact_path=artifact_path,
                repair_lines=repair_lines,
                allowed_paths=allowed_paths,
                allowed_roots=allowed_roots,
            )
    return None


def path_within_allowed_roots(path: str, allowed_roots: tuple[str, ...]) -> bool:
    """Return whether the normalized path stays within the repair artifact set."""

    normalized = normalize_repair_path(path)
    normalized_roots = tuple(
        normalize_repair_path(root) for root in allowed_roots if str(root).strip()
    )
    return any(
        normalized == root or normalized.startswith(f"{root}{sep}")
        for root in normalized_roots
    )


def path_matches_allowed_paths(path: str, allowed_paths: tuple[str, ...]) -> bool:
    """Return whether the normalized path matches one concrete repair file."""

    normalized = normalize_repair_path(path)
    normalized_paths = {
        normalize_repair_path(candidate) for candidate in allowed_paths if str(candidate).strip()
    }
    return normalized in normalized_paths


def normalize_repair_path(raw_path: str) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return str(Path(text).expanduser())


def _path_roots(paths: set[str]) -> set[str]:
    roots: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        roots.add(str(path.parent))
    return roots


def _collapse_roots(roots: set[str]) -> tuple[str, ...]:
    collapsed: list[str] = []
    for root in sorted(roots, key=lambda item: (len(item), item)):
        if any(root == candidate or root.startswith(f"{candidate}{sep}") for candidate in collapsed):
            continue
        collapsed.append(root)
    return tuple(collapsed)
