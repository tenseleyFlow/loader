"""Filesystem safety helpers shared by file-oriented tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

MAX_READ_SIZE = 10 * 1024 * 1024
MAX_WRITE_SIZE = 10 * 1024 * 1024
MAX_BINARY_PROBE_SIZE = 8192


@dataclass(slots=True)
class StructuredPatchHunk:
    """Structured patch hunk for write/edit operations."""

    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    lines: list[str]

    def to_dict(self) -> dict[str, int | list[str]]:
        """Serialize the patch hunk for tool metadata."""
        return asdict(self)


def resolve_workspace_path(
    raw_path: str,
    *,
    workspace_root: Path | None,
    allow_missing: bool = False,
) -> Path:
    """Resolve a path and enforce the workspace boundary when configured."""

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        base_root = workspace_root or Path.cwd()
        candidate = base_root / candidate

    try:
        if allow_missing:
            resolved_parent = candidate.parent.resolve()
            resolved = resolved_parent / candidate.name
        else:
            resolved = candidate.resolve()
    except FileNotFoundError:
        if not allow_missing:
            raise
        resolved_parent = candidate.parent.resolve()
        resolved = resolved_parent / candidate.name

    if workspace_root is not None:
        resolved_root = workspace_root.expanduser().resolve()
        if not resolved.is_relative_to(resolved_root):
            raise PermissionError(
                f"path {resolved} escapes workspace boundary {resolved_root}"
            )

    return resolved


def detect_binary_file(path: Path) -> bool:
    """Detect binary content by scanning for NUL bytes near the start."""

    with path.open("rb") as handle:
        chunk = handle.read(MAX_BINARY_PROBE_SIZE)
    return b"\x00" in chunk


def ensure_safe_to_read(path: Path) -> None:
    """Raise if a path is too large or appears binary."""

    size = path.stat().st_size
    if size > MAX_READ_SIZE:
        raise ValueError(f"file is too large ({size} bytes, max {MAX_READ_SIZE} bytes)")
    if detect_binary_file(path):
        raise ValueError("file appears to be binary")


def ensure_safe_to_write(content: str) -> None:
    """Raise if write content exceeds the maximum size or appears binary."""

    encoded = content.encode("utf-8")
    size = len(encoded)
    if size > MAX_WRITE_SIZE:
        raise ValueError(
            f"content is too large ({size} bytes, max {MAX_WRITE_SIZE} bytes)"
        )
    if b"\x00" in encoded[:MAX_BINARY_PROBE_SIZE]:
        raise ValueError("content appears to be binary")


def make_structured_patch(original: str, updated: str) -> list[StructuredPatchHunk]:
    """Build a simple whole-file patch hunk."""

    lines: list[str] = []
    for line in original.splitlines():
        lines.append(f"-{line}")
    for line in updated.splitlines():
        lines.append(f"+{line}")

    return [
        StructuredPatchHunk(
            old_start=1,
            old_lines=len(original.splitlines()),
            new_start=1,
            new_lines=len(updated.splitlines()),
            lines=lines,
        )
    ]
