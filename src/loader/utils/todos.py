"""Helpers for the active Loader todo store."""

from __future__ import annotations

from pathlib import Path


def active_todo_store_path(workspace_root: Path | str | None) -> Path:
    """Return the path to the active todo store for a workspace."""

    root = (
        Path(workspace_root).expanduser().resolve()
        if workspace_root is not None
        else Path.cwd()
    )
    return root / ".loader" / "todos" / "active.json"


def clear_active_todos(workspace_root: Path | str | None) -> Path:
    """Delete the active todo store if it exists."""

    path = active_todo_store_path(workspace_root)
    if path.exists():
        path.unlink()
    return path
