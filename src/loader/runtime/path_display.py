"""Helpers for consistent user-facing runtime path rendering."""

from __future__ import annotations

from pathlib import Path


def display_runtime_path(path_value: str | Path) -> str:
    """Render a stable user-facing path for prompts and steering messages.

    Paths under the active HOME are shown as `~/...` to avoid leaking
    platform-specific symlink expansions such as `/private/tmp/...`.
    Other paths keep the current normalized absolute rendering.
    """

    path = Path(path_value).expanduser()
    resolved = path.resolve(strict=False)
    home = Path.home().expanduser().resolve(strict=False)
    try:
        relative = resolved.relative_to(home)
    except ValueError:
        return str(resolved)
    if not relative.parts:
        return "~"
    return f"~/{relative.as_posix()}"
