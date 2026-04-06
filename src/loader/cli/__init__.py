"""CLI interface."""

from typing import Any


def main(*args: Any, **kwargs: Any):
    """Lazy wrapper for the Click entrypoint."""
    from .main import main as click_main

    return click_main(*args, **kwargs)


__all__ = ["main"]
