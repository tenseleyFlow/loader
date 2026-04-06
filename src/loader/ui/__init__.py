"""Textual TUI for Loader."""

from typing import Any


def LoaderApp(*args: Any, **kwargs: Any):
    """Lazy wrapper for the Textual application."""
    from .app import LoaderApp as _LoaderApp

    return _LoaderApp(*args, **kwargs)


__all__ = ["LoaderApp"]
