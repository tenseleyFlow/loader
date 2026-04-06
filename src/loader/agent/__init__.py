"""Agent system."""

from typing import Any

__all__ = ["Agent", "AgentConfig"]


def __getattr__(name: str) -> Any:
    """Lazily import the heavy agent loop on demand."""

    if name in {"Agent", "AgentConfig"}:
        from .loop import Agent, AgentConfig

        exports = {
            "Agent": Agent,
            "AgentConfig": AgentConfig,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
