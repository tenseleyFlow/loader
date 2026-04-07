"""Completion-policy helpers for the typed runtime."""

from __future__ import annotations

class CompletionPolicy:
    """Owns final response cleanup."""

    def __init__(self, _context: object | None = None) -> None:
        """Keep the existing construction seam while the helper stays stateless."""

        self._context = _context

    @staticmethod
    def finalize_response_text(
        *,
        content: str,
        actions_taken: list[str],
    ) -> str:
        """Return the assistant response without a synthetic follow-up suffix."""

        _ = actions_taken
        return content
