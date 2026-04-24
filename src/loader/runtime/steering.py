"""Steering-message payloads shared across runtime seams."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SteeringDirective:
    """One queued steering message plus persistence policy."""

    content: str
    persist_to_model: bool = True
