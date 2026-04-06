"""Minimal runtime tracing primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeTraceEvent:
    """One structured runtime trace event."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeTracer:
    """Collects trace events for a turn."""

    events: list[RuntimeTraceEvent] = field(default_factory=list)

    def record(self, name: str, **data: Any) -> None:
        """Record one trace event."""

        self.events.append(RuntimeTraceEvent(name=name, data=data))
