"""Structured runtime logging for diagnosing agent behavior."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOG_PATH = Path("/tmp/loader_runtime.log")


@dataclass(slots=True)
class LogEntry:
    """One structured log entry."""

    ts: float
    event: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_line(self) -> str:
        compact = {k: v for k, v in self.data.items() if v is not None}
        return json.dumps({"t": round(self.ts, 2), "e": self.event, **compact})


class RuntimeLogger:
    """Append-only structured logger for one session."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else _LOG_PATH
        self._start = time.monotonic()
        self._path.write_text("")  # truncate on session start

    def _elapsed(self) -> float:
        return time.monotonic() - self._start

    def log(self, event: str, **data: Any) -> None:
        entry = LogEntry(ts=self._elapsed(), event=event, data=data)
        try:
            with self._path.open("a") as f:
                f.write(entry.to_line() + "\n")
        except Exception:
            pass

    # ── convenience methods ──────────────────────────────────────────

    def turn_start(self, iteration: int, message_count: int, task: str) -> None:
        self.log(
            "turn.start",
            iteration=iteration,
            messages=message_count,
            task=task[:120],
        )

    def turn_response(
        self,
        iteration: int,
        content_len: int,
        tool_call_count: int,
        tool_names: list[str],
        usage: dict[str, int] | None = None,
    ) -> None:
        self.log(
            "turn.response",
            iteration=iteration,
            content_len=content_len,
            tool_calls=tool_call_count,
            tools=tool_names or None,
            usage=usage or None,
        )

    def turn_decision(
        self,
        iteration: int,
        action: str,
        continuation_count: int,
        consecutive_errors: int,
        reason: str | None = None,
    ) -> None:
        self.log(
            "turn.decision",
            iteration=iteration,
            action=action,
            continuations=continuation_count,
            errors=consecutive_errors,
            reason=reason,
        )

    def tool_exec(
        self,
        name: str,
        state: str,
        is_error: bool,
        result_preview: str,
        appended_to_session: bool,
    ) -> None:
        self.log(
            "tool.exec",
            name=name,
            state=state,
            error=is_error or None,
            result=result_preview[:200],
            in_session=appended_to_session,
        )

    def verification_gate(self, tool_name: str, should_continue: bool) -> None:
        self.log(
            "tool.verify_gate",
            tool=tool_name,
            continue_loop=should_continue,
        )

    def completion_check(
        self,
        stage: str,
        outcome: str,
        reason: str | None = None,
    ) -> None:
        self.log("completion", stage=stage, outcome=outcome, reason=reason)

    def session_context(self, message_count: int, roles: dict[str, int]) -> None:
        self.log("session.context", messages=message_count, roles=roles)

    def loop_exit(self, iterations: int, reason_code: str, reason: str) -> None:
        self.log(
            "loop.exit",
            iterations=iterations,
            reason_code=reason_code,
            reason=reason[:200],
        )


# Module-level singleton, initialized lazily by the runtime.
_logger: RuntimeLogger | None = None


def get_runtime_logger() -> RuntimeLogger:
    """Return the active runtime logger, creating one if needed."""
    global _logger
    if _logger is None:
        _logger = RuntimeLogger()
    return _logger


def reset_runtime_logger(path: Path | str | None = None) -> RuntimeLogger:
    """Create a fresh logger (call once per session start)."""
    global _logger
    _logger = RuntimeLogger(path)
    return _logger
