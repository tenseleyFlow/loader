"""Runtime-owned typed surfaces shared with reasoning flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConfidenceLevel(Enum):
    """Confidence levels for actions."""

    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class Subtask:
    """A decomposed subtask with dependencies."""

    id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    verification: str = ""
    status: str = "pending"
    result: str = ""
    attempts: int = 0
    max_attempts: int = 2


@dataclass
class TaskDecomposition:
    """A decomposed task with ordered subtasks."""

    original_task: str
    subtasks: list[Subtask] = field(default_factory=list)
    current_index: int = 0
    rollback_points: list[int] = field(default_factory=list)

    def next_subtask(self) -> Subtask | None:
        """Get the next pending subtask that has all dependencies met."""

        completed_ids = {subtask.id for subtask in self.subtasks if subtask.status == "completed"}
        for subtask in self.subtasks:
            if subtask.status == "pending" and all(
                dependency in completed_ids for dependency in subtask.dependencies
            ):
                return subtask
        return None

    def mark_completed(self, subtask_id: str, result: str = "") -> None:
        """Mark a subtask as completed."""

        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                subtask.status = "completed"
                subtask.result = result
                break

    def mark_failed(self, subtask_id: str, error: str = "") -> None:
        """Mark a subtask as failed."""

        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                subtask.status = "failed"
                subtask.result = error
                subtask.attempts += 1
                break

    def can_retry(self, subtask_id: str) -> bool:
        """Check if a subtask can be retried."""

        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                return subtask.attempts < subtask.max_attempts
        return False

    def reset_for_retry(self, subtask_id: str) -> None:
        """Reset a subtask for retry."""

        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                subtask.status = "pending"
                break

    def progress_str(self) -> str:
        """Get progress string like '[2/5]'."""

        completed = sum(1 for subtask in self.subtasks if subtask.status == "completed")
        return f"[{completed}/{len(self.subtasks)}]"

    def is_complete(self) -> bool:
        """Check if all subtasks are completed."""

        return all(subtask.status in ("completed", "skipped") for subtask in self.subtasks)

    def has_failures(self) -> bool:
        """Check if any subtask has failed and exhausted retries."""

        return any(
            subtask.status == "failed" and subtask.attempts >= subtask.max_attempts
            for subtask in self.subtasks
        )

    def to_prompt(self) -> str:
        """Format decomposition for LLM prompt."""

        lines = [f"Task: {self.original_task}", "", "Subtasks:"]
        for index, subtask in enumerate(self.subtasks, 1):
            status_icon = {
                "pending": "○",
                "in_progress": "◐",
                "completed": "●",
                "failed": "✗",
                "skipped": "⊘",
            }.get(subtask.status, "?")
            dependencies = (
                f" (after: {', '.join(subtask.dependencies)})"
                if subtask.dependencies
                else ""
            )
            lines.append(f"  {status_icon} {index}. {subtask.description}{dependencies}")
            if subtask.verification:
                lines.append(f"      Verify: {subtask.verification}")
        return "\n".join(lines)


@dataclass
class SelfCritique:
    """Result of self-critique analysis."""

    original_response: str
    issues_found: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    should_revise: bool = False
    revised_response: str = ""
    revision_count: int = 0
    max_revisions: int = 2

    def can_revise(self) -> bool:
        """Check if we can do another revision."""

        return self.should_revise and self.revision_count < self.max_revisions


@dataclass
class ConfidenceAssessment:
    """Confidence assessment for an action."""

    action: str
    tool_name: str
    tool_args: dict[str, Any]
    level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    reasoning: str = ""
    risks: list[str] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)
    requires_verification: bool = False

    @property
    def score(self) -> int:
        """Get numeric score 1-5."""

        return self.level.value

    @property
    def is_low_confidence(self) -> bool:
        """Check if confidence is low enough to warrant caution."""

        return self.level.value <= ConfidenceLevel.LOW.value


@dataclass
class ActionVerification:
    """Verification result for a completed action."""

    tool_name: str
    tool_args: dict[str, Any]
    expected_outcome: str
    actual_result: str
    verified: bool = False
    verification_method: str = ""
    discrepancies: list[str] = field(default_factory=list)
    needs_correction: bool = False
    correction_suggestion: str = ""


@dataclass
class TaskCompletionCheck:
    """Result of checking if a task is complete."""

    original_task: str
    is_complete: bool = False
    accomplished: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)
    suggested_next_steps: list[str] = field(default_factory=list)
    continuation_prompt: str = ""
