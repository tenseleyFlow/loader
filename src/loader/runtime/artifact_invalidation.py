"""Semantic artifact invalidation and recovery-strategy selection."""

from __future__ import annotations

import re
from enum import StrEnum

from .workflow_policy import ArtifactFreshness


class WorkflowRecoveryStrategy(StrEnum):
    """Next workflow move when persisted artifacts drift."""

    NONE = "none"
    CLARIFY_REENTRY = "clarify_reentry"
    PLAN_REFRESH = "plan_refresh"
    FULL_REPLAN = "full_replan"


class ArtifactInvalidationAssessor:
    """Assess whether workflow artifacts still match the current task state."""

    def assess(
        self,
        *,
        task_statement: str,
        clarify_text: str | None,
        implementation_text: str | None,
        verification_text: str | None,
        acceptance_criteria: list[str],
        touched_files: list[str],
        last_verification_result: str | None,
    ) -> ArtifactFreshness:
        """Return stale-artifact state and the recommended recovery strategy."""

        if not clarify_text and not implementation_text and not verification_text:
            return ArtifactFreshness()

        plan_text = f"{implementation_text or ''}\n{verification_text or ''}".lower()
        brief_text = (clarify_text or "").lower()
        reasons: list[str] = []
        reason_codes: list[str] = []

        unexpected_paths = [
            name
            for path in touched_files
            if (name := _path_name(path)) and name.lower() not in plan_text
        ]
        stale_plan = False
        stale_brief = False

        if unexpected_paths:
            stale_plan = True
            reason_codes.append("touched_files_outside_plan")
            reasons.append(
                "Touched files outside the current plan: "
                + ", ".join(dict.fromkeys(unexpected_paths))
            )

        uncovered_criteria = [
            item
            for item in acceptance_criteria
            if item.strip()
            and item.strip().lower() != task_statement.strip().lower()
            and "runtime verification evidence" not in item.strip().lower()
            and not _text_covers_requirement(plan_text, item)
        ]
        if uncovered_criteria:
            stale_plan = True
            reason_codes.append("acceptance_criteria_outside_plan")
            reasons.append(
                "Acceptance criteria are missing from the current plan: "
                + "; ".join(uncovered_criteria[:2])
            )

        if brief_text:
            brief_gaps = [
                item
                for item in acceptance_criteria
                if item.strip()
                and item.strip().lower() != task_statement.strip().lower()
                and "runtime verification evidence" not in item.strip().lower()
                and not _text_covers_requirement(brief_text, item)
            ]
            if brief_gaps and last_verification_result == "failed":
                stale_brief = True
                reason_codes.append("brief_missing_acceptance_scope")
                reasons.append(
                    "The clarify brief no longer captures the active acceptance criteria: "
                    + "; ".join(brief_gaps[:2])
                )

            out_of_brief_paths = [
                name for name in unexpected_paths if name.lower() not in brief_text
            ]
            if out_of_brief_paths:
                stale_brief = True
                reason_codes.append("touchpoints_outside_brief")
                reasons.append(
                    "The clarify brief no longer matches the touched files: "
                    + ", ".join(dict.fromkeys(out_of_brief_paths))
                )

            if not _text_covers_requirement(brief_text, task_statement):
                stale_brief = True
                reason_codes.append("task_drifted_beyond_brief")
                reasons.append(
                    "The clarify brief no longer reflects the current task framing."
                )

        recovery_strategy = WorkflowRecoveryStrategy.NONE
        if stale_brief and stale_plan:
            recovery_strategy = WorkflowRecoveryStrategy.FULL_REPLAN
        elif stale_brief:
            recovery_strategy = WorkflowRecoveryStrategy.CLARIFY_REENTRY
        elif stale_plan:
            recovery_strategy = WorkflowRecoveryStrategy.PLAN_REFRESH

        return ArtifactFreshness(
            stale_brief=stale_brief,
            stale_plan=stale_plan,
            reasons=list(dict.fromkeys(reasons)),
            reason_codes=list(dict.fromkeys(reason_codes)),
            recovery_strategy=recovery_strategy.value,
        )


def _path_name(path: str) -> str:
    normalized = str(path).strip()
    if not normalized:
        return ""
    return normalized.rsplit("/", maxsplit=1)[-1].strip()


def _text_covers_requirement(text: str, requirement: str) -> bool:
    normalized_text = text.lower()
    normalized_requirement = requirement.lower()
    if normalized_requirement in normalized_text:
        return True

    tokens = [
        token
        for token in re.findall(r"[a-z0-9_./-]+", normalized_requirement)
        if len(token) > 2 and token not in _STOP_WORDS
    ]
    if not tokens:
        return normalized_requirement.strip() in normalized_text
    matches = sum(1 for token in tokens if token in normalized_text)
    threshold = max(1, min(2, len(tokens)))
    return matches >= threshold


_STOP_WORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "into",
    "from",
    "without",
    "while",
    "when",
    "then",
    "must",
    "should",
    "exists",
}
