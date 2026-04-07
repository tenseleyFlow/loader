"""Intent-aware clarify-slot assessment and question selection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class ClarifySlot(StrEnum):
    """Requirement slots clarify mode tries to lock down before execution."""

    DESIRED_OUTCOME = "desired_outcome"
    NON_GOALS = "non_goals"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    CONSTRAINTS = "constraints"
    DECISION_BOUNDARIES = "decision_boundaries"
    LIKELY_TOUCHPOINTS = "likely_touchpoints"


_DEFAULT_SLOT_ORDER = [
    ClarifySlot.DESIRED_OUTCOME,
    ClarifySlot.NON_GOALS,
    ClarifySlot.ACCEPTANCE_CRITERIA,
    ClarifySlot.CONSTRAINTS,
    ClarifySlot.DECISION_BOUNDARIES,
    ClarifySlot.LIKELY_TOUCHPOINTS,
]

_SLOT_LABELS = {
    ClarifySlot.DESIRED_OUTCOME: "desired outcome",
    ClarifySlot.NON_GOALS: "non-goals",
    ClarifySlot.ACCEPTANCE_CRITERIA: "acceptance criteria",
    ClarifySlot.CONSTRAINTS: "constraints",
    ClarifySlot.DECISION_BOUNDARIES: "decision boundaries",
    ClarifySlot.LIKELY_TOUCHPOINTS: "likely touchpoints",
}


@dataclass(slots=True)
class ClarifySnapshot:
    """Structured clarify brief state used for follow-up decisions."""

    task_statement: str
    explicit_sections: list[str] = field(default_factory=list)
    desired_outcome: list[str] = field(default_factory=list)
    non_goals: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    decision_boundaries: list[str] = field(default_factory=list)
    likely_touchpoints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClarifyAssessment:
    """Missing slots and the next clarify focus."""

    unresolved_slots: list[ClarifySlot] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    focus_slot: ClarifySlot | None = None


def assess_clarify_snapshot(
    *,
    task: str,
    answer: str,
    snapshot: ClarifySnapshot,
) -> ClarifyAssessment:
    """Determine which clarify slots remain unresolved after one round."""

    explicit = {item.strip() for item in snapshot.explicit_sections if item.strip()}
    unresolved_slots: list[ClarifySlot] = []
    unresolved_questions: list[str] = []
    normalized_answer = answer.strip()
    answer_is_short = len(re.findall(r"\w+", normalized_answer)) < 4
    answer_is_broad = _answer_uses_broad_language(normalized_answer)

    if not normalized_answer:
        unresolved_questions.append(
            "No answer was provided to the clarification question."
        )
    if answer_is_short and normalized_answer:
        unresolved_questions.append(
            "The answer is still too short to lock task boundaries."
        )

    if ClarifySlot.DESIRED_OUTCOME.value not in explicit:
        unresolved_slots.append(ClarifySlot.DESIRED_OUTCOME)
        unresolved_questions.append(
            "The desired outcome is still not explicit enough to guide execution."
        )
    if ClarifySlot.NON_GOALS.value not in explicit or any(
        "anything not confirmed" in item.lower() for item in snapshot.non_goals
    ):
        unresolved_slots.append(ClarifySlot.NON_GOALS)
        unresolved_questions.append(
            "Out-of-scope boundaries are still underspecified."
        )
    if ClarifySlot.ACCEPTANCE_CRITERIA.value not in explicit or _looks_generic(
        snapshot.acceptance_criteria,
        task,
    ):
        unresolved_slots.append(ClarifySlot.ACCEPTANCE_CRITERIA)
        unresolved_questions.append(
            "Success criteria are still too generic to define done-ness."
        )
    if ClarifySlot.CONSTRAINTS.value not in explicit:
        unresolved_slots.append(ClarifySlot.CONSTRAINTS)
        unresolved_questions.append(
            "Constraints are still too implicit for a safe implementation pass."
        )
    if ClarifySlot.DECISION_BOUNDARIES.value not in explicit:
        unresolved_slots.append(ClarifySlot.DECISION_BOUNDARIES)
        unresolved_questions.append(
            "Decision boundaries are still too fuzzy for autonomous execution."
        )
    if ClarifySlot.LIKELY_TOUCHPOINTS.value not in explicit and (
        _task_wants_repo_specific_scope(task) or answer_is_broad
    ):
        unresolved_slots.append(ClarifySlot.LIKELY_TOUCHPOINTS)
        unresolved_questions.append(
            "Concrete files or subsystems are still not pinned down."
        )

    ordered_slots = _prioritize_slots(
        unresolved_slots,
        answer_is_broad=answer_is_broad,
        answer_is_short=answer_is_short,
    )
    if answer_is_broad and not unresolved_questions:
        unresolved_questions.append(
            "The clarified scope still uses broad or ambiguous language."
        )
    return ClarifyAssessment(
        unresolved_slots=ordered_slots,
        unresolved_questions=list(dict.fromkeys(unresolved_questions)),
        focus_slot=ordered_slots[0] if ordered_slots else None,
    )


def build_clarify_question(task: str, focus_slot: ClarifySlot | str | None) -> str:
    """Render one targeted question for the current clarify focus slot."""

    slot = (
        focus_slot
        if isinstance(focus_slot, ClarifySlot)
        else ClarifySlot(focus_slot)
        if focus_slot
        else ClarifySlot.DESIRED_OUTCOME
    )
    prompts = {
        ClarifySlot.DESIRED_OUTCOME: (
            "What concrete outcome should this change achieve when it's done?"
        ),
        ClarifySlot.NON_GOALS: (
            "What should stay out of scope while I work on this task?"
        ),
        ClarifySlot.ACCEPTANCE_CRITERIA: (
            "How will you judge that this work is complete and correct?"
        ),
        ClarifySlot.CONSTRAINTS: (
            "What constraint or invariant must I preserve while making this change?"
        ),
        ClarifySlot.DECISION_BOUNDARIES: (
            "Which decisions can I make on my own, and which should I stop and confirm?"
        ),
        ClarifySlot.LIKELY_TOUCHPOINTS: (
            "Which file or subsystem should I focus on, and what should stay unchanged?"
        ),
    }
    prompt = prompts[slot]
    if slot == ClarifySlot.DESIRED_OUTCOME and task.strip():
        return f"For `{task}`, {prompt[0].lower() + prompt[1:]}"
    return prompt


def describe_clarify_slot(slot: ClarifySlot | str | None) -> str:
    """Render a friendly clarify-slot label."""

    if slot is None:
        return "general clarification"
    resolved = slot if isinstance(slot, ClarifySlot) else ClarifySlot(slot)
    return _SLOT_LABELS[resolved]


def _prioritize_slots(
    slots: list[ClarifySlot],
    *,
    answer_is_broad: bool,
    answer_is_short: bool,
) -> list[ClarifySlot]:
    ordered = [slot for slot in _DEFAULT_SLOT_ORDER if slot in slots]
    if answer_is_broad and ClarifySlot.LIKELY_TOUCHPOINTS in ordered:
        ordered.remove(ClarifySlot.LIKELY_TOUCHPOINTS)
        ordered.insert(0, ClarifySlot.LIKELY_TOUCHPOINTS)
    if (
        answer_is_short
        and not answer_is_broad
        and ClarifySlot.DESIRED_OUTCOME in ordered
    ):
        ordered.remove(ClarifySlot.DESIRED_OUTCOME)
        ordered.insert(0, ClarifySlot.DESIRED_OUTCOME)
    return ordered


def _answer_uses_broad_language(answer: str) -> bool:
    lowered = answer.lower()
    if not lowered:
        return False
    return any(
        phrase in lowered
        for phrase in (
            "better",
            "nicer",
            "cleaner",
            "improve it",
            "fix it",
            "something",
            "somehow",
        )
    )


def _looks_generic(items: list[str], task: str) -> bool:
    if not items:
        return True
    lowered_items = [item.strip().lower() for item in items if item.strip()]
    lowered_task = task.strip().lower()
    return all(item == lowered_task or item in lowered_task for item in lowered_items)


def _task_wants_repo_specific_scope(task: str) -> bool:
    return bool(
        re.search(r"[./_\\-]", task)
        or any(
            token in task.lower()
            for token in ("file", "function", "class", "module", "runtime", "cli")
        )
    )
