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


class ClarifyStage(StrEnum):
    """High-level interview stage for bounded clarify mode."""

    INTENT = "intent"
    BOUNDARIES = "boundaries"
    READINESS = "readiness"


class ClarifyPressureKind(StrEnum):
    """Which kind of pressure pass the next clarify round should apply."""

    EXAMPLE = "example"
    TRADEOFF = "tradeoff"
    ASSUMPTION = "assumption"


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
    stage: ClarifyStage = ClarifyStage.INTENT
    pressure_kind: ClarifyPressureKind | None = None
    pressure_pass_complete: bool = False
    missing_readiness_gates: list[str] = field(default_factory=list)


def assess_clarify_snapshot(
    *,
    task: str,
    answer: str,
    snapshot: ClarifySnapshot,
    round_index: int = 1,
    pressure_pass_complete: bool = False,
) -> ClarifyAssessment:
    """Determine which clarify slots remain unresolved after one round."""

    explicit = {item.strip() for item in snapshot.explicit_sections if item.strip()}
    unresolved_slots: list[ClarifySlot] = []
    unresolved_questions: list[str] = []
    normalized_answer = answer.strip()
    answer_is_short = len(re.findall(r"\w+", normalized_answer)) < 4
    answer_is_broad = _answer_uses_broad_language(normalized_answer)
    effective_pressure_pass_complete = (
        pressure_pass_complete or _answer_demonstrates_pressure_pass(normalized_answer)
    )
    missing_readiness_gates: list[str] = []

    non_goals_explicit = ClarifySlot.NON_GOALS.value in explicit and bool(
        [item for item in snapshot.non_goals if item.strip()]
    )
    decision_boundaries_explicit = ClarifySlot.DECISION_BOUNDARIES.value in explicit and bool(
        [item for item in snapshot.decision_boundaries if item.strip()]
    )

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
    if not non_goals_explicit or any(
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
    if not decision_boundaries_explicit:
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

    if not non_goals_explicit:
        missing_readiness_gates.append("non_goals")
    if not decision_boundaries_explicit:
        missing_readiness_gates.append("decision_boundaries")
    if round_index >= 2 and not effective_pressure_pass_complete:
        missing_readiness_gates.append("pressure_pass")

    pressure_kind = _choose_pressure_kind(
        round_index=round_index,
        answer_is_broad=answer_is_broad,
        missing_readiness_gates=missing_readiness_gates,
        pressure_pass_complete=effective_pressure_pass_complete,
        unresolved_slots=ordered_slots,
    )
    if pressure_kind == ClarifyPressureKind.EXAMPLE:
        unresolved_questions.append(
            "Loader still needs a concrete example or counterexample before planning."
        )
    elif pressure_kind == ClarifyPressureKind.TRADEOFF:
        unresolved_questions.append(
            "Loader still needs an explicit tradeoff or stop boundary before planning."
        )
    elif pressure_kind == ClarifyPressureKind.ASSUMPTION:
        unresolved_questions.append(
            "Loader still needs one challenged assumption before it should proceed."
        )

    stage = _resolve_stage(
        unresolved_slots=ordered_slots,
        missing_readiness_gates=missing_readiness_gates,
    )
    return ClarifyAssessment(
        unresolved_slots=ordered_slots,
        unresolved_questions=list(dict.fromkeys(unresolved_questions)),
        focus_slot=ordered_slots[0] if ordered_slots else None,
        stage=stage,
        pressure_kind=pressure_kind,
        pressure_pass_complete=effective_pressure_pass_complete,
        missing_readiness_gates=list(dict.fromkeys(missing_readiness_gates)),
    )


def build_clarify_question(
    task: str,
    focus_slot: ClarifySlot | str | None,
    pressure_kind: ClarifyPressureKind | str | None = None,
) -> str:
    """Render one targeted question for the current clarify focus slot."""

    slot = (
        focus_slot
        if isinstance(focus_slot, ClarifySlot)
        else ClarifySlot(focus_slot)
        if focus_slot
        else ClarifySlot.DESIRED_OUTCOME
    )
    pressure = (
        pressure_kind
        if isinstance(pressure_kind, ClarifyPressureKind)
        else ClarifyPressureKind(pressure_kind)
        if pressure_kind
        else None
    )

    if pressure == ClarifyPressureKind.EXAMPLE:
        prompts = {
            ClarifySlot.DESIRED_OUTCOME: (
                "What is one concrete example of the finished outcome, and one nearby "
                "result that should still count as out of scope?"
            ),
            ClarifySlot.NON_GOALS: (
                "What is one tempting broader change I should avoid even if it seems helpful?"
            ),
            ClarifySlot.ACCEPTANCE_CRITERIA: (
                "What concrete example would prove this is done, and what shortcut "
                "would still be wrong?"
            ),
            ClarifySlot.CONSTRAINTS: (
                "What is one concrete invariant I must preserve, and what change would violate it?"
            ),
            ClarifySlot.DECISION_BOUNDARIES: (
                "Give one example of a choice I may make alone and one example that "
                "should force me to stop and confirm."
            ),
            ClarifySlot.LIKELY_TOUCHPOINTS: (
                "Which file should change first, and which nearby file should I "
                "explicitly leave alone?"
            ),
        }
        return prompts[slot]

    if pressure == ClarifyPressureKind.TRADEOFF:
        prompts = {
            ClarifySlot.DESIRED_OUTCOME: (
                "What result matters most here, and what broader improvement should I "
                "still avoid chasing?"
            ),
            ClarifySlot.NON_GOALS: (
                "What should stay unchanged even if changing it would make the "
                "implementation easier?"
            ),
            ClarifySlot.ACCEPTANCE_CRITERIA: (
                "What outcome would count as success, and what tempting shortcut "
                "should still count as failure?"
            ),
            ClarifySlot.CONSTRAINTS: (
                "What must stay true even if it makes the change slower or less sweeping?"
            ),
            ClarifySlot.DECISION_BOUNDARIES: (
                "Which decision may I take on my own, and which one should I stop "
                "and confirm before proceeding?"
            ),
            ClarifySlot.LIKELY_TOUCHPOINTS: (
                "Which file should I focus on, and what file or surface should stay unchanged?"
            ),
        }
        return prompts[slot]

    if pressure == ClarifyPressureKind.ASSUMPTION:
        prompts = {
            ClarifySlot.DESIRED_OUTCOME: (
                "What assumption about the desired outcome am I most likely to get "
                "wrong if I act now?"
            ),
            ClarifySlot.NON_GOALS: (
                "What assumption about scope should I not make without checking first?"
            ),
            ClarifySlot.ACCEPTANCE_CRITERIA: (
                "What assumption about 'done' would be risky to make without your confirmation?"
            ),
            ClarifySlot.CONSTRAINTS: (
                "What assumption about constraints would be unsafe for me to guess?"
            ),
            ClarifySlot.DECISION_BOUNDARIES: (
                "What decision would be risky for me to assume I can make without checking?"
            ),
            ClarifySlot.LIKELY_TOUCHPOINTS: (
                "What assumption about the right touchpoint or file would be most "
                "dangerous if I guessed wrong?"
            ),
        }
        return prompts[slot]

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


def describe_clarify_stage(stage: ClarifyStage | str | None) -> str:
    """Render a friendly clarify-stage label."""

    if stage is None:
        return "general"
    resolved = stage if isinstance(stage, ClarifyStage) else ClarifyStage(stage)
    return resolved.value


def describe_clarify_pressure_kind(
    pressure_kind: ClarifyPressureKind | str | None,
) -> str:
    """Render a friendly pressure-pass label."""

    if pressure_kind is None:
        return "none"
    resolved = (
        pressure_kind
        if isinstance(pressure_kind, ClarifyPressureKind)
        else ClarifyPressureKind(pressure_kind)
    )
    return resolved.value


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


def _resolve_stage(
    *,
    unresolved_slots: list[ClarifySlot],
    missing_readiness_gates: list[str],
) -> ClarifyStage:
    if missing_readiness_gates:
        return ClarifyStage.READINESS
    if ClarifySlot.DESIRED_OUTCOME in unresolved_slots:
        return ClarifyStage.INTENT
    return ClarifyStage.BOUNDARIES


def _choose_pressure_kind(
    *,
    round_index: int,
    answer_is_broad: bool,
    missing_readiness_gates: list[str],
    pressure_pass_complete: bool,
    unresolved_slots: list[ClarifySlot],
) -> ClarifyPressureKind | None:
    if round_index < 2 or pressure_pass_complete or not unresolved_slots:
        return None
    if answer_is_broad:
        return ClarifyPressureKind.EXAMPLE
    if any(gate in {"non_goals", "decision_boundaries"} for gate in missing_readiness_gates):
        return ClarifyPressureKind.TRADEOFF
    return ClarifyPressureKind.ASSUMPTION


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
            "maybe",
            "around there",
        )
    )


def _answer_demonstrates_pressure_pass(answer: str) -> bool:
    lowered = answer.lower()
    if not lowered:
        return False
    return any(
        phrase in lowered
        for phrase in (
            "do not",
            "don't",
            "keep",
            "leave",
            "unchanged",
            "out of scope",
            "avoid",
            "only",
            "stop and ask",
            "confirm first",
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
