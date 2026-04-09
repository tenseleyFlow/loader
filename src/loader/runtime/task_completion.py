"""Runtime-owned completion heuristics and continuation prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .dod import DefinitionOfDone
from .evidence_provenance import EvidenceProvenance, EvidenceProvenanceStatus
from .reasoning_types import TaskCompletionCheck
from .verification_observations import (
    VerificationObservation,
    VerificationObservationStatus,
)

_ACTION_VERBS = ("create", "write", "make", "edit", "fix", "add", "delete", "run")
_COMPLEX_INDICATORS = (
    "set up a project",
    "create a project",
    "build a complete",
    "scaffold",
    "initialize a new",
    "create a full",
    "implement a full",
    "develop a complete",
)
_SIMPLE_TASK_INDICATORS = (
    "create a file",
    "write a file",
    "make a file",
    "add a function",
    "edit the",
    "fix the",
    "update the",
    "read the",
    "show me",
    "list",
    "design a webpage",
    "create a webpage",
    "make a webpage",
    "create a page",
    "design a page",
    "create an html",
    "make an html",
    "write an html",
    "help me design",
    "create a simple",
    "make a simple",
    "write a simple",
)
_VERIFICATION_INDICATORS = ("and test", "and run", "and verify", "make sure it works")
_DEFLECTION_PHRASES = ("you can now", "you should", "you can run", "you can use")
_INFORMATIONAL_PREFIXES = (
    "explain ",
    "describe ",
    "summarize ",
    "compare ",
    "outline ",
    "review ",
    "analyze ",
    "what ",
    "how ",
    "why ",
    "which ",
    "who ",
    "where ",
    "when ",
)
_EXPLICIT_COMPLETIONS = {
    "done",
    "done.",
    "completed",
    "completed.",
    "all set",
    "all set.",
}
_INSTALL_HINTS = ("install", "dependencies", "set up project")
_NODE_HINTS = ("node", "npm")
_PYTHON_HINTS = ("python", "pip")
_IMPLEMENTATION_ITEM = "Complete the requested work"
_VERIFY_ITEM = "Collect verification evidence"
_ACTION_EVIDENCE = "showing the requested work was actually carried out"
_INSTALL_EVIDENCE = "showing dependencies or setup steps were completed"
_VERIFICATION_EVIDENCE = "showing the result was run or verified"
_COMPLEX_EVIDENCE = "showing the broader end-to-end implementation or setup was completed"

COMPLETION_CHECK_PROMPT = """Evaluate if this task has been FULLY completed.

Original task: {task}

Actions taken so far:
{actions}

Current response: {response}

IMPORTANT: Be strict about completion. A task is NOT complete if:
- Files were created but not tested/verified
- A project was scaffolded but not initialized (npm install, pip install, etc.)
- Code was written but not run/tested
- Setup was done but the result wasn't demonstrated

Respond in this exact JSON format:
{{
  "is_complete": true/false,
  "accomplished": ["What was done 1", "What was done 2"],
  "remaining": ["What still needs to be done 1", "What still needs to be done 2"],
  "next_steps": ["Immediate next action 1", "Immediate next action 2"],
  "reasoning": "Why the task is/isn't complete"
}}

Only output the JSON, no other text."""


@dataclass(slots=True)
class _FollowThroughFacts:
    """Structured runtime evidence for one completion check."""

    has_recorded_work: bool
    has_install_evidence: bool
    has_verification_evidence: bool
    has_failed_verification: bool
    has_pending_verification: bool
    has_planned_verification: bool
    has_stale_verification: bool
    verification_command: str | None
    pending_items: list[str]
    accomplished: list[str]


@dataclass(slots=True)
class CompletionAssessment:
    """Runtime-owned completion assessment with typed evidence provenance."""

    check: TaskCompletionCheck
    evidence_provenance: list[EvidenceProvenance] = field(default_factory=list)
    verification_observations: list[VerificationObservation] = field(default_factory=list)


def detect_premature_completion(
    task: str,
    response: str,
    actions_taken: list[str],
    *,
    dod: DefinitionOfDone | None = None,
) -> bool:
    """Heuristically detect when the assistant is stopping too early."""
    if not actions_taken and response.lower().strip() in _EXPLICIT_COMPLETIONS:
        return False
    return not assess_completion_follow_through(
        task=task,
        response=response,
        actions_taken=actions_taken,
        dod=dod,
    ).is_complete


def get_continuation_prompt(
    task: str,
    actions_taken: list[str],
    response: str,
    *,
    dod: DefinitionOfDone | None = None,
) -> str:
    """Generate a helpful follow-through prompt for incomplete tasks."""
    return assess_completion_follow_through(
        task=task,
        response=response,
        actions_taken=actions_taken,
        dod=dod,
    ).continuation_prompt


def assess_completion_follow_through_with_provenance(
    *,
    task: str,
    response: str,
    actions_taken: list[str],
    dod: DefinitionOfDone | None = None,
) -> CompletionAssessment:
    """Build a typed follow-through assessment for one candidate response."""

    task_lower = task.lower().strip()
    response_lower = response.lower().strip()
    action_types = _action_types(actions_taken)
    facts = _build_follow_through_facts(
        task_lower=task_lower,
        actions_taken=actions_taken,
        action_types=action_types,
        dod=dod,
    )
    informational = _is_informational_task(task_lower)
    complex_task = any(indicator in task_lower for indicator in _COMPLEX_INDICATORS)
    simple_task = any(indicator in task_lower for indicator in _SIMPLE_TASK_INDICATORS)
    requires_verification = any(
        indicator in task_lower for indicator in _VERIFICATION_INDICATORS
    )
    requires_install = any(indicator in task_lower for indicator in _INSTALL_HINTS)
    evidence_provenance = _observed_completion_provenance(
        task_lower=task_lower,
        response_lower=response_lower,
        actions_taken=actions_taken,
        facts=facts,
        informational=informational,
        requires_install=requires_install,
        requires_verification=requires_verification,
        dod=dod,
    )
    verification_observations = _observed_completion_verification(
        dod=dod,
        verification_command=facts.verification_command,
        requires_verification=requires_verification,
    )

    accomplished = list(facts.accomplished)
    required_evidence = _required_evidence(
        task_lower=task_lower,
        informational=informational,
        complex_task=complex_task,
        requires_verification=requires_verification,
        requires_install=requires_install,
    )
    missing_evidence: list[str] = []
    remaining: list[str] = []
    suggested_next_steps: list[str] = []

    if informational:
        return CompletionAssessment(
            check=TaskCompletionCheck(
                original_task=task,
                is_complete=bool(response.strip()),
                accomplished=accomplished,
                required_evidence=required_evidence,
                missing_evidence=[],
                remaining=[],
                suggested_next_steps=[],
                continuation_prompt=_format_continuation_prompt(
                    task=task,
                    missing_evidence=[],
                    suggested_next_steps=[],
                    action_count=len(actions_taken),
                ),
            ),
            evidence_provenance=evidence_provenance,
            verification_observations=verification_observations,
        )

    if facts.pending_items:
        next_item = facts.pending_items[0]
        _append_follow_through_gap(
            missing_evidence,
            remaining,
            suggested_next_steps,
            evidence=f"completion of tracked work items ({next_item})",
            remaining_item="Finish the remaining tracked work items",
            next_step=f"Complete the tracked item: {next_item}",
        )
        _append_unique_provenance(
            evidence_provenance,
            EvidenceProvenance(
                category="tracked_work",
                source="dod.pending_items",
                summary=f"tracked work item still pending: {next_item}",
                status=EvidenceProvenanceStatus.MISSING.value,
                subject=next_item,
            ),
        )

    if _requires_action(task_lower) and not facts.has_recorded_work:
        _append_follow_through_gap(
            missing_evidence,
            remaining,
            suggested_next_steps,
            evidence=_ACTION_EVIDENCE,
            remaining_item="Perform the requested work instead of stopping at intent or narration",
            next_step="Carry out the requested change or command now",
        )
        _append_unique_provenance(
            evidence_provenance,
            EvidenceProvenance(
                category="action",
                source="actions_taken",
                summary=(
                    "runtime history still lacked concrete work showing the "
                    "requested change or command happened"
                ),
                status=EvidenceProvenanceStatus.MISSING.value,
            ),
        )

    if requires_install and not facts.has_install_evidence:
        _append_follow_through_gap(
            missing_evidence,
            remaining,
            suggested_next_steps,
            evidence=_INSTALL_EVIDENCE,
            remaining_item="Install or initialize the required dependencies",
            next_step=_install_follow_up(task_lower, facts.verification_command),
        )
        _append_unique_provenance(
            evidence_provenance,
            EvidenceProvenance(
                category="install",
                source="actions_taken",
                summary="runtime history still lacked install or setup evidence",
                status=EvidenceProvenanceStatus.MISSING.value,
            ),
        )

    if requires_verification:
        if facts.has_failed_verification:
            _append_follow_through_gap(
                missing_evidence,
                remaining,
                suggested_next_steps,
                evidence=_failed_verification_evidence(facts.verification_command),
                remaining_item="Fix the failing verification result and rerun it",
                next_step=_verification_retry_step(facts.verification_command),
            )
            for entry in _verification_provenance(
                dod=dod,
                verification_command=facts.verification_command,
                status=EvidenceProvenanceStatus.CONTRADICTS,
            ):
                _append_unique_provenance(evidence_provenance, entry)
        elif facts.has_pending_verification:
            _append_follow_through_gap(
                missing_evidence,
                remaining,
                suggested_next_steps,
                evidence=_pending_verification_evidence(facts.verification_command),
                remaining_item="Let the active verification run finish and capture the result",
                next_step=_pending_verification_follow_up(facts.verification_command),
            )
            _append_unique_provenance(
                evidence_provenance,
                EvidenceProvenance(
                    category="verification",
                    source="dod.last_verification_result",
                    summary=(
                        "verification is already pending for "
                        f"`{facts.verification_command}`"
                        if facts.verification_command
                        else "verification is already pending"
                    ),
                    status=EvidenceProvenanceStatus.MISSING.value,
                    subject=facts.verification_command,
                ),
            )
        elif facts.has_stale_verification:
            _append_follow_through_gap(
                missing_evidence,
                remaining,
                suggested_next_steps,
                evidence=_stale_verification_evidence(facts.verification_command),
                remaining_item="Rerun verification after the implementation changed again",
                next_step=_stale_verification_follow_up(facts.verification_command),
            )
            _append_unique_provenance(
                evidence_provenance,
                EvidenceProvenance(
                    category="verification",
                    source="dod.last_verification_result",
                    summary=(
                        "previous verification became stale for "
                        f"`{facts.verification_command}` after new mutating work"
                        if facts.verification_command
                        else "previous verification became stale after new mutating work"
                    ),
                    status=EvidenceProvenanceStatus.MISSING.value,
                    subject=facts.verification_command,
                ),
            )
        elif facts.has_planned_verification:
            _append_follow_through_gap(
                missing_evidence,
                remaining,
                suggested_next_steps,
                evidence=_planned_verification_evidence(facts.verification_command),
                remaining_item="Run the planned verification before claiming completion",
                next_step=_planned_verification_follow_up(facts.verification_command),
            )
            _append_unique_provenance(
                evidence_provenance,
                EvidenceProvenance(
                    category="verification",
                    source="dod.verification_commands",
                    summary=(
                        "verification is planned for "
                        f"`{facts.verification_command}`"
                        if facts.verification_command
                        else "verification is planned"
                    ),
                    status=EvidenceProvenanceStatus.MISSING.value,
                    subject=facts.verification_command,
                ),
            )
        elif not facts.has_verification_evidence:
            _append_follow_through_gap(
                missing_evidence,
                remaining,
                suggested_next_steps,
                evidence=_missing_verification_evidence(facts.verification_command),
                remaining_item="Run the result and capture a concrete verification outcome",
                next_step=_verification_follow_up(
                    task_lower=task_lower,
                    verification_command=facts.verification_command,
                ),
            )
            _append_unique_provenance(
                evidence_provenance,
                EvidenceProvenance(
                    category="verification",
                    source=(
                        "dod.verification_commands"
                        if facts.verification_command
                        else "actions_taken"
                    ),
                    summary=(
                        "verification evidence was still missing for "
                        f"`{facts.verification_command}`"
                        if facts.verification_command
                        else "verification evidence was still missing"
                    ),
                    status=EvidenceProvenanceStatus.MISSING.value,
                    subject=facts.verification_command,
                ),
            )

    if complex_task and len(actions_taken) < 3 and not facts.pending_items:
        _append_follow_through_gap(
            missing_evidence,
            remaining,
            suggested_next_steps,
            evidence=_COMPLEX_EVIDENCE,
            remaining_item=(
                "Finish the larger end-to-end task instead of stopping after "
                "a partial step"
            ),
            next_step="Continue through the remaining setup or implementation steps",
        )
        _append_unique_provenance(
            evidence_provenance,
            EvidenceProvenance(
                category="task_scope",
                source="task_statement",
                summary="the runtime only saw partial progress for a broader end-to-end task",
                status=EvidenceProvenanceStatus.MISSING.value,
            ),
        )

    if (
        any(phrase in response_lower for phrase in _DEFLECTION_PHRASES)
        and not facts.has_recorded_work
        and not facts.has_verification_evidence
    ):
        _append_follow_through_gap(
            missing_evidence,
            remaining,
            suggested_next_steps,
            evidence="showing execution evidence rather than instructions handed back to the user",
            remaining_item=(
                "Perform the work yourself or state concretely what you "
                "already verified"
            ),
            next_step="Continue the task instead of handing the next step to the user",
        )
        _append_unique_provenance(
            evidence_provenance,
            EvidenceProvenance(
                category="response",
                source="assistant_response",
                summary=(
                    "the response deflected the next step back to the user "
                    "without runtime evidence"
                ),
                status=EvidenceProvenanceStatus.MISSING.value,
            ),
        )

    if "write" in action_types and actions_taken and simple_task:
        missing_evidence = [
            item
            for item in missing_evidence
            if item != _ACTION_EVIDENCE
        ]
        remaining = [
            item
            for item in remaining
            if item != "Perform the requested work instead of stopping at intent or narration"
        ]

    is_complete = not missing_evidence
    return CompletionAssessment(
        check=TaskCompletionCheck(
            original_task=task,
            is_complete=is_complete,
            accomplished=accomplished,
            required_evidence=required_evidence,
            missing_evidence=missing_evidence,
            remaining=remaining,
            suggested_next_steps=suggested_next_steps,
            continuation_prompt=_format_continuation_prompt(
                task=task,
                missing_evidence=missing_evidence,
                suggested_next_steps=suggested_next_steps,
                action_count=len(actions_taken),
            ),
        ),
        evidence_provenance=evidence_provenance,
        verification_observations=verification_observations,
    )


def assess_completion_follow_through(
    *,
    task: str,
    response: str,
    actions_taken: list[str],
    dod: DefinitionOfDone | None = None,
) -> TaskCompletionCheck:
    """Build the public completion-check contract for one candidate response."""

    return assess_completion_follow_through_with_provenance(
        task=task,
        response=response,
        actions_taken=actions_taken,
        dod=dod,
    ).check


def parse_completion_check(response: str, original_task: str) -> TaskCompletionCheck:
    """Parse an LLM completion-check response."""

    import json

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        return TaskCompletionCheck(original_task=original_task)

    try:
        data = json.loads(json_match.group())
        next_steps = data.get("next_steps", [])

        continuation = ""
        if not data.get("is_complete", True) and next_steps:
            steps = "\n".join(f"- {step}" for step in next_steps[:3])
            continuation = (
                f"Task not complete. Next steps:\n{steps}\n\n"
                "Continue executing these steps now."
            )

        return TaskCompletionCheck(
            original_task=original_task,
            is_complete=data.get("is_complete", False),
            accomplished=data.get("accomplished", []),
            required_evidence=data.get("required_evidence", []),
            missing_evidence=data.get("missing_evidence", data.get("remaining", [])),
            remaining=data.get("remaining", []),
            suggested_next_steps=next_steps,
            continuation_prompt=continuation,
        )
    except json.JSONDecodeError:
        return TaskCompletionCheck(original_task=original_task)


def _action_types(actions_taken: list[str]) -> set[str]:
    action_types: set[str] = set()
    for action in actions_taken:
        action_lower = action.lower()
        if "write" in action_lower:
            action_types.add("write")
        elif "edit" in action_lower or "patch" in action_lower:
            action_types.add("edit")
        elif "bash" in action_lower or "shell" in action_lower:
            action_types.add("bash")
        elif "read" in action_lower:
            action_types.add("read")
        elif "glob" in action_lower or "grep" in action_lower or "search" in action_lower:
            action_types.add("search")
        elif "todo" in action_lower:
            action_types.add("workflow")
    return action_types


def _is_informational_task(task_lower: str) -> bool:
    if task_lower.startswith(_INFORMATIONAL_PREFIXES):
        return True
    if task_lower.endswith("?") and task_lower.startswith(
        ("what ", "how ", "why ", "which ", "who ", "where ", "when ")
    ):
        return True
    return False


def _requires_action(task_lower: str) -> bool:
    return any(verb in task_lower for verb in _ACTION_VERBS) or any(
        indicator in task_lower for indicator in _SIMPLE_TASK_INDICATORS
    )


def _required_evidence(
    *,
    task_lower: str,
    informational: bool,
    complex_task: bool,
    requires_verification: bool,
    requires_install: bool,
) -> list[str]:
    if informational:
        return []

    required: list[str] = []
    if _requires_action(task_lower):
        required.append(_ACTION_EVIDENCE)
    if requires_install:
        required.append(_INSTALL_EVIDENCE)
    if requires_verification:
        required.append(_VERIFICATION_EVIDENCE)
    if complex_task:
        required.append(_COMPLEX_EVIDENCE)
    return required


def _has_install_evidence(
    task_lower: str,
    action_types: set[str],
    actions_taken: list[str],
) -> bool:
    del action_types
    action_text = " ".join(actions_taken).lower()
    if any(hint in task_lower for hint in _NODE_HINTS) and "npm" in action_text:
        return True
    if any(hint in task_lower for hint in _PYTHON_HINTS) and (
        "pip" in action_text or "uv" in action_text
    ):
        return True
    return "install" in action_text or "init" in action_text or "setup" in action_text


def _has_verification_evidence(
    action_types: set[str],
    actions_taken: list[str],
) -> bool:
    if "bash" in action_types:
        return True
    action_text = " ".join(actions_taken).lower()
    return any(
        token in action_text
        for token in ("test", "pytest", "jest", "verify", "run", "execute")
    )


def _install_follow_up(task_lower: str, verification_command: str | None) -> str:
    if any(hint in task_lower for hint in _NODE_HINTS):
        return "Run `npm install` to install dependencies"
    if any(hint in task_lower for hint in _PYTHON_HINTS):
        return "Install the Python dependencies"
    if verification_command:
        return f"Finish setup before rerunning `{verification_command}`"
    return "Install or initialize the required dependencies now"


def _append_follow_through_gap(
    missing_evidence: list[str],
    remaining: list[str],
    suggested_next_steps: list[str],
    *,
    evidence: str,
    remaining_item: str,
    next_step: str,
) -> None:
    if evidence not in missing_evidence:
        missing_evidence.append(evidence)
    if remaining_item not in remaining:
        remaining.append(remaining_item)
    if next_step not in suggested_next_steps:
        suggested_next_steps.append(next_step)


def _observed_completion_provenance(
    *,
    task_lower: str,
    response_lower: str,
    actions_taken: list[str],
    facts: _FollowThroughFacts,
    informational: bool,
    requires_install: bool,
    requires_verification: bool,
    dod: DefinitionOfDone | None,
) -> list[EvidenceProvenance]:
    entries: list[EvidenceProvenance] = []

    if informational and response_lower.strip():
        _append_unique_provenance(
            entries,
            EvidenceProvenance(
                category="response",
                source="assistant_response",
                summary="the assistant provided a direct informational response",
                status=EvidenceProvenanceStatus.SUPPORTS.value,
            ),
        )
        return entries

    has_concrete_task_work = bool(actions_taken) or bool(
        dod
        and (
            dod.touched_files
            or dod.mutating_actions
            or any(
                command and not _looks_like_verification_command(command)
                for command in dod.successful_commands
            )
            or any(
                item and item not in {_IMPLEMENTATION_ITEM, _VERIFY_ITEM}
                for item in dod.completed_items
            )
        )
    )
    if _requires_action(task_lower) and has_concrete_task_work:
        _append_unique_provenance(
            entries,
            EvidenceProvenance(
                category="action",
                source="dod" if dod is not None else "actions_taken",
                summary="runtime history showed concrete work for the requested task",
                status=EvidenceProvenanceStatus.SUPPORTS.value,
            ),
        )

    if requires_install and facts.has_install_evidence:
        _append_unique_provenance(
            entries,
            EvidenceProvenance(
                category="install",
                source="dod.successful_commands" if dod is not None else "actions_taken",
                summary="runtime history included dependency or setup work",
                status=EvidenceProvenanceStatus.SUPPORTS.value,
            ),
        )

    if requires_verification:
        status = (
            EvidenceProvenanceStatus.CONTRADICTS
            if facts.has_failed_verification
            else EvidenceProvenanceStatus.SUPPORTS
            if facts.has_verification_evidence
            else None
        )
        if status is not None:
            for entry in _verification_provenance(
                dod=dod,
                verification_command=facts.verification_command,
                status=status,
            ):
                _append_unique_provenance(entries, entry)

    return entries


def _format_continuation_prompt(
    *,
    task: str,
    missing_evidence: list[str],
    suggested_next_steps: list[str],
    action_count: int,
) -> str:
    if suggested_next_steps:
        evidence_lines = "\n".join(f"- {item}" for item in missing_evidence[:2])
        step_lines = "\n".join(f"- {step}" for step in suggested_next_steps[:3])
        return (
            f'The task was: "{task}"\n\n'
            "The response still needs concrete evidence for:\n"
            f"{evidence_lines}\n\n"
            "Continue with:\n"
            f"{step_lines}\n\n"
            "If the task is actually complete, confirm the missing evidence explicitly."
        )

    return (
        f'Task: "{task}"\n'
        f"You took {action_count} action(s). "
        "If there's more to do, continue. Otherwise, confirm completion."
    )


def _summarize_action(action: str) -> str:
    head, _, _ = action.partition(":")
    return head.strip() or action.strip()


def _build_follow_through_facts(
    *,
    task_lower: str,
    actions_taken: list[str],
    action_types: set[str],
    dod: DefinitionOfDone | None,
) -> _FollowThroughFacts:
    accomplished = [_summarize_action(action) for action in actions_taken]
    has_recorded_work = bool(actions_taken)
    has_install_evidence = _has_install_evidence(task_lower, action_types, actions_taken)
    has_verification_evidence = _has_verification_evidence(action_types, actions_taken)
    has_failed_verification = False
    has_pending_verification = False
    has_planned_verification = False
    has_stale_verification = False
    verification_command: str | None = None
    pending_items: list[str] = []

    if dod is None:
        return _FollowThroughFacts(
            has_recorded_work=has_recorded_work,
            has_install_evidence=has_install_evidence,
            has_verification_evidence=has_verification_evidence,
            has_failed_verification=has_failed_verification,
            has_pending_verification=has_pending_verification,
            has_planned_verification=has_planned_verification,
            has_stale_verification=has_stale_verification,
            verification_command=verification_command,
            pending_items=pending_items,
            accomplished=accomplished,
        )

    pending_items = [
        item.strip()
        for item in dod.pending_items
        if item.strip() and item not in {_IMPLEMENTATION_ITEM, _VERIFY_ITEM}
    ]
    verification_command = _first_verification_command(dod)
    has_install_evidence = has_install_evidence or any(
        token in command.lower()
        for command in dod.successful_commands
        for token in ("install", "init", "setup")
    )
    has_verification_evidence = has_verification_evidence or any(
        evidence.passed for evidence in dod.evidence
    )
    has_failed_verification = (
        dod.last_verification_result == "failed"
        or any(not evidence.passed for evidence in dod.evidence)
    )
    has_pending_verification = (
        dod.last_verification_result == VerificationObservationStatus.PENDING.value
    )
    has_planned_verification = (
        dod.last_verification_result == VerificationObservationStatus.PLANNED.value
    )
    has_stale_verification = dod.last_verification_result == "stale"
    has_recorded_work = has_recorded_work or bool(
        dod.touched_files
        or dod.successful_commands
        or dod.mutating_actions
        or dod.completed_items
        or has_verification_evidence
        or has_failed_verification
        or has_pending_verification
        or has_planned_verification
        or has_stale_verification
    )
    for evidence in dod.evidence:
        if not evidence.passed:
            continue
        if evidence.command:
            _append_unique(accomplished, f"verified: {evidence.command}")
        elif evidence.output:
            _append_unique(accomplished, "verified the runtime result")
    for command in dod.successful_commands:
        if _looks_like_verification_command(command):
            _append_unique(accomplished, f"ran: {command}")
    for item in dod.completed_items:
        if item and item not in {_IMPLEMENTATION_ITEM, _VERIFY_ITEM}:
            _append_unique(accomplished, f"completed: {item}")

    return _FollowThroughFacts(
        has_recorded_work=has_recorded_work,
        has_install_evidence=has_install_evidence,
        has_verification_evidence=has_verification_evidence,
        has_failed_verification=has_failed_verification,
        has_pending_verification=has_pending_verification,
        has_planned_verification=has_planned_verification,
        has_stale_verification=has_stale_verification,
        verification_command=verification_command,
        pending_items=pending_items,
        accomplished=accomplished,
    )


def _first_verification_command(dod: DefinitionOfDone) -> str | None:
    for evidence in dod.evidence:
        if evidence.command:
            return evidence.command
    for command in dod.verification_commands:
        if command:
            return command
    for command in dod.successful_commands:
        if _looks_like_verification_command(command):
            return command
    return None


def _looks_like_verification_command(command: str) -> bool:
    command_lower = command.lower()
    return any(
        token in command_lower
        for token in ("test", "pytest", "jest", "verify", "run", "execute", "check")
    )


def _missing_verification_evidence(verification_command: str | None) -> str:
    if verification_command:
        return f"a passing verification result from `{verification_command}`"
    return _VERIFICATION_EVIDENCE


def _failed_verification_evidence(verification_command: str | None) -> str:
    if verification_command:
        return (
            f"a passing verification result from `{verification_command}` "
            "(current verification is still failing)"
        )
    return "a passing verification result (current verification is still failing)"


def _pending_verification_evidence(verification_command: str | None) -> str:
    if verification_command:
        return (
            f"a completed passing verification result from `{verification_command}` "
            "(verification is still pending)"
        )
    return "a completed passing verification result (verification is still pending)"


def _planned_verification_evidence(verification_command: str | None) -> str:
    if verification_command:
        return (
            f"a passing verification result from `{verification_command}` "
            "(verification is planned but has not run yet)"
        )
    return "a passing verification result (verification is planned but has not run yet)"


def _stale_verification_evidence(verification_command: str | None) -> str:
    if verification_command:
        return (
            f"a fresh passing verification result from `{verification_command}` "
            "(previous verification became stale after new mutating work)"
        )
    return "a fresh passing verification result after new mutating work"


def _verification_follow_up(
    *,
    task_lower: str,
    verification_command: str | None,
) -> str:
    if verification_command:
        return f"Run `{verification_command}` and capture the concrete result"
    return "Execute what you created or run the relevant tests now"


def _verification_retry_step(verification_command: str | None) -> str:
    if verification_command:
        return f"Fix the failing `{verification_command}` result and rerun it"
    return "Fix the failing verification result and rerun it"


def _pending_verification_follow_up(verification_command: str | None) -> str:
    if verification_command:
        return f"Finish running `{verification_command}` and capture the result"
    return "Finish the active verification run and capture the result"


def _planned_verification_follow_up(verification_command: str | None) -> str:
    if verification_command:
        return f"Run the planned verification `{verification_command}` now"
    return "Run the planned verification now"


def _stale_verification_follow_up(verification_command: str | None) -> str:
    if verification_command:
        return f"Rerun `{verification_command}` now that the implementation changed again"
    return "Rerun the relevant verification now that the implementation changed again"


def _verification_provenance(
    *,
    dod: DefinitionOfDone | None,
    verification_command: str | None,
    status: EvidenceProvenanceStatus,
) -> list[EvidenceProvenance]:
    entries: list[EvidenceProvenance] = []
    if dod is not None:
        for evidence in dod.evidence:
            if status is EvidenceProvenanceStatus.SUPPORTS and not evidence.passed:
                continue
            if status is EvidenceProvenanceStatus.CONTRADICTS and evidence.passed:
                continue
            command = evidence.command or verification_command
            summary = (
                f"verification passed for `{command}`"
                if status is EvidenceProvenanceStatus.SUPPORTS and command
                else "verification passed"
                if status is EvidenceProvenanceStatus.SUPPORTS
                else f"verification failed for `{command}`"
                if command
                else "verification was still failing"
            )
            _append_unique_provenance(
                entries,
                EvidenceProvenance(
                    category="verification",
                    source="dod.evidence",
                    summary=summary,
                    status=status.value,
                    subject=command,
                    detail=_verification_detail(evidence),
                ),
            )
    if entries or verification_command is None:
        return entries
    return [
        EvidenceProvenance(
            category="verification",
            source="dod.verification_commands",
            summary=(
                f"verification passed for `{verification_command}`"
                if status is EvidenceProvenanceStatus.SUPPORTS
                else f"verification failed for `{verification_command}`"
            ),
            status=status.value,
            subject=verification_command,
        )
    ]


def _observed_completion_verification(
    *,
    dod: DefinitionOfDone | None,
    verification_command: str | None,
    requires_verification: bool,
) -> list[VerificationObservation]:
    if dod is None or not requires_verification:
        return []

    observations: list[VerificationObservation] = []
    observed_commands: set[str] = set()
    for evidence in dod.evidence:
        command = evidence.command or verification_command
        if command:
            observed_commands.add(command)
        observations.append(
            VerificationObservation(
                status=(
                    VerificationObservationStatus.PASSED.value
                    if evidence.passed
                    else VerificationObservationStatus.FAILED.value
                ),
                summary=(
                    f"verification passed for `{command}`"
                    if evidence.passed and command
                    else "verification passed"
                    if evidence.passed
                    else f"verification failed for `{command}`"
                    if command
                    else "verification was still failing"
                ),
                command=command,
                kind=evidence.kind,
                exit_code=evidence.exit_code,
                detail=_verification_detail(evidence),
            )
        )

    if observations:
        for command in dod.verification_commands:
            if not command or command in observed_commands:
                continue
            observations.append(
                VerificationObservation(
                    status=VerificationObservationStatus.MISSING.value,
                    summary=f"verification did not produce an observed result for `{command}`",
                    command=command,
                )
            )
        return observations

    if dod.last_verification_result == VerificationObservationStatus.PENDING.value:
        if verification_command:
            return [
                VerificationObservation(
                    status=VerificationObservationStatus.PENDING.value,
                    summary=f"verification pending for `{verification_command}`",
                    command=verification_command,
                )
            ]
        return [
            VerificationObservation(
                status=VerificationObservationStatus.PENDING.value,
                summary="verification is pending for the active command set",
            )
        ]

    if dod.last_verification_result == VerificationObservationStatus.PLANNED.value:
        if verification_command:
            return [
                VerificationObservation(
                    status=VerificationObservationStatus.PLANNED.value,
                    summary=f"verification planned for `{verification_command}`",
                    command=verification_command,
                )
            ]
        return [
            VerificationObservation(
                status=VerificationObservationStatus.PLANNED.value,
                summary="verification is planned but has not run yet",
            )
        ]

    if dod.last_verification_result == VerificationObservationStatus.STALE.value:
        if verification_command:
            return [
                VerificationObservation(
                    status=VerificationObservationStatus.STALE.value,
                    summary=(
                        "verification became stale for "
                        f"`{verification_command}` after new mutating work"
                    ),
                    command=verification_command,
                )
            ]
        return [
            VerificationObservation(
                status=VerificationObservationStatus.STALE.value,
                summary="previous verification became stale after new mutating work",
            )
        ]

    if verification_command:
        return [
            VerificationObservation(
                status=VerificationObservationStatus.MISSING.value,
                summary=(
                    "verification did not produce an observed result for "
                    f"`{verification_command}`"
                ),
                command=verification_command,
            )
        ]
    return []


def _verification_detail(evidence) -> str | None:
    for candidate in (evidence.stdout, evidence.stderr, evidence.output):
        text = str(candidate).strip()
        if text:
            return text.splitlines()[0]
    return None


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _append_unique_provenance(
    items: list[EvidenceProvenance],
    item: EvidenceProvenance,
) -> None:
    for existing in items:
        if (
            existing.category == item.category
            and existing.source == item.source
            and existing.summary == item.summary
            and existing.status == item.status
            and existing.subject == item.subject
        ):
            return
    items.append(item)
