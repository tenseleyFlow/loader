"""Runtime-owned completion heuristics and continuation prompts."""

from __future__ import annotations

import re

from .reasoning_types import TaskCompletionCheck

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


def detect_premature_completion(
    task: str,
    response: str,
    actions_taken: list[str],
) -> bool:
    """Heuristically detect when the assistant is stopping too early."""
    if not actions_taken and response.lower().strip() in _EXPLICIT_COMPLETIONS:
        return False
    return not assess_completion_follow_through(
        task=task,
        response=response,
        actions_taken=actions_taken,
    ).is_complete


def get_continuation_prompt(task: str, actions_taken: list[str], response: str) -> str:
    """Generate a helpful follow-through prompt for incomplete tasks."""
    return assess_completion_follow_through(
        task=task,
        response=response,
        actions_taken=actions_taken,
    ).continuation_prompt


def assess_completion_follow_through(
    *,
    task: str,
    response: str,
    actions_taken: list[str],
) -> TaskCompletionCheck:
    """Build a typed follow-through assessment for one candidate response."""

    task_lower = task.lower().strip()
    response_lower = response.lower().strip()
    action_types = _action_types(actions_taken)
    informational = _is_informational_task(task_lower)
    complex_task = any(indicator in task_lower for indicator in _COMPLEX_INDICATORS)
    simple_task = any(indicator in task_lower for indicator in _SIMPLE_TASK_INDICATORS)
    requires_verification = any(
        indicator in task_lower for indicator in _VERIFICATION_INDICATORS
    )
    requires_install = any(indicator in task_lower for indicator in _INSTALL_HINTS)

    accomplished = [_summarize_action(action) for action in actions_taken]
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
        return TaskCompletionCheck(
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
        )

    if not actions_taken and _requires_action(task_lower):
        _append_follow_through_gap(
            missing_evidence,
            remaining,
            suggested_next_steps,
            evidence="showing the requested work was actually carried out",
            remaining_item="Perform the requested work instead of stopping at intent or narration",
            next_step="Carry out the requested change or command now",
        )

    if requires_install and not _has_install_evidence(task_lower, action_types, actions_taken):
        _append_follow_through_gap(
            missing_evidence,
            remaining,
            suggested_next_steps,
            evidence="showing dependencies or setup steps were completed",
            remaining_item="Install or initialize the required dependencies",
            next_step=_install_follow_up(task_lower),
        )

    if requires_verification and not _has_verification_evidence(action_types, actions_taken):
        _append_follow_through_gap(
            missing_evidence,
            remaining,
            suggested_next_steps,
            evidence="showing the result was run or verified",
            remaining_item="Run the result and capture a concrete verification outcome",
            next_step="Execute what you created or run the relevant tests now",
        )

    if complex_task and len(actions_taken) < 3:
        _append_follow_through_gap(
            missing_evidence,
            remaining,
            suggested_next_steps,
            evidence="showing the broader end-to-end implementation or setup was completed",
            remaining_item="Finish the larger end-to-end task instead of stopping after a partial step",
            next_step="Continue through the remaining setup or implementation steps",
        )

    if (
        any(phrase in response_lower for phrase in _DEFLECTION_PHRASES)
        and len(actions_taken) < 2
    ):
        _append_follow_through_gap(
            missing_evidence,
            remaining,
            suggested_next_steps,
            evidence="showing execution evidence rather than instructions handed back to the user",
            remaining_item="Perform the work yourself or state concretely what you already verified",
            next_step="Continue the task instead of handing the next step to the user",
        )

    if "write" in action_types and actions_taken and simple_task:
        missing_evidence = [
            item
            for item in missing_evidence
            if item != "showing the requested work was actually carried out"
        ]
        remaining = [
            item
            for item in remaining
            if item != "Perform the requested work instead of stopping at intent or narration"
        ]

    is_complete = not missing_evidence
    return TaskCompletionCheck(
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
    )


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
        required.append("showing the requested work was actually carried out")
    if requires_install:
        required.append("showing dependencies or setup steps were completed")
    if requires_verification:
        required.append("showing the result was run or verified")
    if complex_task:
        required.append("showing the broader end-to-end implementation or setup was completed")
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


def _install_follow_up(task_lower: str) -> str:
    if any(hint in task_lower for hint in _NODE_HINTS):
        return "Run `npm install` to install dependencies"
    if any(hint in task_lower for hint in _PYTHON_HINTS):
        return "Install the Python dependencies"
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
