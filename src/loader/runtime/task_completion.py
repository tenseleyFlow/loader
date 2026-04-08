"""Runtime-owned completion heuristics and continuation prompts."""

from __future__ import annotations

import re

from .reasoning_types import TaskCompletionCheck

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

    task_lower = task.lower()
    response_lower = response.lower()

    if not actions_taken:
        explicit_completion = response_lower.strip()
        if explicit_completion in {
            "done",
            "done.",
            "completed",
            "completed.",
            "all set",
            "all set.",
        }:
            return False
        action_verbs = ["create", "write", "make", "edit", "fix", "add", "delete", "run"]
        if any(verb in task_lower for verb in action_verbs):
            return True
        return False

    success_indicators = [
        "successfully",
        "created",
        "written",
        "done",
        "completed",
        "file now contains",
        "has been updated",
        "installed",
    ]
    if any(indicator in response_lower for indicator in success_indicators):
        return False

    complex_indicators = [
        "set up a project",
        "create a project",
        "build a complete",
        "scaffold",
        "initialize a new",
        "create a full",
        "implement a full",
        "develop a complete",
    ]
    is_complex = any(indicator in task_lower for indicator in complex_indicators)

    simple_creation = [
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
    ]
    is_simple = any(indicator in task_lower for indicator in simple_creation)

    if "write" in str(actions_taken).lower() and len(actions_taken) >= 1:
        return False
    if is_simple and len(actions_taken) >= 1:
        return False

    explicit_verification = ["and test", "and run", "and verify", "make sure it works"]
    needs_verification = any(indicator in task_lower for indicator in explicit_verification)

    action_types = set()
    for action in actions_taken:
        action_lower = action.lower()
        if "write" in action_lower:
            action_types.add("write")
        elif "edit" in action_lower:
            action_types.add("edit")
        elif "bash" in action_lower:
            action_types.add("bash")
        elif "read" in action_lower:
            action_types.add("read")
        elif "glob" in action_lower or "grep" in action_lower:
            action_types.add("search")

    if is_complex and len(actions_taken) < 3:
        return True
    if needs_verification and "bash" not in action_types:
        return True

    deflection_phrases = ["you can now", "you should", "you can run", "you can use"]
    if any(phrase in response_lower for phrase in deflection_phrases) and len(actions_taken) < 2:
        return True

    return False


def get_continuation_prompt(task: str, actions_taken: list[str], response: str) -> str:
    """Generate a helpful follow-through prompt for incomplete tasks."""

    del response

    task_lower = task.lower()
    follow_ups: list[str] = []

    if any(keyword in task_lower for keyword in ["install", "dependencies", "set up project"]):
        if "node" in task_lower or "npm" in task_lower:
            if not any("npm" in action for action in actions_taken):
                follow_ups.append("Run `npm install` to install dependencies")
        if "python" in task_lower or "pip" in task_lower:
            if not any("pip" in action or "uv" in action for action in actions_taken):
                follow_ups.append("Install dependencies")

    if "test" in task_lower and "run" in task_lower:
        if not any("test" in action or "pytest" in action or "jest" in action for action in actions_taken):
            follow_ups.append("Run the tests")

    if any(keyword in task_lower for keyword in ["and run", "and test", "and verify", "make sure it works"]):
        follow_ups.append("Execute what was created to verify it works")

    if follow_ups:
        steps = "\n".join(f"- {step}" for step in follow_ups[:2])
        return (
            f'The task was: "{task}"\n\n'
            f"You may need to also:\n{steps}\n\n"
            "If the task is actually complete, just confirm what was done."
        )

    return (
        f'Task: "{task}"\n'
        f"You took {len(actions_taken)} action(s). "
        "If there's more to do, continue. Otherwise, confirm completion."
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
            remaining=data.get("remaining", []),
            suggested_next_steps=next_steps,
            continuation_prompt=continuation,
        )
    except json.JSONDecodeError:
        return TaskCompletionCheck(original_task=original_task)
