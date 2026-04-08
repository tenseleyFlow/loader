"""Runtime-owned prompts and helpers for task decomposition and critique."""

from __future__ import annotations

import re

from .reasoning_types import SelfCritique, Subtask, TaskDecomposition

DECOMPOSITION_PROMPT = """Analyze this task and break it down into atomic subtasks.

Task: {task}

Rules:
1. Each subtask should be a single, atomic action (one tool call)
2. Identify dependencies between subtasks (what must complete before what)
3. For each subtask, describe how to verify it succeeded
4. Order subtasks so dependencies are respected
5. Identify safe rollback points (where we can undo changes if needed)

Respond in this exact JSON format:
{{
  "subtasks": [
    {{
      "id": "1",
      "description": "Short description of the subtask",
      "dependencies": [],
      "verification": "How to verify this succeeded"
    }},
    {{
      "id": "2",
      "description": "Second subtask...",
      "dependencies": ["1"],
      "verification": "..."
    }}
  ],
  "rollback_points": [0, 2],
  "complexity": "low|medium|high"
}}

Important: Only output the JSON, no other text."""


SELF_CRITIQUE_PROMPT = """Review your response and identify potential issues.

Your response:
{response}

Task context:
{context}

Analyze for:
1. Correctness - Is the logic/code correct?
2. Completeness - Did you address all parts of the request?
3. Safety - Any potential security or stability issues?
4. Best practices - Following conventions and patterns?
5. Edge cases - Are edge cases handled?

Respond in this exact JSON format:
{{
  "issues": [
    "Issue description 1",
    "Issue description 2"
  ],
  "suggestions": [
    "Suggestion for improvement 1",
    "Suggestion for improvement 2"
  ],
  "should_revise": true/false,
  "severity": "none|minor|moderate|major"
}}

Only output the JSON, no other text."""


def parse_decomposition(response: str, original_task: str) -> TaskDecomposition:
    """Parse an LLM decomposition response into structured subtasks."""

    import json

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        return TaskDecomposition(
            original_task=original_task,
            subtasks=[
                Subtask(
                    id="1",
                    description=original_task,
                    verification="Check completion",
                )
            ],
        )

    try:
        data = json.loads(json_match.group())
        subtasks: list[Subtask] = []
        for subtask_data in data.get("subtasks", []):
            subtasks.append(
                Subtask(
                    id=str(subtask_data.get("id", len(subtasks) + 1)),
                    description=subtask_data.get("description", ""),
                    dependencies=subtask_data.get("dependencies", []),
                    verification=subtask_data.get("verification", ""),
                )
            )

        return TaskDecomposition(
            original_task=original_task,
            subtasks=subtasks,
            rollback_points=data.get("rollback_points", []),
        )
    except json.JSONDecodeError:
        return TaskDecomposition(
            original_task=original_task,
            subtasks=[
                Subtask(
                    id="1",
                    description=original_task,
                    verification="Check completion",
                )
            ],
        )


def parse_self_critique(response: str, original_response: str) -> SelfCritique:
    """Parse an LLM self-critique response."""

    import json

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        return SelfCritique(original_response=original_response)

    try:
        data = json.loads(json_match.group())
        return SelfCritique(
            original_response=original_response,
            issues_found=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            should_revise=data.get("should_revise", False),
        )
    except json.JSONDecodeError:
        return SelfCritique(original_response=original_response)


def should_decompose(task: str) -> bool:
    """Heuristically detect tasks that benefit from decomposition."""

    complexity_markers = [
        " and ",
        " then ",
        " after ",
        " before ",
        "multiple",
        "several",
        "all the",
        "each",
        "refactor",
        "migrate",
        "upgrade",
        "convert",
        "create a",
        "build a",
        "implement",
        "set up",
        "configure",
        "install",
    ]

    task_lower = task.lower()
    marker_count = sum(1 for marker in complexity_markers if marker in task_lower)
    word_count = len(task.split())
    return marker_count >= 2 or word_count > 30 or "step" in task_lower


def should_self_critique(response: str, is_code: bool = False) -> bool:
    """Heuristically detect responses worth reviewing before finalizing."""

    if is_code:
        return len(response) > 500

    if len(response) > 1500:
        return True

    uncertainty_markers = ["might", "could", "perhaps", "maybe", "i think", "not sure"]
    return any(marker in response.lower() for marker in uncertainty_markers)
