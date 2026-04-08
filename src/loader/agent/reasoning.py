"""Reasoning stages for enhanced agent cognition.

This module implements multiple reasoning stages that can be optionally
enabled to improve the agent's decision-making:

1. Decomposition (High complexity) - Break complex tasks into atomic subtasks
2. Self-Critique Loop (Medium) - Review and improve output before finalizing
3. Confidence Scoring (Medium) - Rate certainty of actions
4. Post-Action Verification (Low) - Check if actions produced expected results
"""

import re
from typing import Any

from ..runtime.rollback import (
    RollbackAction,
    RollbackPlan,
    RollbackType,
    create_rollback_plan_for_action,
    execute_rollback,
    get_undo_command,
    is_destructive_tool,
)
from ..runtime.reasoning_types import (
    ActionVerification,
    ConfidenceAssessment,
    ConfidenceLevel,
    SelfCritique,
    Subtask,
    TaskCompletionCheck,
    TaskDecomposition,
)
from ..runtime.task_classification import (
    estimate_complexity,
    get_token_budget,
    is_conversational,
)


# Prompts for reasoning stages

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
      "dependencies": [],  // IDs of subtasks this depends on
      "verification": "How to verify this succeeded"
    }},
    {{
      "id": "2",
      "description": "Second subtask...",
      "dependencies": ["1"],
      "verification": "..."
    }}
  ],
  "rollback_points": [0, 2],  // Indices where we can safely rollback
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


CONFIDENCE_PROMPT = """Rate your confidence in this action before executing.

Action: {action}
Tool: {tool_name}
Arguments: {tool_args}

Previous context:
{context}

Consider:
1. Do you have enough information to proceed?
2. What could go wrong?
3. Is this the right approach?
4. Are there better alternatives?

Respond in this exact JSON format:
{{
  "confidence": 1-5,  // 1=very low, 2=low, 3=medium, 4=high, 5=very high
  "reasoning": "Why this confidence level",
  "risks": ["Risk 1", "Risk 2"],
  "mitigations": ["How to mitigate risk 1"],
  "requires_verification": true/false,
  "alternative_approaches": ["Alternative 1 if confidence is low"]
}}

Only output the JSON, no other text."""


VERIFICATION_PROMPT = """Verify that the action produced the expected result.

Action taken:
- Tool: {tool_name}
- Arguments: {tool_args}

Expected outcome: {expected}

Actual result:
{result}

Analyze:
1. Did the action succeed?
2. Does the result match expectations?
3. Are there any unexpected side effects?
4. Is any correction needed?

Respond in this exact JSON format:
{{
  "verified": true/false,
  "verification_method": "How you verified (e.g., output_contains, no_error, file_created)",
  "discrepancies": ["Discrepancy 1 if any"],
  "needs_correction": true/false,
  "correction_suggestion": "What to do if correction is needed"
}}

Only output the JSON, no other text."""


def parse_decomposition(response: str, original_task: str) -> TaskDecomposition:
    """Parse LLM response into TaskDecomposition."""
    import json
    import re

    # Extract JSON from response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        # Fallback: treat the whole task as a single subtask
        return TaskDecomposition(
            original_task=original_task,
            subtasks=[Subtask(
                id="1",
                description=original_task,
                verification="Check completion",
            )],
        )

    try:
        data = json.loads(json_match.group())
        subtasks = []
        for st_data in data.get("subtasks", []):
            subtasks.append(Subtask(
                id=str(st_data.get("id", len(subtasks) + 1)),
                description=st_data.get("description", ""),
                dependencies=st_data.get("dependencies", []),
                verification=st_data.get("verification", ""),
            ))

        return TaskDecomposition(
            original_task=original_task,
            subtasks=subtasks,
            rollback_points=data.get("rollback_points", []),
        )
    except json.JSONDecodeError:
        return TaskDecomposition(
            original_task=original_task,
            subtasks=[Subtask(
                id="1",
                description=original_task,
                verification="Check completion",
            )],
        )


def parse_self_critique(response: str, original_response: str) -> SelfCritique:
    """Parse LLM response into SelfCritique."""
    import json
    import re

    json_match = re.search(r'\{.*\}', response, re.DOTALL)
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


def parse_confidence(response: str, tool_name: str, tool_args: dict) -> ConfidenceAssessment:
    """Parse LLM response into ConfidenceAssessment."""
    import json
    import re

    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        return ConfidenceAssessment(
            action="",
            tool_name=tool_name,
            tool_args=tool_args,
        )

    try:
        data = json.loads(json_match.group())
        confidence_val = data.get("confidence", 3)
        level = ConfidenceLevel(max(1, min(5, confidence_val)))

        return ConfidenceAssessment(
            action=data.get("action", ""),
            tool_name=tool_name,
            tool_args=tool_args,
            level=level,
            reasoning=data.get("reasoning", ""),
            risks=data.get("risks", []),
            mitigations=data.get("mitigations", []),
            requires_verification=data.get("requires_verification", level.value <= 3),
        )
    except (json.JSONDecodeError, ValueError):
        return ConfidenceAssessment(
            action="",
            tool_name=tool_name,
            tool_args=tool_args,
        )


def parse_verification(
    response: str, tool_name: str, tool_args: dict, expected: str, result: str
) -> ActionVerification:
    """Parse LLM response into ActionVerification."""
    import json
    import re

    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        # Default: assume verified if no error in result
        return ActionVerification(
            tool_name=tool_name,
            tool_args=tool_args,
            expected_outcome=expected,
            actual_result=result,
            verified="error" not in result.lower(),
        )

    try:
        data = json.loads(json_match.group())
        return ActionVerification(
            tool_name=tool_name,
            tool_args=tool_args,
            expected_outcome=expected,
            actual_result=result,
            verified=data.get("verified", False),
            verification_method=data.get("verification_method", ""),
            discrepancies=data.get("discrepancies", []),
            needs_correction=data.get("needs_correction", False),
            correction_suggestion=data.get("correction_suggestion", ""),
        )
    except json.JSONDecodeError:
        return ActionVerification(
            tool_name=tool_name,
            tool_args=tool_args,
            expected_outcome=expected,
            actual_result=result,
            verified="error" not in result.lower(),
        )


# Quick heuristics to avoid calling LLM for every decision

def should_decompose(task: str) -> bool:
    """Heuristic check if task needs decomposition.

    Returns True for complex tasks that benefit from breaking down.
    """
    # Indicators of complexity
    complexity_markers = [
        " and ", " then ", " after ", " before ",  # Sequential actions
        "multiple", "several", "all the", "each",  # Multiple items
        "refactor", "migrate", "upgrade", "convert",  # Complex operations
        "create a", "build a", "implement",  # Building something
        "set up", "configure", "install",  # Setup operations
    ]

    task_lower = task.lower()

    # Count complexity markers
    marker_count = sum(1 for marker in complexity_markers if marker in task_lower)

    # Length-based heuristic
    word_count = len(task.split())

    # Decompose if: multiple markers, or long task, or explicit multi-step
    return marker_count >= 2 or word_count > 30 or "step" in task_lower


def should_self_critique(response: str, is_code: bool = False) -> bool:
    """Heuristic check if response needs self-critique.

    Returns True for responses that might benefit from review.
    """
    # Critique code changes more often
    if is_code:
        # Always critique code changes above a certain size
        return len(response) > 500

    # Critique long responses
    if len(response) > 1500:
        return True

    # Critique responses with uncertainty markers
    uncertainty = ["might", "could", "perhaps", "maybe", "i think", "not sure"]
    if any(marker in response.lower() for marker in uncertainty):
        return True

    return False


def estimate_confidence_quick(tool_name: str, tool_args: dict, context: str = "") -> ConfidenceLevel:
    """Quick confidence estimation without LLM call.

    For simple operations, we can estimate confidence based on patterns.
    """
    # High confidence operations
    if tool_name == "read":
        return ConfidenceLevel.HIGH  # Reading is safe
    if tool_name == "glob" or tool_name == "grep":
        return ConfidenceLevel.HIGH  # Searching is safe

    # Medium confidence by default
    if tool_name == "write":
        # New file creation is generally safe
        file_path = tool_args.get("file_path", "")
        if not file_path:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.MEDIUM

    if tool_name == "edit":
        # Editing requires the file to exist and strings to match
        old_string = tool_args.get("old_string", "")
        if not old_string:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.MEDIUM

    if tool_name == "patch":
        hunks = tool_args.get("hunks", [])
        if not hunks:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.MEDIUM

    if tool_name == "git":
        return ConfidenceLevel.HIGH

    if tool_name == "bash":
        command = tool_args.get("command", "")
        # Dangerous commands get low confidence
        dangerous = ["rm -rf", "sudo", "chmod 777", "dd if=", "> /dev/"]
        if any(d in command for d in dangerous):
            return ConfidenceLevel.VERY_LOW
        # Safe read-only commands get high confidence
        safe_read = ["ls", "cat", "grep", "find", "pwd", "echo", "head", "tail"]
        if any(command.strip().startswith(s) for s in safe_read):
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.MEDIUM


def quick_verify(tool_name: str, tool_args: dict, result: str) -> bool:
    """Quick verification without LLM call.

    For simple operations, verify based on result patterns.
    """
    result_lower = result.lower()

    # Check for obvious failures
    error_indicators = [
        "error:", "failed", "not found", "permission denied",
        "no such file", "command not found", "exception",
        "traceback", "fatal:", "cannot",
    ]
    if any(indicator in result_lower for indicator in error_indicators):
        return False

    # Tool-specific checks
    if tool_name == "write":
        # Write should return a success message
        return "created" in result_lower or "wrote" in result_lower or len(result) < 200

    if tool_name == "edit":
        # Edit should return success or diff
        return "edited" in result_lower or "+" in result or "-" in result

    if tool_name == "patch":
        return "patched" in result_lower or "+" in result or "-" in result

    if tool_name == "read":
        # Read should return file contents (non-empty)
        return len(result.strip()) > 0

    if tool_name == "git":
        return len(result.strip()) > 0

    if tool_name == "bash":
        # Bash success if no error output
        return True  # Already checked error indicators above

    return True


# === Task Completion Detection ===

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


def detect_premature_completion(task: str, response: str, actions_taken: list[str]) -> bool:
    """Quick heuristic to detect if agent is stopping too early.

    Returns True if the agent might be giving up prematurely.
    This should be CONSERVATIVE - only trigger when really needed,
    not for simple tasks that are genuinely complete.
    """
    task_lower = task.lower()
    response_lower = response.lower()

    # If no actions taken at all and task requires action, that's premature
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
        # But only if this looks like an actionable task
        action_verbs = ["create", "write", "make", "edit", "fix", "add", "delete", "run"]
        if any(verb in task_lower for verb in action_verbs):
            return True
        return False  # Informational/conversational tasks don't need actions

    # If we took actions and got successful results, trust that we're done
    # Check for success indicators in response
    success_indicators = [
        "successfully", "created", "written", "done", "completed",
        "file now contains", "has been updated", "installed",
    ]
    if any(ind in response_lower for ind in success_indicators) and len(actions_taken) >= 1:
        return False  # Likely actually done

    # Keywords that suggest COMPLEX multi-step tasks (not simple ones)
    complex_indicators = [
        "set up a project", "create a project", "build a complete",
        "scaffold", "initialize a new", "create a full",
        "implement a full", "develop a complete",
    ]
    is_complex = any(ind in task_lower for ind in complex_indicators)

    # Simple creation tasks don't need follow-up
    simple_creation = [
        "create a file", "write a file", "make a file",
        "add a function", "edit the", "fix the", "update the",
        "read the", "show me", "list",
        # Web page / design tasks are also typically simple
        "design a webpage", "create a webpage", "make a webpage",
        "create a page", "design a page", "create an html",
        "make an html", "write an html", "help me design",
        "create a simple", "make a simple", "write a simple",
    ]
    is_simple = any(ind in task_lower for ind in simple_creation)

    # If we already created/wrote files, the task is probably done
    if "write" in str(actions_taken).lower() and len(actions_taken) >= 1:
        return False  # File was written, trust it's done

    # If it's a simple task with at least one action, it's probably done
    if is_simple and len(actions_taken) >= 1:
        return False

    # Explicit verification requests need bash
    explicit_verification = ["and test", "and run", "and verify", "make sure it works"]
    needs_verification = any(ind in task_lower for ind in explicit_verification)

    # Categorize what actions were taken
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

    # Detection rules (more conservative):

    # 1. Complex project tasks with very few actions
    if is_complex and len(actions_taken) < 3:
        return True

    # 2. Explicitly requested verification but no bash run
    if needs_verification and "bash" not in action_types:
        return True

    # 3. Chatbot-style deflection with no real work done
    deflection_phrases = ["you can now", "you should", "you can run", "you can use"]
    if any(phrase in response_lower for phrase in deflection_phrases) and len(actions_taken) < 2:
        return True

    return False


def get_continuation_prompt(task: str, actions_taken: list[str], response: str) -> str:
    """Generate a prompt to encourage the agent to continue.

    Returns a prompt that nudges the agent to follow through.
    Should be helpful, not aggressive.
    """
    task_lower = task.lower()
    actions_str = ", ".join(a.split(":")[0] for a in actions_taken[-5:]) if actions_taken else "none"

    # Determine what type of follow-up is needed
    follow_ups = []

    # Only suggest package install if explicitly mentioned in task
    if any(kw in task_lower for kw in ["install", "dependencies", "set up project"]):
        if "node" in task_lower or "npm" in task_lower:
            if not any("npm" in a for a in actions_taken):
                follow_ups.append("Run `npm install` to install dependencies")
        if "python" in task_lower or "pip" in task_lower:
            if not any("pip" in a or "uv" in a for a in actions_taken):
                follow_ups.append("Install dependencies")

    # Only suggest running tests if "test" is explicitly in task
    if "test" in task_lower and "run" in task_lower:
        if not any("test" in a or "pytest" in a or "jest" in a for a in actions_taken):
            follow_ups.append("Run the tests")

    # If task explicitly asks to run/verify, remind to do so
    if any(kw in task_lower for kw in ["and run", "and test", "and verify", "make sure it works"]):
        follow_ups.append("Execute what was created to verify it works")

    if follow_ups:
        steps = "\n".join(f"- {step}" for step in follow_ups[:2])
        return (
            f"The task was: \"{task}\"\n\n"
            f"You may need to also:\n{steps}\n\n"
            f"If the task is actually complete, just confirm what was done."
        )

    # Generic - be gentle
    return (
        f"Task: \"{task}\"\n"
        f"You took {len(actions_taken)} action(s). "
        f"If there's more to do, continue. Otherwise, confirm completion."
    )


def parse_completion_check(response: str, original_task: str) -> TaskCompletionCheck:
    """Parse LLM completion check response."""
    import json
    import re

    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        return TaskCompletionCheck(original_task=original_task)

    try:
        data = json.loads(json_match.group())
        remaining = data.get("remaining", [])
        next_steps = data.get("next_steps", [])

        continuation = ""
        if not data.get("is_complete", True) and next_steps:
            steps = "\n".join(f"- {step}" for step in next_steps[:3])
            continuation = (
                f"Task not complete. Next steps:\n{steps}\n\n"
                f"Continue executing these steps now."
            )

        return TaskCompletionCheck(
            original_task=original_task,
            is_complete=data.get("is_complete", False),
            accomplished=data.get("accomplished", []),
            remaining=remaining,
            suggested_next_steps=next_steps,
            continuation_prompt=continuation,
        )
    except json.JSONDecodeError:
        return TaskCompletionCheck(original_task=original_task)
