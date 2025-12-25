"""Reasoning stages for enhanced agent cognition.

This module implements multiple reasoning stages that can be optionally
enabled to improve the agent's decision-making:

1. Decomposition (High complexity) - Break complex tasks into atomic subtasks
2. Self-Critique Loop (Medium) - Review and improve output before finalizing
3. Confidence Scoring (Medium) - Rate certainty of actions
4. Post-Action Verification (Low) - Check if actions produced expected results
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ConfidenceLevel(Enum):
    """Confidence levels for actions."""
    VERY_LOW = 1      # < 20% - Need more information
    LOW = 2           # 20-40% - Uncertain, may need verification
    MEDIUM = 3        # 40-60% - Reasonable guess
    HIGH = 4          # 60-80% - Confident
    VERY_HIGH = 5     # 80-100% - Certain


@dataclass
class Subtask:
    """A decomposed subtask with dependencies."""
    id: str
    description: str
    dependencies: list[str] = field(default_factory=list)  # IDs of subtasks this depends on
    verification: str = ""  # How to verify this subtask succeeded
    status: str = "pending"  # pending, in_progress, completed, failed, skipped
    result: str = ""
    attempts: int = 0
    max_attempts: int = 2


@dataclass
class TaskDecomposition:
    """A decomposed task with ordered subtasks."""
    original_task: str
    subtasks: list[Subtask] = field(default_factory=list)
    current_index: int = 0
    rollback_points: list[int] = field(default_factory=list)  # Indices where we can safely rollback

    def next_subtask(self) -> Subtask | None:
        """Get the next pending subtask that has all dependencies met."""
        completed_ids = {st.id for st in self.subtasks if st.status == "completed"}

        for st in self.subtasks:
            if st.status == "pending":
                # Check if all dependencies are completed
                if all(dep in completed_ids for dep in st.dependencies):
                    return st
        return None

    def mark_completed(self, subtask_id: str, result: str = "") -> None:
        """Mark a subtask as completed."""
        for st in self.subtasks:
            if st.id == subtask_id:
                st.status = "completed"
                st.result = result
                break

    def mark_failed(self, subtask_id: str, error: str = "") -> None:
        """Mark a subtask as failed."""
        for st in self.subtasks:
            if st.id == subtask_id:
                st.status = "failed"
                st.result = error
                st.attempts += 1
                break

    def can_retry(self, subtask_id: str) -> bool:
        """Check if a subtask can be retried."""
        for st in self.subtasks:
            if st.id == subtask_id:
                return st.attempts < st.max_attempts
        return False

    def reset_for_retry(self, subtask_id: str) -> None:
        """Reset a subtask for retry."""
        for st in self.subtasks:
            if st.id == subtask_id:
                st.status = "pending"
                break

    def progress_str(self) -> str:
        """Get progress string like '[2/5]'."""
        completed = sum(1 for st in self.subtasks if st.status == "completed")
        total = len(self.subtasks)
        return f"[{completed}/{total}]"

    def is_complete(self) -> bool:
        """Check if all subtasks are completed."""
        return all(st.status in ("completed", "skipped") for st in self.subtasks)

    def has_failures(self) -> bool:
        """Check if any subtask has failed (and can't be retried)."""
        return any(
            st.status == "failed" and st.attempts >= st.max_attempts
            for st in self.subtasks
        )

    def to_prompt(self) -> str:
        """Format decomposition for LLM prompt."""
        lines = [f"Task: {self.original_task}", "", "Subtasks:"]
        for i, st in enumerate(self.subtasks, 1):
            status_icon = {
                "pending": "○",
                "in_progress": "◐",
                "completed": "●",
                "failed": "✗",
                "skipped": "⊘",
            }.get(st.status, "?")
            deps = f" (after: {', '.join(st.dependencies)})" if st.dependencies else ""
            lines.append(f"  {status_icon} {i}. {st.description}{deps}")
            if st.verification:
                lines.append(f"      Verify: {st.verification}")
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
    action: str  # Description of the action
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
    verification_method: str = ""  # How we verified (e.g., "file_exists", "output_contains")
    discrepancies: list[str] = field(default_factory=list)
    needs_correction: bool = False
    correction_suggestion: str = ""


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

    if tool_name == "read":
        # Read should return file contents (non-empty)
        return len(result.strip()) > 0

    if tool_name == "bash":
        # Bash success if no error output
        return True  # Already checked error indicators above

    return True


# === Task Completion Detection ===

@dataclass
class TaskCompletionCheck:
    """Result of checking if a task is complete."""
    original_task: str
    is_complete: bool = False
    accomplished: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)
    suggested_next_steps: list[str] = field(default_factory=list)
    continuation_prompt: str = ""


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
    """
    task_lower = task.lower()
    response_lower = response.lower()

    # Keywords that suggest the task should involve multiple steps
    multi_step_indicators = [
        "create a", "build a", "make a", "set up", "setup",
        "initialize", "scaffold", "generate", "implement",
        "project", "application", "app", "website", "api",
    ]

    # Keywords that suggest testing/verification should happen
    verification_indicators = [
        "test", "run", "start", "launch", "verify", "check",
        "demo", "show", "demonstrate",
    ]

    # Keywords in response that suggest premature completion
    premature_phrases = [
        "i've created", "i created", "file has been created",
        "here's the", "i've set up the basic",
        "you can now", "you should now",
        "that's it", "all done", "complete",
    ]

    # Check if this looks like a multi-step task
    is_multi_step = any(ind in task_lower for ind in multi_step_indicators)

    # Check if verification was expected but not done
    expects_verification = any(ind in task_lower for ind in verification_indicators)

    # Check for premature completion phrases
    has_premature_phrase = any(phrase in response_lower for phrase in premature_phrases)

    # Few actions taken for a multi-step task
    few_actions = len(actions_taken) < 3

    # Heuristic: if it's a multi-step task with few actions and premature phrases
    if is_multi_step and few_actions and has_premature_phrase:
        return True

    # If verification was expected but only file creation happened
    if expects_verification and few_actions:
        action_types = set()
        for action in actions_taken:
            if "write" in action or "create" in action:
                action_types.add("write")
            elif "bash" in action or "run" in action:
                action_types.add("run")
            elif "read" in action:
                action_types.add("read")

        # Only wrote files, never ran anything
        if action_types == {"write"} or action_types == {"write", "read"}:
            return True

    return False


def get_continuation_prompt(task: str, actions_taken: list[str], response: str) -> str:
    """Generate a prompt to encourage the agent to continue.

    Returns a prompt that nudges the agent to follow through.
    """
    task_lower = task.lower()

    # Determine what type of follow-up is needed
    follow_ups = []

    # Project setup tasks should initialize
    if any(kw in task_lower for kw in ["node", "npm", "javascript", "react", "vue", "next"]):
        if not any("npm" in a for a in actions_taken):
            follow_ups.append("Run `npm install` to install dependencies")
            follow_ups.append("Start the development server to verify it works")

    if any(kw in task_lower for kw in ["python", "pip", "django", "flask", "fastapi"]):
        if not any("pip" in a or "uv" in a for a in actions_taken):
            follow_ups.append("Install dependencies with pip/uv")
            follow_ups.append("Run the application to verify it works")

    # Test tasks should run tests
    if "test" in task_lower:
        if not any("test" in a or "pytest" in a or "jest" in a for a in actions_taken):
            follow_ups.append("Run the tests to verify they pass")

    # Build tasks should verify build
    if "build" in task_lower or "compile" in task_lower:
        if not any("build" in a or "compile" in a for a in actions_taken):
            follow_ups.append("Run the build to verify it succeeds")

    # Generic follow-ups for creation tasks
    if any(kw in task_lower for kw in ["create", "make", "build", "set up"]):
        if len(actions_taken) < 3:
            follow_ups.append("Verify the creation was successful")
            follow_ups.append("Demonstrate that it works as expected")

    if follow_ups:
        steps = "\n".join(f"- {step}" for step in follow_ups[:3])
        return (
            f"WAIT - You haven't finished yet. The task was: \"{task}\"\n\n"
            f"You only completed {len(actions_taken)} action(s). You should also:\n{steps}\n\n"
            f"Continue executing the remaining steps. Don't just describe what to do - USE YOUR TOOLS to do it."
        )

    # Generic continuation
    return (
        f"The task \"{task}\" may not be fully complete. "
        f"You've taken {len(actions_taken)} action(s). "
        f"Consider: Did you verify the result works? Did you test it? "
        f"If there's more to do, continue. If truly complete, explain what was accomplished."
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
