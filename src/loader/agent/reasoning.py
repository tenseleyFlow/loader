"""Reasoning stages for enhanced agent cognition.

This module implements multiple reasoning stages that can be optionally
enabled to improve the agent's decision-making:

1. Decomposition (High complexity) - Break complex tasks into atomic subtasks
2. Self-Critique Loop (Medium) - Review and improve output before finalizing
3. Confidence Scoring (Medium) - Rate certainty of actions
4. Post-Action Verification (Low) - Check if actions produced expected results
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# === Query Classification ===

def is_conversational(message: str) -> bool:
    """Detect if a message is conversational rather than a task.

    Returns True for greetings, casual chat, simple questions about the agent.
    These don't need tool calling - just a quick response.
    """
    msg = message.lower().strip()

    # Very short messages are usually conversational
    if len(msg) < 15:
        # Greetings
        greetings = [
            "hi", "hello", "hey", "yo", "sup", "hiya", "howdy",
            "ello", "hallo", "greetings", "good morning", "good afternoon",
            "good evening", "morning", "evening", "afternoon",
            "what's up", "whats up", "wassup", "how are you",
            "how's it going", "hows it going",
        ]
        if any(msg.startswith(g) or msg == g for g in greetings):
            return True

    # Questions about the agent itself
    agent_questions = [
        "who are you", "what are you", "what can you do",
        "how do you work", "what is loader", "what's loader",
        "help", "what is this", "how does this work",
    ]
    if any(q in msg for q in agent_questions):
        return True

    # Casual/social messages
    casual = [
        "thanks", "thank you", "thx", "ty",
        "cool", "nice", "great", "awesome", "ok", "okay",
        "bye", "goodbye", "see you", "later", "cya",
        "lol", "haha", "hehe", "lmao",
        "please", "sorry", "oops",
    ]
    if msg in casual or any(msg == c for c in casual):
        return True

    # Messages that are clearly NOT conversational (tasks)
    task_indicators = [
        "create", "make", "build", "write", "edit", "delete", "remove",
        "run", "execute", "install", "fix", "debug", "test", "check",
        "find", "search", "show", "list", "read", "open", "close",
        "add", "update", "change", "modify", "refactor", "implement",
        "file", "folder", "directory", "code", "function", "class",
        "git", "npm", "pip", "python", "node", "bash", "command",
    ]
    if any(ind in msg for ind in task_indicators):
        return False

    # Short messages without task indicators are likely conversational
    if len(msg) < 30 and not any(c in msg for c in [".", "/", "\\", "`"]):
        return True

    return False


def estimate_complexity(message: str) -> str:
    """Estimate query complexity for token budgeting.

    Returns: "trivial", "simple", "moderate", or "complex"
    """
    msg = message.lower()
    word_count = len(message.split())

    # Trivial: greetings, thanks, very short
    if is_conversational(message) or word_count < 5:
        return "trivial"

    # Complex indicators
    complex_indicators = [
        "project", "application", "website", "api", "database",
        "refactor", "migrate", "upgrade", "implement", "design",
        "multiple", "several", "all", "entire", "whole",
        "and then", "after that", "also", "as well",
    ]
    complex_count = sum(1 for ind in complex_indicators if ind in msg)

    if complex_count >= 2 or word_count > 50:
        return "complex"

    # Simple indicators
    simple_indicators = [
        "what is", "how do", "show me", "list", "read",
        "single", "one", "just", "only", "quick",
    ]
    if any(ind in msg for ind in simple_indicators) and word_count < 20:
        return "simple"

    return "moderate"


def get_token_budget(complexity: str) -> tuple[int, int]:
    """Get (max_tokens, context_tokens) for a complexity level."""
    budgets = {
        "trivial": (256, 2048),    # Quick response, minimal context
        "simple": (512, 4096),     # Short response, some context
        "moderate": (1024, 8192),  # Normal response
        "complex": (2048, 16384),  # Full response, full context
    }
    return budgets.get(complexity, (1024, 8192))


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
        "add", "write", "develop", "design", "help me",
    ]

    # Keywords that suggest testing/verification should happen
    verification_indicators = [
        "test", "run", "start", "launch", "verify", "check",
        "demo", "show", "demonstrate", "work", "function",
    ]

    # Keywords in response that suggest premature completion
    premature_phrases = [
        "i've created", "i created", "file has been created",
        "here's the", "i've set up the basic", "i've written",
        "you can now", "you should now", "you can run",
        "that's it", "all done", "complete", "finished",
        "let me know", "feel free to", "hope this helps",
        "is there anything else",
    ]

    # Check if this looks like a multi-step task
    is_multi_step = any(ind in task_lower for ind in multi_step_indicators)

    # Check if verification was expected but not done
    expects_verification = any(ind in task_lower for ind in verification_indicators)

    # Check for premature completion phrases
    has_premature_phrase = any(phrase in response_lower for phrase in premature_phrases)

    # Action count thresholds
    few_actions = len(actions_taken) < 3
    very_few_actions = len(actions_taken) < 2

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

    # More aggressive detection:

    # 1. Multi-step task with premature phrases and few actions
    if is_multi_step and has_premature_phrase and few_actions:
        return True

    # 2. Multi-step task with very few actions (regardless of phrases)
    if is_multi_step and very_few_actions:
        return True

    # 3. Only wrote/edited files but never ran/tested anything
    if action_types and action_types <= {"write", "edit", "read"} and few_actions:
        # Wrote files but never executed bash to test
        if "write" in action_types or "edit" in action_types:
            return True

    # 4. Verification expected but no bash commands run
    if expects_verification and "bash" not in action_types:
        return True

    # 5. Response has chatbot-style "let me know" phrases
    chatbot_phrases = ["let me know", "feel free", "hope this", "happy to help"]
    if any(phrase in response_lower for phrase in chatbot_phrases):
        return True

    # 6. Response is very short but task seems substantial
    if len(response) < 200 and is_multi_step and len(actions_taken) > 0:
        return True

    return False


def get_continuation_prompt(task: str, actions_taken: list[str], response: str) -> str:
    """Generate a prompt to encourage the agent to continue.

    Returns a prompt that nudges the agent to follow through.
    """
    task_lower = task.lower()
    actions_str = ", ".join(a.split(":")[0] for a in actions_taken[-5:]) if actions_taken else "none"

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
            f"STOP - You are NOT done. The task was: \"{task}\"\n\n"
            f"Actions so far: {actions_str}\n"
            f"You MUST also:\n{steps}\n\n"
            f"DO NOT respond with text. USE YOUR TOOLS NOW to complete these steps."
        )

    # Generic continuation - be forceful
    return (
        f"INCOMPLETE. Task: \"{task}\"\n"
        f"Actions taken: {actions_str} ({len(actions_taken)} total)\n\n"
        f"You stopped too early. What about:\n"
        f"- Testing/verifying the result?\n"
        f"- Running what you created?\n"
        f"- Installing dependencies?\n\n"
        f"USE YOUR TOOLS to continue. Do not just describe - EXECUTE."
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


# === Rollback Planning ===

class RollbackType(Enum):
    """Types of rollback actions."""
    FILE_RESTORE = auto()      # Restore file from backup
    FILE_DELETE = auto()       # Delete a created file
    GIT_CHECKOUT = auto()      # git checkout to restore
    GIT_STASH_POP = auto()     # git stash pop to restore
    COMMAND_UNDO = auto()      # Run an undo command
    NO_ROLLBACK = auto()       # Cannot be rolled back


@dataclass
class RollbackAction:
    """A single rollback action."""
    type: RollbackType
    description: str
    file_path: str = ""
    original_content: str = ""  # For file restores
    undo_command: str = ""      # For command undos
    executed: bool = False


@dataclass
class RollbackPlan:
    """Plan for rolling back a series of actions."""
    actions: list[RollbackAction] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    modified_files: dict[str, str] = field(default_factory=dict)  # path -> original content
    git_stashed: bool = False
    can_rollback: bool = True

    def add_file_creation(self, file_path: str) -> None:
        """Track a file that was created (can be deleted to rollback)."""
        self.created_files.append(file_path)
        self.actions.append(RollbackAction(
            type=RollbackType.FILE_DELETE,
            description=f"Delete created file: {file_path}",
            file_path=file_path,
        ))

    def add_file_modification(self, file_path: str, original_content: str) -> None:
        """Track a file modification (can restore original content)."""
        if file_path not in self.modified_files:
            self.modified_files[file_path] = original_content
            self.actions.append(RollbackAction(
                type=RollbackType.FILE_RESTORE,
                description=f"Restore original: {file_path}",
                file_path=file_path,
                original_content=original_content,
            ))

    def add_git_stash(self) -> None:
        """Track that we stashed git changes."""
        if not self.git_stashed:
            self.git_stashed = True
            self.actions.append(RollbackAction(
                type=RollbackType.GIT_STASH_POP,
                description="Restore stashed changes: git stash pop",
            ))

    def add_command_undo(self, description: str, undo_command: str) -> None:
        """Track a command that can be undone."""
        self.actions.append(RollbackAction(
            type=RollbackType.COMMAND_UNDO,
            description=description,
            undo_command=undo_command,
        ))

    def add_no_rollback(self, description: str) -> None:
        """Track an action that cannot be rolled back."""
        self.can_rollback = False
        self.actions.append(RollbackAction(
            type=RollbackType.NO_ROLLBACK,
            description=f"Cannot undo: {description}",
        ))

    def get_rollback_steps(self) -> list[str]:
        """Get human-readable rollback steps (in reverse order)."""
        steps = []
        for action in reversed(self.actions):
            if action.type == RollbackType.FILE_DELETE:
                steps.append(f"Delete: {action.file_path}")
            elif action.type == RollbackType.FILE_RESTORE:
                steps.append(f"Restore: {action.file_path}")
            elif action.type == RollbackType.GIT_CHECKOUT:
                steps.append(f"Git restore: {action.file_path}")
            elif action.type == RollbackType.GIT_STASH_POP:
                steps.append("Run: git stash pop")
            elif action.type == RollbackType.COMMAND_UNDO:
                steps.append(f"Run: {action.undo_command}")
            elif action.type == RollbackType.NO_ROLLBACK:
                steps.append(f"⚠ {action.description}")
        return steps

    def to_prompt(self) -> str:
        """Format rollback plan for display."""
        if not self.actions:
            return "No rollback actions recorded."

        lines = ["Rollback plan:"]
        for i, step in enumerate(self.get_rollback_steps(), 1):
            lines.append(f"  {i}. {step}")

        if not self.can_rollback:
            lines.append("\n⚠ Warning: Some actions cannot be undone!")

        return "\n".join(lines)


def is_destructive_tool(tool_name: str, tool_args: dict) -> bool:
    """Check if a tool call is potentially destructive."""
    if tool_name == "write":
        return True  # Creating/overwriting files

    if tool_name == "edit":
        return True  # Modifying files

    if tool_name == "bash":
        command = tool_args.get("command", "").lower()
        destructive_patterns = [
            "rm ", "rm -", "rmdir",           # Delete
            "mv ", "rename",                   # Move/rename
            "> ", ">>",                        # Redirect/overwrite
            "chmod", "chown",                  # Permissions
            "git reset", "git checkout",       # Git destructive
            "git clean", "git stash",
            "npm uninstall", "pip uninstall",  # Package removal
            "drop ", "delete ", "truncate",    # Database
        ]
        return any(p in command for p in destructive_patterns)

    return False


def get_undo_command(command: str) -> str | None:
    """Get the undo command for a bash command, if possible."""
    command_lower = command.lower().strip()

    # mkdir -> rmdir (only for empty dirs)
    if command_lower.startswith("mkdir "):
        dir_path = command.split("mkdir", 1)[1].strip().split()[0]
        return f"rmdir {dir_path}"

    # git stash -> git stash pop
    if "git stash" in command_lower and "pop" not in command_lower:
        return "git stash pop"

    # npm install -> npm uninstall (for specific packages)
    if "npm install " in command_lower or "npm i " in command_lower:
        # Extract package name
        parts = command.split()
        for i, part in enumerate(parts):
            if part in ("install", "i") and i + 1 < len(parts):
                pkg = parts[i + 1]
                if not pkg.startswith("-"):
                    return f"npm uninstall {pkg}"

    # pip install -> pip uninstall
    if "pip install " in command_lower or "pip3 install " in command_lower:
        parts = command.split()
        for i, part in enumerate(parts):
            if part == "install" and i + 1 < len(parts):
                pkg = parts[i + 1]
                if not pkg.startswith("-"):
                    return f"pip uninstall -y {pkg}"

    return None


async def create_rollback_plan_for_action(
    tool_name: str,
    tool_args: dict,
    read_file_func,  # async function to read file contents
) -> RollbackAction | None:
    """Create a rollback action for a tool call.

    Args:
        tool_name: Name of the tool
        tool_args: Tool arguments
        read_file_func: Async function that reads file contents given a path

    Returns:
        RollbackAction or None if no rollback needed/possible
    """
    import os

    if tool_name == "write":
        file_path = tool_args.get("file_path", "")
        if not file_path:
            return None

        # Check if file exists (we'd be overwriting)
        if os.path.exists(file_path):
            try:
                original = await read_file_func(file_path)
                return RollbackAction(
                    type=RollbackType.FILE_RESTORE,
                    description=f"Restore original: {file_path}",
                    file_path=file_path,
                    original_content=original,
                )
            except Exception:
                return RollbackAction(
                    type=RollbackType.NO_ROLLBACK,
                    description=f"Could not backup: {file_path}",
                    file_path=file_path,
                )
        else:
            # New file - can delete to rollback
            return RollbackAction(
                type=RollbackType.FILE_DELETE,
                description=f"Delete created file: {file_path}",
                file_path=file_path,
            )

    if tool_name == "edit":
        file_path = tool_args.get("file_path", "")
        if not file_path:
            return None

        try:
            original = await read_file_func(file_path)
            return RollbackAction(
                type=RollbackType.FILE_RESTORE,
                description=f"Restore original: {file_path}",
                file_path=file_path,
                original_content=original,
            )
        except Exception:
            return RollbackAction(
                type=RollbackType.NO_ROLLBACK,
                description=f"Could not backup: {file_path}",
                file_path=file_path,
            )

    if tool_name == "bash":
        command = tool_args.get("command", "")
        undo = get_undo_command(command)
        if undo:
            return RollbackAction(
                type=RollbackType.COMMAND_UNDO,
                description=f"Undo with: {undo}",
                undo_command=undo,
            )
        elif is_destructive_tool(tool_name, tool_args):
            return RollbackAction(
                type=RollbackType.NO_ROLLBACK,
                description=f"Cannot undo: {command[:50]}...",
            )

    return None


async def execute_rollback(plan: RollbackPlan, write_file_func, run_command_func) -> list[str]:
    """Execute a rollback plan.

    Args:
        plan: The rollback plan to execute
        write_file_func: Async function to write file contents
        run_command_func: Async function to run shell commands

    Returns:
        List of results/errors from rollback actions
    """
    import os

    results = []

    # Execute in reverse order
    for action in reversed(plan.actions):
        if action.executed:
            continue

        try:
            if action.type == RollbackType.FILE_DELETE:
                if os.path.exists(action.file_path):
                    os.remove(action.file_path)
                    results.append(f"✓ Deleted: {action.file_path}")
                    action.executed = True

            elif action.type == RollbackType.FILE_RESTORE:
                await write_file_func(action.file_path, action.original_content)
                results.append(f"✓ Restored: {action.file_path}")
                action.executed = True

            elif action.type == RollbackType.COMMAND_UNDO:
                result = await run_command_func(action.undo_command)
                results.append(f"✓ Ran: {action.undo_command}")
                action.executed = True

            elif action.type == RollbackType.GIT_STASH_POP:
                result = await run_command_func("git stash pop")
                results.append("✓ Restored git stash")
                action.executed = True

            elif action.type == RollbackType.NO_ROLLBACK:
                results.append(f"⚠ Skipped (no rollback): {action.description}")

        except Exception as e:
            results.append(f"✗ Failed {action.description}: {e}")

    return results
