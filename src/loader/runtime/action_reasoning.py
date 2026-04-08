"""Runtime-owned prompts and heuristics for action reasoning."""

from __future__ import annotations

import re

from .reasoning_types import (
    ActionVerification,
    ConfidenceAssessment,
    ConfidenceLevel,
)

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


def parse_confidence(
    response: str,
    tool_name: str,
    tool_args: dict,
) -> ConfidenceAssessment:
    """Parse LLM response into ConfidenceAssessment."""

    import json

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
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
    response: str,
    tool_name: str,
    tool_args: dict,
    expected: str,
    result: str,
) -> ActionVerification:
    """Parse LLM response into ActionVerification."""

    import json

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
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


def estimate_confidence_quick(
    tool_name: str,
    tool_args: dict,
    context: str = "",
) -> ConfidenceLevel:
    """Estimate action confidence with heuristics before an LLM call."""

    del context

    if tool_name in {"read", "glob", "grep", "git"}:
        return ConfidenceLevel.HIGH

    if tool_name == "write":
        file_path = tool_args.get("file_path", "")
        if not file_path:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.MEDIUM

    if tool_name == "edit":
        old_string = tool_args.get("old_string", "")
        if not old_string:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.MEDIUM

    if tool_name == "patch":
        hunks = tool_args.get("hunks", [])
        if not hunks:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.MEDIUM

    if tool_name == "bash":
        command = tool_args.get("command", "")
        dangerous = ["rm -rf", "sudo", "chmod 777", "dd if=", "> /dev/"]
        if any(pattern in command for pattern in dangerous):
            return ConfidenceLevel.VERY_LOW
        safe_read = ["ls", "cat", "grep", "find", "pwd", "echo", "head", "tail"]
        if any(command.strip().startswith(prefix) for prefix in safe_read):
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.MEDIUM


def quick_verify(tool_name: str, tool_args: dict, result: str) -> bool:
    """Estimate whether a tool action succeeded without an LLM call."""

    del tool_args

    result_lower = result.lower()
    error_indicators = [
        "error:",
        "failed",
        "not found",
        "permission denied",
        "no such file",
        "command not found",
        "exception",
        "traceback",
        "fatal:",
        "cannot",
    ]
    if any(indicator in result_lower for indicator in error_indicators):
        return False

    if tool_name == "write":
        return "created" in result_lower or "wrote" in result_lower or len(result) < 200
    if tool_name in {"edit", "patch"}:
        return "edited" in result_lower or "patched" in result_lower or "+" in result or "-" in result
    if tool_name in {"read", "git"}:
        return len(result.strip()) > 0
    if tool_name == "bash":
        return True

    return True
