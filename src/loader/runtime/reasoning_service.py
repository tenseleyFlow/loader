"""Runtime-owned confidence and verification services."""

from __future__ import annotations

from typing import Any

from ..llm.base import LLMBackend, Message, Role
from .action_reasoning import (
    CONFIDENCE_PROMPT,
    VERIFICATION_PROMPT,
    estimate_confidence_quick,
    parse_confidence,
    parse_verification,
    quick_verify,
)
from .reasoning_types import (
    ActionVerification,
    ConfidenceAssessment,
    ConfidenceLevel,
)


class RuntimeReasoningService:
    """Provide confidence scoring and verification without Agent callbacks."""

    def __init__(self, backend: LLMBackend, config: Any) -> None:
        self.backend = backend
        self.config = config

    async def assess_confidence(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: str = "",
    ) -> ConfidenceAssessment:
        """Assess confidence in a planned tool action."""

        cfg = self.config.reasoning

        if getattr(cfg, "use_quick_confidence", True):
            quick_level = estimate_confidence_quick(tool_name, tool_args, context)
            if quick_level.value >= ConfidenceLevel.MEDIUM.value:
                return ConfidenceAssessment(
                    action=f"{tool_name} with {tool_args}",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    level=quick_level,
                    reasoning="Quick heuristic assessment",
                )

        action = f"Call {tool_name} with arguments: {tool_args}"
        prompt = CONFIDENCE_PROMPT.format(
            action=action,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context[-2000:] if context else "No prior context",
        )
        response = await self.backend.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            tools=None,
            temperature=0.3,
            max_tokens=300,
        )
        return parse_confidence(response.content, tool_name, tool_args)

    async def verify_action(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        """Verify that a completed tool action achieved its goal."""

        cfg = self.config.reasoning

        if getattr(cfg, "use_quick_verification", True):
            if quick_verify(tool_name, tool_args, result):
                return ActionVerification(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    expected_outcome=expected or "Success",
                    actual_result=result[:500],
                    verified=True,
                    verification_method="quick_heuristic",
                )

        prompt = VERIFICATION_PROMPT.format(
            tool_name=tool_name,
            tool_args=tool_args,
            expected=expected or "The action should complete successfully",
            result=result[:2000],
        )
        response = await self.backend.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            tools=None,
            temperature=0.3,
            max_tokens=300,
        )
        return parse_verification(response.content, tool_name, tool_args, expected, result)
