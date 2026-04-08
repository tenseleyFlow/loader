"""Tests for runtime-owned confidence and verification services."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from loader.llm.base import CompletionResponse
from loader.runtime.reasoning_service import RuntimeReasoningService
from tests.helpers.runtime_harness import ScriptedBackend


def build_config(*, use_quick_confidence: bool = True, use_quick_verification: bool = True):
    return SimpleNamespace(
        reasoning=SimpleNamespace(
            use_quick_confidence=use_quick_confidence,
            use_quick_verification=use_quick_verification,
        )
    )


@pytest.mark.asyncio
async def test_runtime_reasoning_service_uses_quick_confidence_without_backend() -> None:
    backend = ScriptedBackend()
    service = RuntimeReasoningService(backend, build_config())

    assessment = await service.assess_confidence(
        "read",
        {"file_path": "README.md"},
        "Please inspect the docs first.",
    )

    assert assessment.score >= 3
    assert assessment.reasoning == "Quick heuristic assessment"
    assert backend.invocations == []


@pytest.mark.asyncio
async def test_runtime_reasoning_service_calls_backend_for_low_confidence_actions() -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content=(
                    '{"confidence": 2, "reasoning": "This is risky.", '
                    '"risks": ["Potential deletion"], "mitigations": ["Inspect first"]}'
                )
            )
        ]
    )
    service = RuntimeReasoningService(backend, build_config())

    assessment = await service.assess_confidence(
        "bash",
        {"command": "rm -rf build"},
        "Need to clean up artifacts.",
    )

    assert assessment.score == 2
    assert assessment.reasoning == "This is risky."
    assert len(backend.invocations) == 1


@pytest.mark.asyncio
async def test_runtime_reasoning_service_uses_quick_verification_without_backend() -> None:
    backend = ScriptedBackend()
    service = RuntimeReasoningService(backend, build_config())

    verification = await service.verify_action(
        "read",
        {"file_path": "README.md"},
        "loader docs",
    )

    assert verification.verified is True
    assert verification.verification_method == "quick_heuristic"
    assert backend.invocations == []
