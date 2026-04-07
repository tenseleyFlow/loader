"""Runtime integration coverage for Sprint 04 workflow routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import AgentConfig
from loader.llm.base import CompletionResponse, ToolCall
from tests.helpers.runtime_harness import ScriptedBackend, run_scenario


def non_streaming_config() -> AgentConfig:
    """Shared config for deterministic workflow-mode runtime tests."""

    return AgentConfig(auto_context=False, stream=False, max_iterations=8)


def workflow_modes(run) -> list[str]:
    """Return emitted workflow modes in order."""

    return [
        event.workflow_mode
        for event in run.events
        if event.type == "workflow_mode" and event.workflow_mode
    ]


def artifact_kinds(run) -> list[str]:
    """Return emitted artifact kinds in order."""

    return [
        event.artifact_kind
        for event in run.events
        if event.type == "artifact" and event.artifact_kind
    ]


@pytest.mark.asyncio
async def test_ambiguous_prompt_routes_to_clarify_and_persists_brief(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I need one clarification before I proceed.",
                tool_calls=[
                    ToolCall(
                        id="ask-1",
                        name="AskUserQuestion",
                        arguments={
                            "question": "What should stay out of scope for this Loader improvement?",
                        },
                    )
                ],
            ),
            CompletionResponse(
                content="\n".join(
                    [
                        "## Task Statement",
                        "Improve Loader so it feels more like claw-code.",
                        "",
                        "## Desired Outcome",
                        "- Make Loader more reliable without broad redesign.",
                        "",
                        "## In Scope",
                        "- Tighten the runtime workflow around the user-facing goal.",
                        "",
                        "## Non Goals",
                        "- Rebuild unrelated subsystems.",
                        "",
                        "## Decision Boundaries",
                        "- Escalate before changing unrelated UX patterns.",
                        "",
                        "## Constraints",
                        "- Stay within the current repository.",
                        "",
                        "## Likely Touchpoints",
                        "- Runtime entry points and prompt behavior.",
                        "",
                        "## Assumptions",
                        "- The user wants a narrow runtime-quality improvement.",
                        "",
                        "## Acceptance Criteria",
                        "- The improvement stays focused on runtime behavior.",
                    ]
                )
            ),
            CompletionResponse(content="I have the brief and can move forward."),
        ]
    )

    async def answer(question: str, options: list[str] | None) -> str:
        assert "out of scope" in question.lower()
        assert options is None
        return "Do not redesign the whole interface."

    run = await run_scenario(
        "Improve Loader so it feels more like claw-code.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
        on_user_question=answer,
    )

    dod = run.agent.last_turn_summary.definition_of_done
    assert dod is not None
    assert workflow_modes(run)[:2] == ["clarify", "execute"]
    assert artifact_kinds(run) == ["clarify_brief"]
    assert dod.clarify_brief is not None
    assert Path(dod.clarify_brief).exists()
    brief_markdown = Path(dod.clarify_brief).read_text()
    assert "single-question clarify brief" in brief_markdown
    assert "return control to `execute` mode" in brief_markdown
    assert run.agent.session.workflow_artifact_status == "active"
    assert run.agent.session.workflow_artifact_sources == ["clarify_brief"]
    assert "runtime behavior" in dod.acceptance_criteria[0].lower()
    assert "## Clarify Mode" in backend.invocations[0].messages[0].content


@pytest.mark.asyncio
async def test_complex_prompt_routes_to_plan_and_uses_verification_artifact(
    temp_dir: Path,
) -> None:
    target = temp_dir / "planned.txt"
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="\n".join(
                    [
                        "# Implementation Plan",
                        "",
                        "## File Changes",
                        f"- Create {target.name} in the workspace root.",
                        "",
                        "## Execution Order",
                        f"1. Write {target.name}.",
                        "2. Confirm the file exists.",
                        "",
                        "## Risks",
                        "- Writing the wrong file path.",
                        "",
                        "<<<VERIFICATION>>>",
                        "",
                        "# Verification Plan",
                        "",
                        "## Acceptance Criteria",
                        f"- {target.name} exists in the workspace root.",
                        "",
                        "## Verification Commands",
                        f"- `test -f {target}`",
                        "",
                        "## Notes",
                        "- Use a deterministic file existence check.",
                    ]
                )
            ),
            CompletionResponse(
                content="I'll create the file now.",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={
                            "file_path": str(target),
                            "content": "planned output\n",
                        },
                    )
                ],
            ),
            CompletionResponse(content="The file is in place."),
        ]
    )

    run = await run_scenario(
        "Implement a persistent workflow mode router with clarify artifacts, "
        "planning artifacts, and verification-plan wiring in the runtime.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    dod = run.agent.last_turn_summary.definition_of_done
    assert dod is not None
    assert workflow_modes(run)[:3] == ["plan", "execute", "verify"]
    assert artifact_kinds(run) == ["implementation_plan", "verification_plan"]
    assert dod.implementation_plan is not None
    assert dod.verification_plan is not None
    assert Path(dod.implementation_plan).exists()
    assert Path(dod.verification_plan).exists()
    implementation_markdown = Path(dod.implementation_plan).read_text()
    verification_markdown = Path(dod.verification_plan).read_text()
    assert "single-pass planning artifact generation" in implementation_markdown
    assert "planner/critic consensus loop" in implementation_markdown
    assert "single-pass planning artifact generation" in verification_markdown
    assert run.agent.session.workflow_artifact_status == "active"
    assert run.agent.session.workflow_artifact_sources == [
        "implementation_plan",
        "verification_plan",
    ]
    assert dod.verification_commands == [f"test -f {target}"]
    assert "## Plan Mode" in backend.invocations[0].messages[0].content
    verify_calls = [
        event
        for event in run.events
        if event.type == "tool_call" and event.phase == "verification"
    ]
    assert [event.tool_args["command"] for event in verify_calls] == [f"test -f {target}"]


@pytest.mark.asyncio
async def test_verify_failure_returns_to_execute_without_retriggering_plan(
    temp_dir: Path,
) -> None:
    target = temp_dir / "retry.txt"
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="\n".join(
                    [
                        "# Implementation Plan",
                        "",
                        "## File Changes",
                        f"- Create {target.name}.",
                        "",
                        "## Execution Order",
                        f"1. Write {target.name}.",
                        "2. Fix it if verification fails.",
                        "",
                        "## Risks",
                        "- Initial content may be wrong.",
                        "",
                        "<<<VERIFICATION>>>",
                        "",
                        "# Verification Plan",
                        "",
                        "## Acceptance Criteria",
                        "- The file contains the word fixed.",
                        "",
                        "## Verification Commands",
                        f"- `grep -q fixed {target}`",
                        "",
                        "## Notes",
                        "- Retry if the first write misses the target string.",
                    ]
                )
            ),
            CompletionResponse(
                content="I'll write the first draft.",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={
                            "file_path": str(target),
                            "content": "draft output\n",
                        },
                    )
                ],
            ),
            CompletionResponse(content="First draft is written."),
            CompletionResponse(
                content="I'll correct the file.",
                tool_calls=[
                    ToolCall(
                        id="write-2",
                        name="write",
                        arguments={
                            "file_path": str(target),
                            "content": "fixed output\n",
                        },
                    )
                ],
            ),
            CompletionResponse(content="The file now contains the fixed output."),
        ]
    )

    run = await run_scenario(
        "Implement a persistent workflow mode router with clarify artifacts, "
        "planning artifacts, and verification-plan wiring in the runtime.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    modes = workflow_modes(run)
    assert modes.count("plan") == 1
    assert modes.count("clarify") == 0
    assert modes.count("execute") >= 2
    assert modes.count("verify") >= 2
    assert "fixed output" in target.read_text()
