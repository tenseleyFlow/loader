"""Tests for the Sprint 06 read-only explore lane."""

from __future__ import annotations

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.llm.base import CompletionResponse, ToolCall
from loader.runtime.explore_state import ExploreStateStore
from loader.runtime.permissions import PermissionMode
from tests.helpers.runtime_harness import ScriptedBackend


class DescribingBackend(ScriptedBackend):
    async def describe_model(self) -> dict[str, object]:
        self._supports_native_tools = False
        return {"supports_native_tools": False}


@pytest.mark.asyncio
async def test_explore_mode_skips_workflow_router_and_definition_of_done(temp_dir) -> None:
    target = temp_dir / "feature.py"
    target.write_text("def important_helper():\n    return 1\n")
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I'll search for that helper.",
                tool_calls=[
                    ToolCall(
                        id="grep-1",
                        name="grep",
                        arguments={
                            "pattern": "important_helper",
                            "path": str(temp_dir),
                            "include": "*.py",
                        },
                    )
                ],
            ),
            CompletionResponse(
                content="important_helper is defined in feature.py.",
            ),
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(
            auto_context=False,
            stream=False,
            permission_mode=PermissionMode.WORKSPACE_WRITE,
        ),
        project_root=temp_dir,
    )
    events = []

    async def capture(event) -> None:
        events.append(event)

    response = await agent.run_explore(
        "Where is important_helper defined?",
        on_event=capture,
    )

    assert "feature.py" in response
    assert not any(event.type == "dod_status" for event in events)
    assert not any(event.type == "workflow_mode" for event in events)
    assert agent.last_turn_summary is not None
    assert agent.last_turn_summary.definition_of_done is None
    assert agent.last_turn_summary.workflow_mode == "explore"
    assert not (temp_dir / ".loader" / "dod").exists()


@pytest.mark.asyncio
async def test_explore_mode_denies_write_attempts_even_with_workspace_write(temp_dir) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I'll write a file.",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={
                            "file_path": str(temp_dir / "new.txt"),
                            "content": "not allowed\n",
                        },
                    )
                ],
            ),
            CompletionResponse(
                content="Explore mode is read-only, so I cannot make that change here.",
            ),
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(
            auto_context=False,
            stream=False,
            permission_mode=PermissionMode.WORKSPACE_WRITE,
        ),
        project_root=temp_dir,
    )
    events = []

    async def capture(event) -> None:
        events.append(event)

    response = await agent.run_explore(
        "Create a new file anyway.",
        on_event=capture,
    )

    tool_results = [event.content for event in events if event.type == "tool_result"]
    assert "read-only" in "\n".join(tool_results).lower()
    assert "cannot make that change" in response.lower()
    assert not (temp_dir / "new.txt").exists()


@pytest.mark.asyncio
async def test_explore_mode_refreshes_capabilities_before_request(temp_dir) -> None:
    backend = DescribingBackend(
        completions=[CompletionResponse(content="I checked the repo in read-only mode.")],
        supports_native_tools=True,
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(
            auto_context=False,
            stream=False,
            permission_mode=PermissionMode.WORKSPACE_WRITE,
        ),
        project_root=temp_dir,
    )

    response = await agent.run_explore(
        "Give me a quick repo summary.",
    )

    assert response == "I checked the repo in read-only mode."
    assert agent.capability_profile.supports_native_tools is False
    assert backend.invocations
    assert backend.invocations[0].tools is None


@pytest.mark.asyncio
async def test_explore_mode_ignores_global_allow_rules(temp_dir) -> None:
    loader_root = temp_dir / ".loader"
    loader_root.mkdir()
    (loader_root / "permission-rules.json").write_text(
        '{"allow": [{"tool": "write", "path_contains": "new.txt"}]}\n'
    )
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I'll write a file.",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={
                            "file_path": str(temp_dir / "new.txt"),
                            "content": "still not allowed\n",
                        },
                    )
                ],
            ),
            CompletionResponse(
                content="Explore mode is read-only, so I cannot make that change here.",
            ),
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(
            auto_context=False,
            stream=False,
            permission_mode=PermissionMode.ALLOW,
        ),
        project_root=temp_dir,
    )
    events = []

    async def capture(event) -> None:
        events.append(event)

    response = await agent.run_explore(
        "Create a new file anyway.",
        on_event=capture,
    )

    tool_results = [event.content for event in events if event.type == "tool_result"]
    assert "read-only" in "\n".join(tool_results).lower()
    assert "cannot make that change" in response.lower()
    assert not (temp_dir / "new.txt").exists()


@pytest.mark.asyncio
async def test_explore_mode_persists_recent_history_for_follow_up_queries(temp_dir) -> None:
    first_agent = Agent(
        backend=ScriptedBackend(
            completions=[CompletionResponse(content="Start with README.md.")]
        ),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )

    first_response = await first_agent.run_explore("Where should I start?")

    assert first_response == "Start with README.md."

    follow_up_backend = ScriptedBackend(
        completions=[CompletionResponse(content="I mentioned README.md.")]
    )
    second_agent = Agent(
        backend=follow_up_backend,
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )

    follow_up_response = await second_agent.run_explore("What file did you mention?")

    assert follow_up_response == "I mentioned README.md."
    messages = follow_up_backend.invocations[0].messages
    assert messages[1].content == "Where should I start?"
    assert messages[2].content == "Start with README.md."
    assert messages[3].content == "What file did you mention?"

    snapshot = ExploreStateStore(temp_dir).load()
    assert snapshot is not None
    assert snapshot.turn_count == 2
    assert snapshot.last_history_mode == "continue"
    assert snapshot.last_query == "What file did you mention?"
    assert snapshot.last_response == "I mentioned README.md."


@pytest.mark.asyncio
async def test_explore_mode_fresh_query_ignores_persisted_history(temp_dir) -> None:
    first_agent = Agent(
        backend=ScriptedBackend(
            completions=[CompletionResponse(content="Start with README.md.")]
        ),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    await first_agent.run_explore("Where should I start?")

    fresh_backend = ScriptedBackend(
        completions=[CompletionResponse(content="Fresh answer only.")]
    )
    second_agent = Agent(
        backend=fresh_backend,
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )

    response = await second_agent.run_explore(
        "Ignore the previous lookup.",
        fresh=True,
    )

    assert response == "Fresh answer only."
    messages = fresh_backend.invocations[0].messages
    assert len(messages) == 2
    assert messages[1].content == "Ignore the previous lookup."

    snapshot = ExploreStateStore(temp_dir).load()
    assert snapshot is not None
    assert snapshot.turn_count == 1
    assert snapshot.last_history_mode == "fresh"
    assert snapshot.last_query == "Ignore the previous lookup."
    assert snapshot.last_response == "Fresh answer only."
