"""Deterministic runtime parity coverage for the current Loader loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.llm.base import CompletionResponse, Role, StreamChunk, ToolCall
from loader.runtime.capabilities import resolve_capability_profile
from tests.helpers.runtime_harness import ScriptedBackend, run_scenario

SCENARIO_NAMES = [
    "streaming_text",
    "read_file_roundtrip",
    "multi_tool_turn_roundtrip",
    "turn_summary_smoke_for_multi_tool_turn",
    "write_file_allowed",
    "write_file_denied",
    "bash_stdout_roundtrip",
    "bash_confirmation_prompt_approved",
    "bash_confirmation_prompt_denied",
    "raw_json_tool_call_fallback",
    "native_and_raw_tool_paths_share_executor_trace",
    "backend_capability_probe_refreshes_native_tool_mode",
    "run_streaming_delegates_to_primary_runtime",
    "completion_check_continuation",
    "tool_result_contract_regression",
]


def load_manifest() -> list[dict[str, str]]:
    """Load the auditable parity scenario manifest."""

    manifest_path = Path(__file__).parent / "fixtures" / "runtime_parity_manifest.json"
    return json.loads(manifest_path.read_text())


def non_streaming_config(*, completion_check: bool = False) -> AgentConfig:
    """Shared config for deterministic complete() tests."""

    config = AgentConfig(auto_context=False, stream=False, max_iterations=8)
    config.reasoning.completion_check = completion_check
    return config


def native_tool_response(
    *tool_calls: ToolCall,
    content: str = "Using tools.",
) -> CompletionResponse:
    """Build a completion that includes native tool calls."""

    return CompletionResponse(content=content, tool_calls=list(tool_calls))


def final_response(content: str) -> CompletionResponse:
    """Build a completion with no further tool calls."""

    return CompletionResponse(content=content)


def tool_event_names(run) -> list[str]:
    """Return emitted tool event names in order."""

    return [
        event.tool_name
        for event in run.events
        if event.type == "tool_call" and event.tool_name
    ]


def tool_result_messages(run) -> list[str]:
    """Return emitted tool result messages in order."""

    return [event.content for event in run.events if event.type == "tool_result"]


def trace_event_names(run) -> list[str]:
    """Return recorded runtime trace event names."""

    summary = run.agent.last_turn_summary
    assert summary is not None
    return [event.name for event in summary.trace]


@pytest.mark.asyncio
async def test_runtime_parity_manifest_matches_implemented_cases() -> None:
    manifest_names = [entry["name"] for entry in load_manifest()]
    assert manifest_names == SCENARIO_NAMES


@pytest.mark.asyncio
async def test_streaming_text_scenario() -> None:
    backend = ScriptedBackend(
        streams=[
            [
                StreamChunk(content="Mock streaming ", is_done=False),
                StreamChunk(
                    content="says hello from Loader.",
                    full_content="Mock streaming says hello from Loader.",
                    is_done=True,
                ),
            ]
        ]
    )

    run = await run_scenario("hello there", backend, config=AgentConfig(auto_context=False))

    assert run.response == "Mock streaming says hello from Loader."
    assert [call.mode for call in run.invocations] == ["stream"]
    assert not tool_event_names(run)


@pytest.mark.asyncio
async def test_read_file_roundtrip(temp_dir: Path) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("alpha parity line\nbeta line\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)}),
                content="I'll inspect that file.",
            ),
            final_response("The file contains alpha parity line."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file and summarize it.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert "alpha parity line" in run.response
    assert tool_event_names(run) == ["read"]
    assert any("alpha parity line" in message for message in tool_result_messages(run))
    assert len(run.invocations) == 2
    assert any(message.role == Role.TOOL for message in run.invocations[1].messages)


@pytest.mark.asyncio
async def test_multi_tool_turn_roundtrip(temp_dir: Path) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("alpha parity line\nbeta line\ngamma parity line\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)}),
                ToolCall(
                    id="grep-1",
                    name="grep",
                    arguments={"pattern": "parity", "path": str(fixture)},
                ),
                content="I'll inspect the file and count parity matches.",
            ),
            final_response("The file has two parity lines, including alpha parity line."),
        ]
    )

    run = await run_scenario(
        "Inspect the fixture and find parity lines.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read", "grep"]
    assert len(tool_result_messages(run)) == 2
    assert "two parity lines" in run.response


@pytest.mark.asyncio
async def test_turn_summary_smoke_for_multi_tool_turn(temp_dir: Path) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("alpha parity line\nbeta line\ngamma parity line\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)}),
                ToolCall(
                    id="grep-1",
                    name="grep",
                    arguments={"pattern": "parity", "path": str(fixture)},
                ),
                content="I'll inspect the file and count parity matches.",
            ),
            final_response("The file has two parity lines, including alpha parity line."),
        ]
    )

    run = await run_scenario(
        "Inspect the fixture and find parity lines.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    summary = run.agent.last_turn_summary
    assert summary is not None
    assert summary.final_response == run.response
    assert summary.iterations == 2
    assert len(summary.assistant_messages) == 2
    assert len(summary.tool_result_messages) == 2
    assert "assistant.tool_batch" in trace_event_names(run)


@pytest.mark.asyncio
async def test_write_file_allowed(temp_dir: Path) -> None:
    target = temp_dir / "allowed.txt"
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "hello from loader\n"},
                ),
                content="I'll create the file now.",
            ),
            final_response("Successfully created the file."),
        ]
    )

    run = await run_scenario(
        "Create allowed.txt with a greeting.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert target.read_text() == "hello from loader\n"
    assert "Successfully created the file." in run.response
    assert tool_event_names(run) == ["write"]


@pytest.mark.asyncio
async def test_write_file_denied(temp_dir: Path) -> None:
    target = temp_dir / "denied.txt"
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "should not exist\n"},
                ),
                content="I'll create the file if you approve it.",
            ),
            final_response("I skipped the write as requested."),
        ]
    )

    async def deny_confirmation(tool_name: str, message: str, details: str) -> bool:
        assert tool_name == "write"
        assert "Write to file" in message
        assert details
        return False

    run = await run_scenario(
        "Create denied.txt with a greeting.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
        on_confirmation=deny_confirmation,
    )

    assert not target.exists()
    assert "skipped the write" in run.response.lower()
    assert any(event.type == "confirmation" for event in run.events)


@pytest.mark.asyncio
async def test_bash_stdout_roundtrip(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(temp_dir)
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="bash-1", name="bash", arguments={"command": "pwd"}),
                content="I'll check the current directory.",
            ),
            final_response("Confirmed the working directory."),
        ]
    )

    run = await run_scenario(
        "Tell me the current directory.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert str(temp_dir) in tool_result_messages(run)[0]
    assert "Confirmed the working directory." in run.response


@pytest.mark.asyncio
async def test_bash_confirmation_prompt_approved(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(temp_dir)
    target = temp_dir / "approved.txt"
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="bash-1", name="bash", arguments={"command": "touch approved.txt"}),
                content="I'll create the file after approval.",
            ),
            final_response("The shell command completed."),
        ]
    )

    async def approve_confirmation(tool_name: str, message: str, details: str) -> bool:
        assert tool_name == "bash"
        assert "Run command" in message
        assert "touch approved.txt" in details
        return True

    run = await run_scenario(
        "Create approved.txt using bash.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
        on_confirmation=approve_confirmation,
    )

    assert target.exists()
    assert "shell command completed" in run.response.lower()
    assert any(event.type == "confirmation" for event in run.events)


@pytest.mark.asyncio
async def test_bash_confirmation_prompt_denied(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(temp_dir)
    target = temp_dir / "denied-bash.txt"
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="bash-1", name="bash", arguments={"command": "touch denied-bash.txt"}),
                content="I'll create the file if you allow it.",
            ),
            final_response("I left the shell command undone."),
        ]
    )

    async def deny_confirmation(tool_name: str, message: str, details: str) -> bool:
        assert tool_name == "bash"
        assert "touch denied-bash.txt" in details
        return False

    run = await run_scenario(
        "Create denied-bash.txt using bash.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
        on_confirmation=deny_confirmation,
    )

    assert not target.exists()
    assert "left the shell command undone" in run.response.lower()
    assert any(event.type == "confirmation" for event in run.events)


@pytest.mark.asyncio
async def test_raw_json_tool_call_fallback(temp_dir: Path) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("alpha parity line\n")
    raw_json = f'{{"name": "read", "arguments": {{"file_path": "{fixture}"}}}}'

    backend = ScriptedBackend(
        streams=[
            [
                StreamChunk(content=raw_json[:25], is_done=False),
                StreamChunk(content=raw_json[25:], full_content=raw_json, is_done=True),
            ],
            [
                StreamChunk(
                    content="Recovered the raw JSON tool call and read the file.",
                    full_content="Recovered the raw JSON tool call and read the file.",
                    is_done=True,
                )
            ],
        ]
    )

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=AgentConfig(auto_context=False, max_iterations=8),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read"]
    assert any("alpha parity line" in message for message in tool_result_messages(run))
    assert "Recovered the raw JSON tool call" in run.response


@pytest.mark.asyncio
async def test_native_and_raw_tool_paths_share_executor_trace(temp_dir: Path) -> None:
    native_fixture = temp_dir / "native.txt"
    native_fixture.write_text("native parity line\n")
    native_backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(native_fixture)}),
                content="I'll inspect the native tool result.",
            ),
            final_response("Native read complete."),
        ]
    )
    native_run = await run_scenario(
        "Read native.txt.",
        native_backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    raw_fixture = temp_dir / "raw.txt"
    raw_fixture.write_text("raw parity line\n")
    raw_json = f'{{"name": "read", "arguments": {{"file_path": "{raw_fixture}"}}}}'
    raw_backend = ScriptedBackend(
        streams=[
            [
                StreamChunk(content=raw_json[:20], is_done=False),
                StreamChunk(content=raw_json[20:], full_content=raw_json, is_done=True),
            ],
            [
                StreamChunk(
                    content="Raw read complete.",
                    full_content="Raw read complete.",
                    is_done=True,
                )
            ],
        ]
    )
    raw_run = await run_scenario(
        "Read raw.txt.",
        raw_backend,
        config=AgentConfig(auto_context=False, max_iterations=8),
        project_root=temp_dir,
    )

    for run in (native_run, raw_run):
        names = trace_event_names(run)
        assert "assistant.tool_batch" in names
        assert "tool.received" in names
        assert "tool.executed" in names

    native_summary = native_run.agent.last_turn_summary
    raw_summary = raw_run.agent.last_turn_summary
    assert native_summary is not None
    assert raw_summary is not None
    assert any(
        event.name == "tool.received" and event.data["source"] == "native"
        for event in native_summary.trace
    )
    assert any(
        event.name == "tool.received" and event.data["source"] == "raw_text"
        for event in raw_summary.trace
    )


@pytest.mark.asyncio
async def test_backend_capability_probe_refreshes_native_tool_mode(
    temp_dir: Path,
) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("capability probe line\n")

    class LazyCapabilityBackend(ScriptedBackend):
        def __init__(self, completions: list[CompletionResponse]) -> None:
            super().__init__(completions=completions, supports_native_tools=False)
            self.model = "custom-qwen-build"
            self._described = False

        async def describe_model(self) -> dict[str, dict[str, list[str]]]:
            self._described = True
            return {"details": {"families": ["qwen2.5"]}}

        def capability_profile(self):
            model_details = (
                {"details": {"families": ["qwen2.5"]}} if self._described else None
            )
            return resolve_capability_profile(
                self.model,
                model_details=model_details,
            )

    backend = LazyCapabilityBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)}),
                content="I'll inspect that file after probing capabilities.",
            ),
            final_response("Capability probing enabled the native read."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file after checking model capabilities.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert backend._described
    assert not run.agent.use_react
    assert run.invocations[0].tools is not None
    assert tool_event_names(run) == ["read"]
    assert "Capability probing enabled the native read." in run.response


@pytest.mark.asyncio
async def test_run_streaming_delegates_to_primary_runtime(temp_dir: Path) -> None:
    fixture = temp_dir / "streaming.txt"
    fixture.write_text("streamed runtime line\n")
    backend = ScriptedBackend(
        streams=[
            [
                StreamChunk(
                    content="I'll inspect the file now.",
                    full_content="I'll inspect the file now.",
                    tool_calls=[
                        ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)})
                    ],
                    is_done=True,
                )
            ],
            [
                StreamChunk(
                    content="Finished reading the streamed fixture.",
                    full_content="Finished reading the streamed fixture.",
                    is_done=True,
                )
            ],
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(auto_context=False, max_iterations=8),
        project_root=temp_dir,
    )

    events = [event async for event in agent.run_streaming("Read the streamed fixture file.")]

    assert any(event.type == "tool_call" and event.tool_name == "read" for event in events)
    assert any(
        event.type == "tool_result" and "streamed runtime line" in event.content
        for event in events
    )
    assert agent.last_turn_summary is not None
    assert agent.last_turn_summary.final_response.startswith(
        "Finished reading the streamed fixture."
    )


@pytest.mark.asyncio
async def test_completion_check_continuation(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(temp_dir)
    target = temp_dir / "hello.py"
    backend = ScriptedBackend(
        completions=[
            final_response("Done."),
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "print('hello from loader')\n"},
                ),
                content="You're right, I'll create the file first.",
            ),
            native_tool_response(
                ToolCall(id="bash-1", name="bash", arguments={"command": f"python {target.name}"}),
                content="Now I'll run the script.",
            ),
            final_response("Successfully created and ran hello.py."),
        ]
    )
    config = non_streaming_config(completion_check=True)

    run = await run_scenario(
        "Create a hello.py file and run it.",
        backend,
        config=config,
        project_root=temp_dir,
    )

    assert target.exists()
    assert any(event.type == "completion_check" for event in run.events)
    assert tool_event_names(run) == ["write", "bash"]
    assert any("hello from loader" in message for message in tool_result_messages(run))
    assert "Successfully created and ran hello.py." in run.response


@pytest.mark.asyncio
async def test_tool_result_contract_regression() -> None:
    errors: list[str] = []
    duplicate_path = "/tmp/already-created.txt"

    duplicate_backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="dup-1",
                    name="write",
                    arguments={"file_path": duplicate_path, "content": "already there\n"},
                ),
                content="I'll create the file again.",
            ),
            final_response("Skipped the duplicate write."),
        ]
    )
    duplicate_agent = Agent(duplicate_backend, config=non_streaming_config())
    duplicate_agent.safeguards.record_action(
        "write",
        {"file_path": duplicate_path, "content": "already there\n"},
    )

    try:
        await duplicate_agent.run("Create /tmp/already-created.txt again.")
    except TypeError as exc:
        errors.append(f"duplicate branch raised {exc}")

    validation_backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="invalid-1", name="bash", arguments={"command": ""}),
                content="I'll run that command.",
            ),
            final_response("Blocked the invalid command."),
        ]
    )
    validation_agent = Agent(validation_backend, config=non_streaming_config())

    try:
        await validation_agent.run("Run an empty command.")
    except TypeError as exc:
        errors.append(f"validation branch raised {exc}")

    assert not errors, "\n".join(errors)
