"""Tests for definition-of-done state and persistence."""

from pathlib import Path

from loader.llm.base import ToolCall
from loader.runtime.dod import (
    DefinitionOfDoneStore,
    create_definition_of_done,
    derive_verification_commands,
    determine_task_size,
    record_successful_tool_call,
)


def test_determine_task_size_boundaries() -> None:
    assert determine_task_size(1, 10) == "small"
    assert determine_task_size(3, 99) == "small"
    assert determine_task_size(4, 99) == "standard"
    assert determine_task_size(15, 499) == "standard"
    assert determine_task_size(16, 499) == "large"
    assert determine_task_size(15, 500) == "large"


def test_definition_of_done_round_trip(tmp_path: Path) -> None:
    store = DefinitionOfDoneStore(tmp_path)
    dod = create_definition_of_done(
        "Create hello.py and verify it runs.",
        retry_budget=2,
    )
    dod.status = "fixing"
    dod.retry_count = 1
    dod.verification_commands = ["python hello.py"]
    dod.touched_files = [str(tmp_path / "hello.py")]
    saved_path = store.save(dod)

    reloaded = store.load(saved_path)

    assert reloaded.task_statement == dod.task_statement
    assert reloaded.status == "fixing"
    assert reloaded.retry_count == 1
    assert reloaded.verification_commands == ["python hello.py"]
    assert reloaded.touched_files == [str(tmp_path / "hello.py")]


def test_verification_command_derivation_prefers_runtime_evidence(tmp_path: Path) -> None:
    project_root = tmp_path
    dod = create_definition_of_done("Create hello.py and make sure it runs.")
    hello_path = project_root / "hello.py"
    record_successful_tool_call(
        dod,
        ToolCall(
            id="write-1",
            name="write",
            arguments={"file_path": str(hello_path), "content": "print('hi')\n"},
        ),
    )
    record_successful_tool_call(
        dod,
        ToolCall(
            id="bash-1",
            name="bash",
            arguments={"command": "python hello.py"},
        ),
    )

    commands = derive_verification_commands(
        dod,
        project_root=project_root,
        task_statement=dod.task_statement,
    )

    assert commands == ["python hello.py"]
