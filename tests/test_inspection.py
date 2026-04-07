"""Tests for doctor, status, and session inspection surfaces."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import loader.cli.main as cli_main_module
from loader.llm.base import Message, Role
from loader.runtime.dod import DefinitionOfDoneStore, create_definition_of_done
from loader.runtime.inspection import (
    CheckStatus,
    collect_doctor_report,
    collect_status_snapshot,
    list_session_summaries,
    load_session_detail,
)
from loader.runtime.session import SessionSnapshot, SessionStore


class FakeOllamaBackend:
    """Small async backend stub for doctor tests."""

    def __init__(
        self,
        *,
        model: str,
        health: bool,
        models: list[dict[str, object]],
        model_details: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self._health = health
        self._models = models
        self._model_details = model_details

    async def list_models(self) -> list[dict[str, object]]:
        return list(self._models)

    async def health_check(self) -> bool:
        return self._health

    async def describe_model(self) -> dict[str, object] | None:
        return self._model_details

    async def close(self) -> None:
        return None


def _write_python_workspace(temp_dir: Path) -> None:
    (temp_dir / "pyproject.toml").write_text(
        "\n".join(
            [
                "[build-system]",
                'requires = ["hatchling"]',
                'build-backend = "hatchling.build"',
                "",
                "[tool.pytest.ini_options]",
                'testpaths = ["tests"]',
                "",
            ]
        )
        + "\n"
    )
    (temp_dir / "src").mkdir()
    (temp_dir / "tests").mkdir()


def _ensure_loader_dirs(temp_dir: Path) -> None:
    loader_root = temp_dir / ".loader"
    for name in ("sessions", "state", "dod", "briefs", "plans"):
        (loader_root / name).mkdir(parents=True, exist_ok=True)
    (loader_root / "project-memory.json").write_text("{}\n")


def _persist_session_with_dod(temp_dir: Path) -> tuple[str, str]:
    dod = create_definition_of_done("Fix the failing tests")
    dod.status = "fixing"
    dod.pending_items = ["Re-run pytest"]
    dod.completed_items = ["Patch the broken parser"]
    dod.last_verification_result = "failed"
    dod.evidence = []
    dod_path = DefinitionOfDoneStore(temp_dir).save(dod)

    snapshot = SessionSnapshot(
        session_id="20260406T120000Z-abcdef01",
        created_at="2026-04-06T12:00:00Z",
        updated_at="2026-04-06T12:05:00Z",
        messages=[
            Message(role=Role.USER, content="Fix the failing tests"),
            Message(role=Role.ASSISTANT, content="I updated the parser."),
        ],
        usage={"turns": 1, "tool_calls": 2},
        active_dod_path=str(dod_path),
        current_task="Fix the failing tests",
        workflow_mode="execute",
        permission_mode="workspace-write",
    )
    SessionStore(temp_dir).save(snapshot)
    return snapshot.session_id, str(dod_path)


@pytest.mark.asyncio
async def test_collect_doctor_report_passes_for_healthy_workspace(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)

    report = await collect_doctor_report(
        temp_dir,
        model="qwen2.5-coder:14b",
        backend_factory=lambda model: FakeOllamaBackend(
            model=model,
            health=True,
            models=[{"name": "qwen2.5-coder:14b"}],
            model_details={"details": {"family": "qwen2.5"}},
        ),
    )

    assert report.overall_status == CheckStatus.PASS
    assert {check.name for check in report.checks} == {
        "backend",
        "capabilities",
        "workspace",
        "write_access",
        "commands",
        "state",
        "permissions",
    }
    backend_check = next(check for check in report.checks if check.name == "backend")
    state_check = next(check for check in report.checks if check.name == "state")

    assert backend_check.status == CheckStatus.PASS
    assert state_check.status == CheckStatus.PASS


@pytest.mark.asyncio
async def test_collect_doctor_report_surfaces_backend_and_state_failures(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    (temp_dir / ".loader").mkdir()
    (temp_dir / ".loader" / "project-memory.json").write_text("{broken json")

    report = await collect_doctor_report(
        temp_dir,
        model="missing-model:latest",
        backend_factory=lambda model: FakeOllamaBackend(
            model=model,
            health=False,
            models=[{"name": "llama3.1:8b"}],
            model_details=None,
        ),
    )

    backend_check = next(check for check in report.checks if check.name == "backend")
    state_check = next(check for check in report.checks if check.name == "state")

    assert report.overall_status == CheckStatus.FAIL
    assert backend_check.status == CheckStatus.FAIL
    assert "not pulled" in backend_check.message
    assert state_check.status == CheckStatus.FAIL
    assert "corrupted" in state_check.message


@pytest.mark.asyncio
async def test_collect_doctor_report_fails_closed_on_invalid_permission_rules(
    temp_dir: Path,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    (temp_dir / ".loader" / "permission-rules.json").write_text('{"allow": "nope"}\n')

    report = await collect_doctor_report(
        temp_dir,
        model="qwen2.5-coder:14b",
        permission_mode="prompt",
        backend_factory=lambda model: FakeOllamaBackend(
            model=model,
            health=True,
            models=[{"name": "qwen2.5-coder:14b"}],
        ),
    )

    permission_check = next(check for check in report.checks if check.name == "permissions")
    assert report.overall_status == CheckStatus.FAIL
    assert permission_check.status == CheckStatus.FAIL
    assert report.permission_rules_valid is False
    assert "invalid" in permission_check.message.lower()


def test_status_and_session_surfaces_reflect_persisted_state(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id, dod_path = _persist_session_with_dod(temp_dir)

    snapshot = collect_status_snapshot(
        temp_dir,
        model="llama3.1:8b",
    )
    sessions = list_session_summaries(temp_dir)
    detail = load_session_detail(session_id, project_root=temp_dir)

    assert snapshot.active_session_id == session_id
    assert snapshot.dod_status == "fixing"
    assert snapshot.dod_pending_items_count == 1
    assert snapshot.last_verification_result == "failed"
    assert snapshot.active_dod_path == dod_path
    assert snapshot.permission_rule_counts == {"allow": 0, "deny": 0, "ask": 0}
    assert snapshot.permission_prompting_enabled is False
    assert snapshot.permission_rules_valid is True

    assert len(sessions) == 1
    assert sessions[0].session_id == session_id
    assert sessions[0].is_current is True
    assert sessions[0].dod_status == "fixing"

    assert detail.snapshot.session_id == session_id
    assert detail.is_current is True
    assert detail.definition_of_done is not None
    assert detail.definition_of_done.status == "fixing"


def test_status_and_session_commands_render_persisted_state(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id, _ = _persist_session_with_dod(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    status_result = runner.invoke(cli_main_module.status_cli, ["--model", "llama3.1:8b"])
    list_result = runner.invoke(cli_main_module.session_cli, ["list"])
    show_result = runner.invoke(cli_main_module.session_cli, ["show", session_id])

    assert status_result.exit_code == 0
    assert session_id in status_result.output
    assert "fixing" in status_result.output

    assert list_result.exit_code == 0
    assert "Session" in list_result.output

    assert show_result.exit_code == 0
    assert session_id in show_result.output
    assert "Patch the broken parser" in show_result.output


def test_status_snapshot_reports_invalid_permission_rules(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    (temp_dir / ".loader" / "permission-rules.json").write_text("{broken json")

    snapshot = collect_status_snapshot(temp_dir, permission_mode="prompt")

    assert snapshot.permission_rules_valid is False
    assert snapshot.permission_prompting_enabled is True


def test_root_help_lists_special_commands() -> None:
    help_text = cli_main_module._loader_help_text()

    assert "loader doctor" in help_text
    assert "loader status" in help_text
    assert "loader explore <prompt>" in help_text
    assert "loader session resume <id>" in help_text


def test_main_dispatches_session_resume_to_primary_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_cli_main(*, args: list[str], prog_name: str) -> None:
        captured["args"] = args
        captured["prog_name"] = prog_name

    monkeypatch.setattr(cli_main_module.cli, "main", fake_cli_main)
    monkeypatch.setattr(sys, "argv", ["loader", "session", "resume", "abc123", "--no-tui"])

    cli_main_module.main()

    assert captured == {
        "args": ["--resume-target", "abc123", "--no-tui"],
        "prog_name": "loader",
    }
