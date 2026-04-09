"""Tests for runtime-first CLI owner selection."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import loader.agent.loop as agent_loop_module
import loader.cli.main as cli_main_module
import loader.runtime.runtime_api as runtime_api_module
import loader.runtime.runtime_handle as runtime_handle_module
from loader.agent.loop import AgentConfig


class _FakeBackend:
    def __init__(self, **kwargs) -> None:
        self.model = kwargs.get("model", "fake-model")
        self.timeout = kwargs.get("timeout", 60)

    async def health_check(self) -> bool:
        return True

    async def describe_model(self) -> dict[str, object]:
        return {}

    def supports_native_tools(self) -> bool:
        return True


def _install_fake_ollama_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("loader.llm.ollama")
    module.OllamaBackend = _FakeBackend
    monkeypatch.setitem(sys.modules, "loader.llm.ollama", module)


def test_build_runtime_shell_owner_uses_runtime_handle_for_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, object]] = []

    class FakeHandle:
        def __init__(self, **kwargs) -> None:
            seen.append(kwargs)

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("Agent should not be used for internal CLI paths")

    monkeypatch.setattr(runtime_handle_module, "RuntimeHandle", FakeHandle)
    monkeypatch.setattr(agent_loop_module, "Agent", FakeAgent)

    owner = runtime_api_module.build_runtime_shell_owner(
        backend="backend",
        registry="registry",
        config="config",
        owner_kind="runtime",
    )

    assert isinstance(owner, FakeHandle)
    assert seen == [
        {
            "backend": "backend",
            "registry": "registry",
            "config": "config",
            "project_root": None,
        }
    ]


def test_build_runtime_shell_owner_uses_agent_for_public_compat_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, object]] = []

    class FakeHandle:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("RuntimeHandle should not be used for public TUI paths")

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            seen.append(kwargs)

    monkeypatch.setattr(runtime_handle_module, "RuntimeHandle", FakeHandle)
    monkeypatch.setattr(agent_loop_module, "Agent", FakeAgent)

    owner = runtime_api_module.build_runtime_shell_owner(
        backend="backend",
        registry="registry",
        config="config",
        owner_kind="public-compat",
    )

    assert isinstance(owner, FakeAgent)
    assert seen == [
        {
            "backend": "backend",
            "registry": "registry",
            "config": "config",
            "project_root": None,
        }
    ]


@pytest.mark.asyncio
async def test_main_uses_runtime_first_owner_for_tui_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_owner = SimpleNamespace(
        capability_profile=SimpleNamespace(
            preferred_tool_call_format="native",
            verification_strictness="strict",
        ),
        workflow_mode="execute",
        active_permission_mode="workspace-write",
        session=SimpleNamespace(session_id="session-123", active_turn_phase=""),
        project_context=None,
        resume_session=lambda session_id=None: False,
    )
    owner_calls: list[dict[str, object]] = []
    app_calls: list[dict[str, object]] = []

    class FakeApp:
        def __init__(self, **kwargs) -> None:
            app_calls.append(kwargs)

        async def run_async(self) -> None:
            app_calls.append({"ran": True})

    _install_fake_ollama_module(monkeypatch)
    monkeypatch.setattr("loader.config.get_default_model", lambda: "fake-model")
    monkeypatch.setattr("loader.config.get_last_model", lambda: None)
    monkeypatch.setattr("loader.config.set_last_model", lambda model: None)
    monkeypatch.setattr(
        "loader.tools.base.create_default_registry",
        lambda: SimpleNamespace(skip_confirmation=False),
    )
    fake_ui_app = ModuleType("loader.ui.app")
    fake_ui_app.LoaderApp = FakeApp
    monkeypatch.setitem(sys.modules, "loader.ui.app", fake_ui_app)

    def fake_build_owner(*, backend, registry, config, owner_kind):
        owner_calls.append(
            {
                "backend": backend,
                "registry": registry,
                "config": config,
                "owner_kind": owner_kind,
            }
        )
        return fake_owner

    monkeypatch.setattr(cli_main_module, "build_runtime_shell_owner", fake_build_owner)

    await cli_main_module._main(
        model="fake-model",
        select_model=False,
        backend="ollama",
        yes=False,
        permission_mode="workspace-write",
        react=False,
        no_context=True,
        plan=False,
        clarify=False,
        resume_target=None,
        no_recover=False,
        no_tui=False,
        ctx=8192,
        gpu=-1,
        timeout=60,
        decompose=False,
        critique=False,
        confidence=False,
        verify=False,
        reason=False,
        prompt=None,
    )

    assert owner_calls and owner_calls[0]["owner_kind"] == "runtime"
    assert app_calls[0]["shell_owner"] is fake_owner
    assert app_calls[-1] == {"ran": True}


@pytest.mark.asyncio
async def test_main_uses_runtime_first_owner_for_single_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, object]] = []
    fake_owner = SimpleNamespace(
        capability_profile=SimpleNamespace(
            preferred_tool_call_format="native",
            verification_strictness="strict",
        ),
        workflow_mode="execute",
        active_permission_mode="workspace-write",
        session=SimpleNamespace(session_id="session-123"),
        project_context=None,
        resume_session=lambda session_id=None: False,
    )

    _install_fake_ollama_module(monkeypatch)
    monkeypatch.setattr("loader.config.get_default_model", lambda: "fake-model")
    monkeypatch.setattr("loader.config.get_last_model", lambda: None)
    monkeypatch.setattr("loader.config.set_last_model", lambda model: None)
    monkeypatch.setattr(
        "loader.tools.base.create_default_registry",
        lambda: SimpleNamespace(skip_confirmation=False),
    )

    def fake_build_owner(*, backend, registry, config, owner_kind):
        seen.append(
            {
                "backend": backend,
                "registry": registry,
                "config": config,
                "owner_kind": owner_kind,
            }
        )
        return fake_owner

    captured = {}

    async def fake_run_once(owner, prompt: str, skip_confirmation: bool = False) -> None:
        captured["owner"] = owner
        captured["prompt"] = prompt
        captured["skip_confirmation"] = skip_confirmation

    monkeypatch.setattr(cli_main_module, "build_runtime_shell_owner", fake_build_owner)
    monkeypatch.setattr(cli_main_module, "run_once", fake_run_once)

    await cli_main_module._main(
        model="fake-model",
        select_model=False,
        backend="ollama",
        yes=False,
        permission_mode="workspace-write",
        react=False,
        no_context=True,
        plan=False,
        clarify=False,
        resume_target=None,
        no_recover=False,
        no_tui=False,
        ctx=8192,
        gpu=-1,
        timeout=60,
        decompose=False,
        critique=False,
        confidence=False,
        verify=False,
        reason=False,
        prompt="Summarize the runtime-first shell path.",
    )

    assert seen and seen[0]["owner_kind"] == "runtime"
    assert isinstance(seen[0]["config"], AgentConfig)
    assert captured == {
        "owner": fake_owner,
        "prompt": "Summarize the runtime-first shell path.",
        "skip_confirmation": False,
    }


@pytest.mark.asyncio
async def test_explore_main_uses_runtime_first_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, object]] = []

    class FakeExploreOwner:
        use_react = False

        async def run_explore(self, prompt: str, on_event=None, *, fresh: bool = False) -> str:
            seen.append({"prompt": prompt, "fresh": fresh, "on_event": on_event})
            return "Explore reply."

    _install_fake_ollama_module(monkeypatch)
    monkeypatch.setattr("loader.config.get_default_model", lambda: "fake-model")
    monkeypatch.setattr("loader.config.get_last_model", lambda: None)
    monkeypatch.setattr("loader.config.set_last_model", lambda model: None)

    owner_calls: list[dict[str, object]] = []

    def fake_build_owner(*, backend, registry, config, owner_kind):
        owner_calls.append(
            {
                "backend": backend,
                "registry": registry,
                "config": config,
                "owner_kind": owner_kind,
            }
        )
        return FakeExploreOwner()

    monkeypatch.setattr(cli_main_module, "build_runtime_shell_owner", fake_build_owner)

    await cli_main_module._explore_main(
        model="fake-model",
        select_model=False,
        backend="ollama",
        react=False,
        no_context=True,
        fresh=True,
        ctx=8192,
        gpu=-1,
        timeout=60,
        prompt="Where should I start?",
    )

    assert owner_calls and owner_calls[0]["owner_kind"] == "runtime"
    assert isinstance(owner_calls[0]["config"], AgentConfig)
    assert len(seen) == 1
    assert seen[0]["prompt"] == "Where should I start?"
    assert seen[0]["fresh"] is True
    assert callable(seen[0]["on_event"])
