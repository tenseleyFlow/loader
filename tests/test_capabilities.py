"""Tests for runtime capability profile resolution."""

from loader.runtime.capabilities import (
    CapabilityProfile,
    resolve_backend_capability_profile,
    resolve_capability_profile,
)


def test_explicit_override_wins() -> None:
    override = CapabilityProfile(
        model_name="override-model",
        supports_native_tools=True,
        supports_streaming=True,
        context_window=32000,
        preferred_tool_call_format="native",
        verification_strictness="strict",
        notes=["explicit override"],
    )

    resolved = resolve_capability_profile("phi3", override=override)
    assert resolved is override


def test_exact_match_registry_resolution() -> None:
    resolved = resolve_capability_profile("deepseek-r1")

    assert not resolved.supports_native_tools
    assert resolved.preferred_tool_call_format == "json_tag"
    assert resolved.verification_strictness == "strict"


def test_family_heuristic_resolution_uses_model_details() -> None:
    resolved = resolve_capability_profile(
        "custom-qwen-build",
        model_details={"details": {"families": ["qwen2.5"]}},
    )

    assert resolved.supports_native_tools
    assert resolved.preferred_tool_call_format == "native"
    assert "heuristic" in resolved.notes[0].lower()


def test_unknown_models_default_to_safe_react_profile() -> None:
    resolved = resolve_capability_profile("mystery-model")

    assert not resolved.supports_native_tools
    assert resolved.preferred_tool_call_format == "json_tag"
    assert "defaulting to safe" in resolved.notes[0].lower()


def test_qwen25_coder_inherits_strict_verification() -> None:
    resolved = resolve_capability_profile("qwen2.5-coder:32b")

    assert resolved.supports_native_tools
    assert resolved.preferred_tool_call_format == "native"
    assert resolved.verification_strictness == "strict"


def test_devstral_resolves_native_tools() -> None:
    resolved = resolve_capability_profile("devstral:24b")

    assert resolved.supports_native_tools
    assert resolved.preferred_tool_call_format == "native"


def test_gpt_oss_resolves_native_tools_via_registry() -> None:
    resolved = resolve_capability_profile("gpt-oss")

    assert resolved.supports_native_tools
    assert resolved.preferred_tool_call_format == "native"


def test_gpt_oss_custom_variant_resolves_native_via_family() -> None:
    resolved = resolve_capability_profile("gpt-oss-custom:20b")

    assert resolved.supports_native_tools
    assert resolved.preferred_tool_call_format == "native"
    assert "heuristic" in resolved.notes[0].lower()


def test_backend_capability_profile_prefers_explicit_backend_surface() -> None:
    class DummyBackend:
        def supports_native_tools(self) -> bool:
            return True

    resolved = resolve_backend_capability_profile(DummyBackend())

    assert resolved.supports_native_tools
    assert resolved.preferred_tool_call_format == "native"
    assert "backend capability surface" in resolved.notes[0].lower()
