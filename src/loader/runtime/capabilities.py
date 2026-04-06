"""Capability profile resolution for runtime behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ToolCallFormat = Literal["native", "json_tag", "bracket"]
VerificationStrictness = Literal["lax", "standard", "strict"]


@dataclass(frozen=True)
class CapabilityProfile:
    """Resolved model/runtime capability profile."""

    model_name: str
    supports_native_tools: bool
    supports_streaming: bool
    context_window: int
    preferred_tool_call_format: ToolCallFormat
    verification_strictness: VerificationStrictness
    notes: list[str] = field(default_factory=list)


def _profile(
    model_name: str,
    *,
    supports_native_tools: bool,
    supports_streaming: bool = True,
    context_window: int = 8192,
    preferred_tool_call_format: ToolCallFormat = "native",
    verification_strictness: VerificationStrictness = "standard",
    notes: list[str] | None = None,
) -> CapabilityProfile:
    return CapabilityProfile(
        model_name=model_name,
        supports_native_tools=supports_native_tools,
        supports_streaming=supports_streaming,
        context_window=context_window,
        preferred_tool_call_format=preferred_tool_call_format,
        verification_strictness=verification_strictness,
        notes=list(notes or []),
    )


KNOWN_CAPABILITY_PROFILES: dict[str, CapabilityProfile] = {
    "llama3.1": _profile(
        "llama3.1",
        supports_native_tools=True,
        preferred_tool_call_format="native",
        verification_strictness="standard",
    ),
    "llama3.2": _profile(
        "llama3.2",
        supports_native_tools=True,
        preferred_tool_call_format="native",
        verification_strictness="standard",
    ),
    "llama3.3": _profile(
        "llama3.3",
        supports_native_tools=True,
        preferred_tool_call_format="native",
        verification_strictness="standard",
    ),
    "qwen2.5": _profile(
        "qwen2.5",
        supports_native_tools=True,
        preferred_tool_call_format="native",
        verification_strictness="strict",
    ),
    "mistral": _profile(
        "mistral",
        supports_native_tools=True,
        preferred_tool_call_format="native",
        verification_strictness="standard",
    ),
    "mixtral": _profile(
        "mixtral",
        supports_native_tools=True,
        preferred_tool_call_format="native",
        verification_strictness="standard",
    ),
    "codestral": _profile(
        "codestral",
        supports_native_tools=False,
        preferred_tool_call_format="json_tag",
        verification_strictness="strict",
        notes=["Use ReAct-style prompting for tool calls."],
    ),
    "deepseek-coder": _profile(
        "deepseek-coder",
        supports_native_tools=False,
        preferred_tool_call_format="json_tag",
        verification_strictness="strict",
        notes=["Better with extracted JSON-tag tool calls than native tools."],
    ),
    "deepseek-r1": _profile(
        "deepseek-r1",
        supports_native_tools=False,
        preferred_tool_call_format="json_tag",
        verification_strictness="strict",
        notes=["Reasoning-oriented family; filter think blocks and use ReAct."],
    ),
    "phi3": _profile(
        "phi3",
        supports_native_tools=False,
        preferred_tool_call_format="bracket",
        verification_strictness="lax",
    ),
    "gemma2": _profile(
        "gemma2",
        supports_native_tools=False,
        preferred_tool_call_format="bracket",
        verification_strictness="lax",
    ),
}

NATIVE_TOOL_FAMILIES = {"llama3", "qwen2", "qwen2.5", "mistral", "mixtral", "command-r", "granite"}
NO_TOOL_FAMILIES = {
    "llama2",
    "phi",
    "phi3",
    "gemma",
    "gemma2",
    "tinyllama",
    "codestral",
    "deepseek-coder",
    "deepseek-r1",
    "starcoder",
    "codegemma",
}


def _family_tokens(model_name: str, model_details: dict[str, Any] | None) -> set[str]:
    """Collect lowercase family tokens from model name and model details."""

    tokens = {model_name.lower()}
    if model_details:
        details = model_details.get("details", model_details)
        families = details.get("families", [])
        family = details.get("family")
        if isinstance(families, list):
            tokens.update(str(token).lower() for token in families)
        if family:
            tokens.add(str(family).lower())
    return tokens


def resolve_capability_profile(
    model_name: str,
    *,
    override: CapabilityProfile | None = None,
    model_details: dict[str, Any] | None = None,
) -> CapabilityProfile:
    """Resolve the capability profile for a model.

    Resolution order:
    1. explicit override
    2. exact-name match in the built-in registry
    3. heuristic fallback using model details / family tokens
    """

    if override is not None:
        return override

    normalized = model_name.lower().strip()
    if normalized in KNOWN_CAPABILITY_PROFILES:
        known = KNOWN_CAPABILITY_PROFILES[normalized]
        return CapabilityProfile(
            model_name=model_name,
            supports_native_tools=known.supports_native_tools,
            supports_streaming=known.supports_streaming,
            context_window=known.context_window,
            preferred_tool_call_format=known.preferred_tool_call_format,
            verification_strictness=known.verification_strictness,
            notes=list(known.notes),
        )

    tokens = _family_tokens(normalized, model_details)

    if any(token in NATIVE_TOOL_FAMILIES for token in tokens):
        return _profile(
            model_name,
            supports_native_tools=True,
            preferred_tool_call_format="native",
            verification_strictness="standard",
            notes=["Resolved from model family heuristic."],
        )

    if any(token in NO_TOOL_FAMILIES for token in tokens):
        return _profile(
            model_name,
            supports_native_tools=False,
            preferred_tool_call_format="json_tag",
            verification_strictness="standard",
            notes=["Resolved from conservative no-native-tools heuristic."],
        )

    return _profile(
        model_name,
        supports_native_tools=False,
        preferred_tool_call_format="json_tag",
        verification_strictness="standard",
        notes=["Unknown model family; defaulting to safe ReAct-style tool use."],
    )
