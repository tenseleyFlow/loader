"""Capability profile resolution for runtime behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ToolCallFormat = Literal["native", "json_tag", "bracket"]
VerificationStrictness = Literal["lax", "standard", "strict"]


class SupportsCapabilityProfile(Protocol):
    """Runtime interface for backends that can describe capabilities."""

    def capability_profile(self) -> CapabilityProfile: ...


class SupportsNativeTools(Protocol):
    """Runtime interface for backends that can explicitly report tool support."""

    def supports_native_tools(self) -> bool: ...


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
    "qwen2.5-coder": _profile(
        "qwen2.5-coder",
        supports_native_tools=True,
        preferred_tool_call_format="native",
        verification_strictness="strict",
    ),
    "devstral": _profile(
        "devstral",
        supports_native_tools=True,
        preferred_tool_call_format="native",
        verification_strictness="standard",
        notes=["Agentic coding model; well-suited to loader's tool loop."],
    ),
    "gpt-oss": _profile(
        "gpt-oss",
        supports_native_tools=True,
        preferred_tool_call_format="native",
        verification_strictness="standard",
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

NATIVE_TOOL_FAMILIES = {
    "llama3", "qwen2", "qwen2.5", "qwen3", "mistral", "mixtral",
    "command-r", "granite", "devstral", "gemma4", "gpt-oss",
}
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

    name = model_name.lower()
    tokens = {name}
    # Strip :tag so "devstral:24b" also produces "devstral"
    if ":" in name:
        tokens.add(name.split(":")[0])
    if model_details:
        details = model_details.get("details", model_details)
        families = details.get("families", [])
        family = details.get("family")
        if isinstance(families, list):
            tokens.update(str(token).lower() for token in families)
        if family:
            tokens.add(str(family).lower())
    return tokens


def _any_prefix_match(tokens: set[str], family_set: set[str]) -> bool:
    """Check if any family entry is a prefix of any token."""
    for token in tokens:
        for family in family_set:
            if token.startswith(family):
                return True
    return False


def _coerce_positive_int(value: Any) -> int | None:
    """Return one positive integer when the input looks numeric."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _infer_context_window(model_details: dict[str, Any] | None) -> int | None:
    """Infer one model context window from Ollama model metadata."""

    if not isinstance(model_details, dict):
        return None

    candidates: list[int] = []

    details = model_details.get("details")
    if isinstance(details, dict):
        context_length = _coerce_positive_int(details.get("context_length"))
        if context_length is not None:
            candidates.append(context_length)

    model_info = model_details.get("model_info")
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if str(key).endswith(".context_length"):
                context_length = _coerce_positive_int(value)
                if context_length is not None:
                    candidates.append(context_length)

    return max(candidates) if candidates else None


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

    inferred_context_window = _infer_context_window(model_details)

    if override is not None:
        if inferred_context_window is None:
            return override
        return CapabilityProfile(
            model_name=override.model_name,
            supports_native_tools=override.supports_native_tools,
            supports_streaming=override.supports_streaming,
            context_window=inferred_context_window,
            preferred_tool_call_format=override.preferred_tool_call_format,
            verification_strictness=override.verification_strictness,
            notes=list(override.notes),
        )

    normalized = model_name.lower().strip()
    # Try full name first, then without :tag (e.g. "deepseek-r1:14b" -> "deepseek-r1")
    for key in (normalized, normalized.split(":")[0]):
        if key in KNOWN_CAPABILITY_PROFILES:
            known = KNOWN_CAPABILITY_PROFILES[key]
            return CapabilityProfile(
                model_name=model_name,
                supports_native_tools=known.supports_native_tools,
                supports_streaming=known.supports_streaming,
                context_window=inferred_context_window or known.context_window,
                preferred_tool_call_format=known.preferred_tool_call_format,
                verification_strictness=known.verification_strictness,
                notes=list(known.notes),
            )

    tokens = _family_tokens(normalized, model_details)

    if _any_prefix_match(tokens, NATIVE_TOOL_FAMILIES):
        return _profile(
            model_name,
            supports_native_tools=True,
            context_window=inferred_context_window or 8192,
            preferred_tool_call_format="native",
            verification_strictness="standard",
            notes=["Resolved from model family heuristic."],
        )

    if _any_prefix_match(tokens, NO_TOOL_FAMILIES):
        return _profile(
            model_name,
            supports_native_tools=False,
            context_window=inferred_context_window or 8192,
            preferred_tool_call_format="json_tag",
            verification_strictness="standard",
            notes=["Resolved from conservative no-native-tools heuristic."],
        )

    return _profile(
        model_name,
        supports_native_tools=False,
        context_window=inferred_context_window or 8192,
        preferred_tool_call_format="json_tag",
        verification_strictness="standard",
        notes=["Unknown model family; defaulting to safe ReAct-style tool use."],
    )


def resolve_backend_capability_profile(backend: Any) -> CapabilityProfile:
    """Resolve capabilities from the backend first, then fall back to model heuristics."""

    explicit_profile = getattr(backend, "capability_profile", None)
    if callable(explicit_profile):
        profile = explicit_profile()
        if isinstance(profile, CapabilityProfile):
            return profile

    model_name = getattr(backend, "model", backend.__class__.__name__)
    explicit_native_tools = getattr(backend, "supports_native_tools", None)
    if callable(explicit_native_tools):
        supports_native_tools = bool(explicit_native_tools())
        preferred_tool_call_format: ToolCallFormat = (
            "native" if supports_native_tools else "json_tag"
        )
        return _profile(
            model_name,
            supports_native_tools=supports_native_tools,
            preferred_tool_call_format=preferred_tool_call_format,
            verification_strictness="standard",
            notes=["Resolved from backend capability surface."],
        )

    return resolve_capability_profile(model_name)
