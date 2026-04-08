"""Assistant-response repair and fallback helpers for the typed runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agent.parsing import parse_tool_calls
from ..llm.base import ToolCall
from .parsing import parse_tool_calls as parse_runtime_tool_calls


@dataclass(slots=True)
class EmptyResponseDecision:
    """Decision for an empty assistant response."""

    should_continue: bool
    retry_message: str | None = None
    final_response: str | None = None
    failure: str | None = None


@dataclass(slots=True)
class ToolCallAnalysis:
    """Normalized assistant-output analysis for tool execution."""

    content: str
    response_content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_source: str = "native"
    clear_stream: bool = False
    is_final_answer: bool = False
    should_stop: bool = False
    final_response: str | None = None
    failure: str | None = None
    extracted_iterations: int = 0


class ResponseRepairer:
    """Owns response-repair heuristics that used to live inline in the loop."""

    def __init__(self, agent) -> None:
        self.agent = agent

    def handle_empty_response(
        self,
        *,
        task: str,
        original_task: str | None,
        empty_retry_count: int,
        max_empty_retries: int,
    ) -> EmptyResponseDecision:
        """Return the next action when the assistant responds with empty content."""

        _ = task, original_task, max_empty_retries
        if empty_retry_count == 1:
            return EmptyResponseDecision(
                should_continue=True,
                retry_message=(
                    "[EMPTY ASSISTANT RESPONSE]\n"
                    "Your last response was empty. Respond directly to the task "
                    "or call tools if needed. Do not return an empty response."
                ),
            )

        return EmptyResponseDecision(
            should_continue=False,
            final_response=(
                "I didn't get a usable response from the model after retrying once. "
                "Please try again or switch to a different backend/model."
            ),
            failure="assistant returned empty output repeatedly",
        )

    def analyze_response(
        self,
        *,
        content: str,
        response_content: str,
        tool_calls: list[ToolCall],
        extracted_iterations: int,
        max_extracted_iterations: int,
    ) -> ToolCallAnalysis:
        """Normalize assistant output into final-answer, tool, or repair outcomes."""

        normalized_content = content
        normalized_tool_calls = list(tool_calls)
        tool_source = "native"

        if self.agent.use_react:
            parsed = parse_tool_calls(content)
            normalized_tool_calls = parsed.tool_calls
            normalized_content = parsed.content

            if parsed.is_final_answer and not normalized_tool_calls:
                return ToolCallAnalysis(
                    content=normalized_content,
                    response_content=response_content,
                    is_final_answer=True,
                    final_response=normalized_content,
                )

        clear_stream = False
        next_extracted_iterations = extracted_iterations
        if not normalized_tool_calls:
            raw_tool_calls = self._extract_raw_tool_calls(response_content)
            if raw_tool_calls:
                normalized_tool_calls = raw_tool_calls
                tool_source = "raw_text"
                clear_stream = True

        if normalized_tool_calls and tool_source == "raw_text":
            next_extracted_iterations += 1
            if next_extracted_iterations > max_extracted_iterations:
                return ToolCallAnalysis(
                    content=normalized_content,
                    response_content=response_content,
                    tool_calls=normalized_tool_calls,
                    tool_source=tool_source,
                    clear_stream=clear_stream,
                    extracted_iterations=next_extracted_iterations,
                    should_stop=True,
                    final_response=(
                        normalized_content
                        + "\n\nLet me know if you'd like me to continue or make changes."
                    ),
                    failure="raw tool extraction exceeded iteration budget",
                )

        return ToolCallAnalysis(
            content=normalized_content,
            response_content=response_content,
            tool_calls=normalized_tool_calls,
            tool_source=tool_source,
            clear_stream=clear_stream,
            extracted_iterations=next_extracted_iterations,
        )

    def _extract_raw_tool_calls(self, response_content: str) -> list[ToolCall]:
        """Recover raw-text tool calls for either agent or RuntimeContext callers."""

        extractor = getattr(self.agent, "_extract_raw_json_tool_calls", None)
        if callable(extractor):
            return extractor(response_content)

        registry = getattr(self.agent, "registry", None)
        allowed_tool_names = None
        if registry is not None:
            allowed_tool_names = [tool.name for tool in registry.list_tools()]
        parsed = parse_runtime_tool_calls(
            response_content,
            allowed_tool_names=allowed_tool_names,
        )
        return parsed.tool_calls
