"""Assistant-response repair and fallback helpers for the typed runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agent.parsing import parse_tool_calls
from ..llm.base import ToolCall
from .context import RuntimeContext


@dataclass(slots=True)
class EmptyResponseDecision:
    """Decision for an empty assistant response."""

    should_continue: bool
    retry_prompt: str | None = None
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

    def __init__(self, context: RuntimeContext) -> None:
        self.context = context

    def handle_empty_response(
        self,
        *,
        task: str,
        original_task: str | None,
        empty_retry_count: int,
        max_empty_retries: int,
    ) -> EmptyResponseDecision:
        """Return the next action when the assistant responds with empty content."""

        if empty_retry_count <= max_empty_retries:
            task_context = original_task or task
            retry_prompts = [
                "Great! Now let me proceed with the task. I'll start by using my tools.",
                "I understand. Let me create that now using my tools (write, bash, etc.).",
                (
                    f"Proceeding with: {task_context[:80]}. "
                    "I'll use the write tool to create the files."
                ),
                "Starting now. First step: create the necessary files and directories.",
                (
                    "Let me complete this task step by step. "
                    f"The goal is: {task_context[:100]}"
                ),
            ]
            retry_prompt = retry_prompts[
                min(empty_retry_count - 1, len(retry_prompts) - 1)
            ]
            return EmptyResponseDecision(
                should_continue=True,
                retry_prompt=retry_prompt,
            )

        return EmptyResponseDecision(
            should_continue=False,
            final_response=(
                "I need a bit more direction. "
                "What specifically would you like me to create or do?"
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

        if self.context.use_react:
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
            parsed_raw = parse_tool_calls(response_content)
            if parsed_raw.tool_calls:
                normalized_tool_calls = parsed_raw.tool_calls
                normalized_content = parsed_raw.content or normalized_content
                tool_source = "raw_text"
                clear_stream = True
            else:
                raw_tool_calls = self.context.legacy.extract_raw_json_tool_calls(
                    response_content
                )
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
