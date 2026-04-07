"""Read-only explore lane for fast lookup-oriented questions."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from ..llm.base import Message, Role
from ..runtime.events import AgentEvent, TurnSummary
from ..tools.base import create_explore_registry
from .context import RuntimeContext
from .executor import ToolExecutionState, ToolExecutor
from .hooks import build_default_tool_hooks
from .parsing import parse_tool_calls
from .permissions import PermissionMode, PermissionRuleSet, build_permission_policy
from .prompting import format_tool_descriptions, get_project_specific_tips
from .tracing import RuntimeTracer

EventSink = Callable[[AgentEvent], Awaitable[None]]

EXPLORE_SYSTEM_PROMPT = """You are Loader Explore, a read-only codebase inspection agent.

Current directory: {cwd}

## Mission
- Answer repository lookup questions quickly.
- Use only read-only tools.
- Prefer concrete evidence from files, search results, or git state.
- Do not modify files, run verification loops, or create workflow artifacts.
- Do not ask planning questions; answer the lookup directly when possible.

## Tools Available
{tool_descriptions}

## Tool Use
Output tool calls in this format:
[tool: param="value", param2="value2"]

## Rules
1. Stay in read-only inspection mode.
2. Use tools before making factual claims about the repository.
3. Keep answers concise and evidence-grounded.
4. If the user asks for changes, tell them to switch back to the main Loader runtime.
"""

EXPLORE_REACT_SYSTEM_PROMPT = """You are Loader Explore, a read-only codebase inspection agent.

Current directory: {cwd}

## Mission
- Answer repository lookup questions quickly.
- Use only read-only tools.
- Prefer concrete evidence from files, search results, or git state.
- Do not modify files, run verification loops, or create workflow artifacts.
- Do not ask planning questions; answer the lookup directly when possible.

## Tools Available
{tool_descriptions}

## Tool Use
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

## Rules
1. Stay in read-only inspection mode.
2. Use tools before making factual claims about the repository.
3. Keep answers concise and evidence-grounded.
4. If the user asks for changes, tell them to switch back to the main Loader runtime.
"""


class ExploreRuntime:
    """Minimal read-only runtime for lookup-oriented tasks."""

    def __init__(self, agent) -> None:
        self.context: RuntimeContext = agent._build_runtime_context()
        self.registry = create_explore_registry(self.context.project_root)
        explore_rules = PermissionRuleSet(
            deny=list(self.context.permission_policy.rules.deny),
            ask=list(self.context.permission_policy.rules.ask),
            source_path=self.context.permission_policy.rules.source_path,
        )
        self.permission_policy = build_permission_policy(
            active_mode=PermissionMode.READ_ONLY,
            workspace_root=self.context.project_root,
            tool_requirements=self.registry.get_tool_requirements(),
            rules=explore_rules,
        )
        self.context.registry = self.registry
        self.context.permission_policy = self.permission_policy
        self.context.workflow_mode = "explore"
        self.tracer = RuntimeTracer()
        self.executor = ToolExecutor(
            self.registry,
            self.tracer,
            self.permission_policy,
            hooks=build_default_tool_hooks(
                action_tracker=self.context.safeguards.action_tracker,
                validator=self.context.safeguards.validator,
                registry=self.registry,
                rollback_plan=None,
            ),
        )

    async def run_query(
        self,
        prompt: str,
        emit: EventSink,
    ) -> TurnSummary:
        await self._prepare_runtime_capabilities()

        summary = TurnSummary(final_response="", workflow_mode="explore")
        messages = [
            Message(role=Role.SYSTEM, content=self._build_system_prompt()),
            Message(role=Role.USER, content=prompt),
        ]
        use_react = self.context.use_react
        tools = None if use_react else self.registry.get_schemas()

        for iteration in range(1, min(self.context.config.max_iterations, 6) + 1):
            summary.iterations = iteration
            self.tracer.record("explore.iteration_started", iteration=iteration)
            await emit(AgentEvent(type="thinking"))

            response = await self.context.backend.complete(
                messages=messages,
                tools=tools,
                temperature=min(self.context.config.temperature, 0.2),
                max_tokens=min(self.context.config.max_tokens, 1024),
            )

            parsed = parse_tool_calls(
                response.content,
                allowed_tool_names=[tool.name for tool in self.context.registry.list_tools()],
            )
            tool_calls = list(response.tool_calls or parsed.tool_calls)
            cleaned_content = parsed.content if parsed.tool_calls else response.content

            if tool_calls:
                assistant_message = Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=tool_calls,
                )
                messages.append(assistant_message)
                summary.assistant_messages.append(assistant_message)
                self.tracer.record(
                    "explore.tool_batch",
                    iteration=iteration,
                    tool_count=len(tool_calls),
                )
                for tool_call in tool_calls:
                    await emit(
                        AgentEvent(
                            type="tool_call",
                            tool_name=tool_call.name,
                            tool_args=tool_call.arguments,
                            phase="explore",
                        )
                    )
                    outcome = await self.executor.execute_tool_call(
                        tool_call,
                        source="explore",
                        skip_duplicate_check=True,
                        record_action=False,
                        skip_confirmation=True,
                    )
                    await emit(
                        AgentEvent(
                            type="tool_result",
                            content=outcome.event_content,
                            tool_name=tool_call.name,
                            is_error=outcome.is_error,
                            phase="explore",
                        )
                    )
                    messages.append(outcome.message)
                    summary.tool_result_messages.append(outcome.message)

                    if outcome.state == ToolExecutionState.EXECUTED and outcome.is_error:
                        summary.failures.append(outcome.result_output)
                continue

            final_response = cleaned_content.strip() or response.content.strip()
            assistant_message = Message(role=Role.ASSISTANT, content=response.content)
            messages.append(assistant_message)
            summary.assistant_messages.append(assistant_message)
            summary.final_response = final_response
            self.tracer.record("explore.completed", iteration=iteration)
            await emit(AgentEvent(type="response", content=final_response))
            break

        if not summary.final_response:
            summary.final_response = (
                "I couldn't complete the lookup cleanly in explore mode. "
                "Try narrowing the question or use the main runtime for a deeper task."
            )
            await emit(AgentEvent(type="response", content=summary.final_response))
            summary.failures.append("explore iteration budget exhausted")

        summary.trace = list(self.tracer.events)
        return summary

    async def _prepare_runtime_capabilities(self) -> None:
        describe_model = getattr(self.context.backend, "describe_model", None)
        if callable(describe_model):
            await describe_model()
        self.context.legacy.refresh_capability_profile()

    def _build_system_prompt(self) -> str:
        tool_descriptions = format_tool_descriptions(self.registry.get_schemas())
        project_tips = ""
        if self.context.project_context is not None:
            project_tips = "\n\n## Project Tips\n" + get_project_specific_tips(
                self.context.project_context
            )
        template = (
            EXPLORE_REACT_SYSTEM_PROMPT
            if self.context.use_react
            else EXPLORE_SYSTEM_PROMPT
        )
        return template.format(
            cwd=os.getcwd(),
            tool_descriptions=tool_descriptions,
        ) + project_tips
