"""The main agent loop."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from ..context.project import ProjectContext, detect_project
from ..llm.base import LLMBackend, Message, Role
from ..runtime.capabilities import resolve_backend_capability_profile
from ..runtime.conversation import ConversationRuntime
from ..runtime.deliberation import (
    DECOMPOSITION_PROMPT,
    parse_decomposition,
    should_decompose,
)
from ..runtime.dod import DefinitionOfDoneStore
from ..runtime.events import AgentEvent, TurnSummary
from ..runtime.explore import ExploreRuntime
from ..runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from ..runtime.prompt_history import PromptSnapshot
from ..runtime.reasoning_types import TaskDecomposition
from ..runtime.safeguards import RuntimeSafeguards
from ..runtime.session import ConversationSession
from ..runtime.task_classification import is_conversational
from ..runtime.workflow import WorkflowMode
from ..tools.base import ToolRegistry, create_default_registry
from .prompts import build_system_prompt_result


@dataclass
class ReasoningConfig:
    """Configuration for reasoning stages."""
    # Decomposition: break complex tasks into atomic subtasks
    decomposition: bool = False
    decomposition_threshold: int = 30  # Word count threshold for auto-decomposition

    # Self-critique: review output before finalizing
    self_critique: bool = False
    max_critique_revisions: int = 2

    # Confidence scoring: rate certainty before actions
    confidence_scoring: bool = False
    min_confidence_for_action: int = 2  # Minimum ConfidenceLevel value to proceed
    use_quick_confidence: bool = True  # Use heuristics before LLM

    # Post-action verification: check results after execution
    verification: bool = False
    use_quick_verification: bool = True  # Use heuristics before LLM

    # Task completion: prevent premature stopping
    completion_check: bool = True  # ON by default - prevents "giving up"
    use_quick_completion: bool = True  # Use heuristics before LLM
    max_continuation_prompts: int = 5  # Max times to nudge agent to continue

    # Rollback planning: track how to undo destructive actions
    rollback: bool = True  # ON by default - track undo capability
    show_rollback_plan: bool = False  # Show rollback plan in output (verbose)


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_iterations: int = 15  # Reduced from 20
    temperature: float = 0.3  # Low for better instruction following
    max_tokens: int = 2048  # Reduced from 4096, most responses are shorter
    force_react: bool = False  # Force ReAct even if model supports native tools
    auto_context: bool = True  # Auto-detect project context on startup
    auto_plan: bool = False  # Auto-plan complex tasks (disabled by default - confuses smaller models)
    auto_recover: bool = True  # Auto-recover from tool errors
    max_recovery_attempts: int = 2  # Reduced from 3
    verification_retry_budget: int = 3  # Retry budget for verify/fix loop
    clarify_max_rounds: int = 2  # Bounded clarify depth before carrying ambiguity forward
    permission_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE
    workflow_mode_override: str | None = None
    stream: bool = True  # Stream LLM responses for real-time output
    session_rotate_after_bytes: int = 256 * 1024
    session_auto_compaction_input_tokens_threshold: int = 100_000
    session_compaction_keep_last_messages: int = 4

    # Reasoning stages configuration
    reasoning: ReasoningConfig = None  # type: ignore

    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning = ReasoningConfig()


class Agent:
    """The main agent that orchestrates the LLM and tools."""

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        project_root: Path | str | None = None,
    ):
        self.backend = backend
        self.config = config or AgentConfig()
        self.project_root = Path(project_root or ".").expanduser().resolve()
        self.registry = registry or create_default_registry(self.project_root)
        self.registry.configure_workspace_root(self.project_root)
        self.permission_config_status = load_permission_rules(self.project_root)
        if not self.permission_config_status.valid:
            raise ValueError(
                "Invalid permission policy configuration at "
                f"{self.permission_config_status.source_path}: "
                f"{self.permission_config_status.error}"
            )
        self.permission_policy = build_permission_policy(
            active_mode=self.config.permission_mode,
            workspace_root=self.project_root,
            tool_requirements=self.registry.get_tool_requirements(),
            rules=self.permission_config_status.rules,
        )
        self.workflow_mode = WorkflowMode.EXECUTE.value
        self.messages: list[Message] = []
        self.prompt_format: str | None = None
        self.prompt_sections: list[str] = []
        self.session = self._create_session(messages=self.messages)
        self._system_message: Message | None = None
        self._use_react: bool | None = None
        self.capability_profile = resolve_backend_capability_profile(self.backend)
        self.last_turn_summary: TurnSummary | None = None

        # Steering: allow user to send messages during execution
        self._steering_queue: asyncio.Queue[str] = asyncio.Queue()
        self._is_running: bool = False

        # Track original task for multi-turn conversations
        self._current_task: str | None = None

        # Runtime safeguards for filtering, steering, and deduplication
        self.safeguards = RuntimeSafeguards()

        # Load project context if enabled
        self.project_context: ProjectContext | None = None
        if self.config.auto_context:
            self.project_context = detect_project(self.project_root)

    def _create_session(
        self,
        *,
        messages: list[Message] | None = None,
    ) -> ConversationSession:
        """Create a fresh persisted conversation session."""

        session = ConversationSession(
            system_message_factory=self._get_system_message,
            few_shot_factory=self._get_few_shot_examples,
            project_root=self.project_root,
            messages=messages or [],
            permission_mode=self.permission_policy.active_mode.as_str(),
            permission_prompting_enabled=self.permission_policy.prompting_enabled,
            permission_rule_counts=self.permission_policy.rule_counts(),
            permission_rules_source=str(self.permission_config_status.source_path),
            prompt_format=self.prompt_format,
            prompt_sections=list(self.prompt_sections),
            workflow_mode=self.workflow_mode,
            rotate_after_bytes=self.config.session_rotate_after_bytes,
            auto_compaction_input_tokens_threshold=(
                self.config.session_auto_compaction_input_tokens_threshold
            ),
            compaction_keep_last_messages=(
                self.config.session_compaction_keep_last_messages
            ),
        )
        return session

    def _replace_session(self, session: ConversationSession) -> None:
        """Install a loaded session as the agent's active conversation."""

        self.session = session
        self.messages = session.messages
        self._current_task = session.current_task
        self.set_workflow_mode(session.workflow_mode)
        self.permission_policy.active_mode = PermissionMode.from_str(
            session.permission_mode
        )
        self.prompt_format = session.prompt_format
        self.prompt_sections = list(session.prompt_sections)
        self.last_turn_summary = None
        if session.active_dod_path:
            dod_path = Path(session.active_dod_path)
            if dod_path.exists():
                dod = DefinitionOfDoneStore(self.project_root).load(dod_path)
                self.last_turn_summary = TurnSummary(
                    final_response="",
                    definition_of_done=dod,
                    workflow_mode=session.workflow_mode,
                    workflow_reason_code=session.workflow_reason_code,
                    workflow_reason_summary=session.workflow_reason_summary,
                    workflow_decision_kind=session.workflow_decision_kind,
                    workflow_timeline=list(session.workflow_timeline),
                    session_id=session.session_id,
                    cumulative_usage=dict(session.usage_totals),
                )

    def resume_session(self, session_id: str | None = None) -> bool:
        """Resume the latest or named persisted session."""

        loaded = ConversationSession.load(
            project_root=self.project_root,
            system_message_factory=self._get_system_message,
            few_shot_factory=self._get_few_shot_examples,
            session_id=session_id,
            rotate_after_bytes=self.config.session_rotate_after_bytes,
            auto_compaction_input_tokens_threshold=(
                self.config.session_auto_compaction_input_tokens_threshold
            ),
            compaction_keep_last_messages=(
                self.config.session_compaction_keep_last_messages
            ),
        )
        if loaded is None:
            return False
        self._replace_session(loaded)
        return True

    def steer(self, message: str) -> bool:
        """Send a steering message to the agent during execution.

        Returns True if the agent is running and the message was queued,
        False if the agent is not running.
        """
        if not self._is_running:
            return False
        self._steering_queue.put_nowait(message)
        return True

    @property
    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self._is_running

    @property
    def active_permission_mode(self) -> str:
        """Return the current runtime permission mode."""
        return self.permission_policy.active_mode.as_str()

    @property
    def active_permission_rule_counts(self) -> dict[str, int]:
        """Return rule counts for the active permission policy."""

        return self.permission_policy.rule_counts()

    def _drain_steering_queue(self) -> list[str]:
        """Get all pending steering messages without blocking."""
        messages = []
        while True:
            try:
                msg = self._steering_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    @property
    def use_react(self) -> bool:
        """Determine whether to use ReAct prompting or native tools."""
        if self._use_react is not None:
            return self._use_react

        if self.config.force_react:
            self._use_react = True
            return True

        self._use_react = not self.capability_profile.supports_native_tools

        return self._use_react

    def _get_system_message(self) -> Message:
        """Get the system message with current context."""
        if self._system_message is None:
            tool_schemas = self.registry.get_schemas()
            prompt_result = build_system_prompt_result(
                tools=tool_schemas,
                use_react=self.use_react,
                project_context=self.project_context,
                workflow_mode=self.workflow_mode,
                permission_mode=self.active_permission_mode,
                cwd=self.project_root,
                current_task=self._current_task,
            )
            self.prompt_format = prompt_result.prompt_format
            self.prompt_sections = list(prompt_result.dynamic_section_names)
            self.session.update_runtime_state(
                prompt_format=prompt_result.prompt_format,
                prompt_sections=prompt_result.dynamic_section_names,
            )
            self.session.append_prompt_snapshot(
                PromptSnapshot.create(
                    workflow_mode=self.workflow_mode,
                    permission_mode=self.active_permission_mode,
                    current_task=self._current_task,
                    prompt_format=prompt_result.prompt_format,
                    prompt_sections=prompt_result.dynamic_section_names,
                    content=prompt_result.content,
                )
            )
            self._system_message = Message(
                role=Role.SYSTEM,
                content=prompt_result.content,
            )
        return self._system_message

    def set_workflow_mode(self, workflow_mode: str) -> None:
        """Update the active workflow mode used by the system prompt."""

        if workflow_mode == self.workflow_mode:
            return
        self.workflow_mode = workflow_mode
        self._system_message = None

    def _build_messages(self) -> list[Message]:
        """Build the full message list for the LLM."""
        return self.session.build_request_messages()

    def refresh_capability_profile(self) -> None:
        """Refresh the runtime capability profile from the current backend."""
        previous_profile = self.capability_profile
        refreshed_profile = resolve_backend_capability_profile(self.backend)
        self.capability_profile = refreshed_profile
        if refreshed_profile != previous_profile:
            self._system_message = None
        self._use_react = None

    def queue_steering_message(self, message: str) -> None:
        """Queue one runtime steering message."""

        self._steering_queue.put_nowait(message)

    def drain_steering_messages(self) -> list[str]:
        """Drain queued runtime steering messages."""

        return self._drain_steering_queue()

    def _get_few_shot_examples(self) -> list[Message]:
        """Get few-shot examples demonstrating proper tool use."""
        if self.use_react:
            # ReAct format examples
            return [
                Message(role=Role.USER, content="Create a file called hello.py that prints hello"),
                Message(role=Role.ASSISTANT, content='<tool_call>\n{"name": "write", "arguments": {"file_path": "hello.py", "content": "print(\'hello\')"}}\n</tool_call>'),
                Message(role=Role.TOOL, content="Created hello.py"),
                Message(role=Role.ASSISTANT, content="Done."),
            ]
        else:
            # Bracket format examples
            return [
                Message(role=Role.USER, content="Create a file called hello.py that prints hello"),
                Message(role=Role.ASSISTANT, content='[write: file_path="hello.py", content="print(\'hello\')"]'),
                Message(role=Role.TOOL, content="Created hello.py"),
                Message(role=Role.ASSISTANT, content="Done."),
            ]

    # === Reasoning Stage Methods ===

    async def _decompose_task(self, task: str) -> TaskDecomposition:
        """Decompose a complex task into atomic subtasks."""
        prompt = DECOMPOSITION_PROMPT.format(task=task)
        response = await self.backend.complete(
            messages=[
                self._get_system_message(),
                Message(role=Role.USER, content=prompt),
            ],
            tools=None,
            temperature=0.3,  # Lower temp for structured output
            max_tokens=1000,
        )
        return parse_decomposition(response.content, task)

    async def _handle_conversational(
        self,
        user_message: str,
        emit: Callable[[AgentEvent], Awaitable[None]],
    ) -> str:
        """Fast path for conversational messages - no tools, quick response."""
        await emit(AgentEvent(type="thinking"))

        # Add to history
        self.session.append(Message(role=Role.USER, content=user_message))

        # Simple system prompt for chat (no tools)
        chat_system = Message(
            role=Role.SYSTEM,
            content=(
                "You are Loader, a friendly local coding assistant. "
                "Respond naturally and briefly to conversational messages. "
                "If the user wants to do a coding task, tell them to describe it. "
                "Keep responses short (1-3 sentences)."
            ),
        )

        # Use only recent context for speed
        recent_messages = self.messages[-4:] if len(self.messages) > 4 else self.messages

        # Stream the response
        full_content = ""
        async for chunk in self.backend.stream(
            messages=[chat_system] + recent_messages,
            tools=None,  # No tools for chat
            temperature=0.7,  # More natural
            max_tokens=256,  # Short response
        ):
            if chunk.content:
                await emit(AgentEvent(
                    type="stream",
                    content=chunk.content,
                    is_stream_end=chunk.is_done,
                ))
                full_content += chunk.content

        # Add to history
        self.session.append(Message(role=Role.ASSISTANT, content=full_content))

        await emit(AgentEvent(type="response", content=full_content))
        return full_content

    async def run(
        self,
        user_message: str,
        on_event: Callable[[AgentEvent], None] | Callable[[AgentEvent], Awaitable[None]] | None = None,
        on_confirmation: Callable[[str, str, str], Awaitable[bool]] | None = None,
        on_user_question: Callable[[str, list[str] | None], Awaitable[str]] | None = None,
        use_plan: bool | None = None,
    ) -> str:
        """Run the agent with a user message.

        Args:
            user_message: The user's input
            on_event: Optional callback for streaming events (sync or async)
            on_confirmation: Optional callback for tool confirmation. Takes (tool_name, message, details) and returns True to confirm.
            on_user_question: Optional callback for AskUserQuestion. Takes (question, options) and returns the answer.
            use_plan: Force planning on/off. None = auto-detect.

        Returns:
            The final response text
        """
        import inspect

        async def emit(event: AgentEvent) -> None:
            if on_event:
                result = on_event(event)
                # Support both sync and async callbacks
                if inspect.iscoroutine(result):
                    await result

        # Mark agent as running (enables steering)
        self._is_running = True
        try:
            return await self._run_with_steering(
                user_message,
                emit,
                on_confirmation,
                on_user_question,
                use_plan,
            )
        finally:
            self._is_running = False

    async def _run_with_steering(
        self,
        user_message: str,
        emit: Callable[[AgentEvent], Awaitable[None]],
        on_confirmation: Callable[[str, str, str], Awaitable[bool]] | None,
        on_user_question: Callable[[str, list[str] | None], Awaitable[str]] | None,
        use_plan: bool | None,
    ) -> str:
        """Internal run method that supports steering."""
        cfg = self.config.reasoning

        # Fast path: conversational messages don't need tools
        if is_conversational(user_message):
            return await self._handle_conversational(user_message, emit)

        # Track original task for multi-turn conversations
        # Only set on first non-conversational message
        if self._current_task is None:
            self._current_task = user_message

        # Check if we should decompose the task (higher priority than planning)
        if cfg.decomposition and should_decompose(user_message):
            await emit(AgentEvent(type="thinking", content="Analyzing task complexity..."))
            decomposition = await self._decompose_task(user_message)

            if len(decomposition.subtasks) > 1:
                await emit(AgentEvent(
                    type="decomposition",
                    content=decomposition.to_prompt(),
                    decomposition=decomposition,
                ))

                # Execute each subtask
                while not decomposition.is_complete() and not decomposition.has_failures():
                    subtask = decomposition.next_subtask()
                    if not subtask:
                        break

                    subtask.status = "in_progress"
                    await emit(AgentEvent(
                        type="subtask",
                        content=f"{decomposition.progress_str()} {subtask.description}",
                        subtask=subtask,
                    ))

                    # Run the subtask
                    self.session.append(Message(
                        role=Role.USER,
                        content=f"Execute this subtask: {subtask.description}\n\n"
                                f"Verification: {subtask.verification}",
                    ))
                    subtask_response = await self._run_inner(
                        subtask.description,
                        emit,
                        on_confirmation,
                        on_user_question=on_user_question,
                        original_task=self._current_task,
                    )

                    # Mark based on result (simple heuristic)
                    if "error" in subtask_response.lower() or "failed" in subtask_response.lower():
                        decomposition.mark_failed(subtask.id, subtask_response)
                        if decomposition.can_retry(subtask.id):
                            decomposition.reset_for_retry(subtask.id)
                            await emit(AgentEvent(
                                type="subtask",
                                content=f"Retrying subtask: {subtask.description}",
                                subtask=subtask,
                            ))
                    else:
                        decomposition.mark_completed(subtask.id, subtask_response)

                # Final summary
                if decomposition.is_complete():
                    summary_prompt = (
                        f"All subtasks completed for: {user_message}\n\n"
                        f"{decomposition.to_prompt()}\n\n"
                        "Provide a brief summary of what was accomplished."
                    )
                    self.session.append(Message(role=Role.USER, content=summary_prompt))
                    return await self._run_inner(
                        summary_prompt,
                        emit,
                        on_confirmation,
                        on_user_question=on_user_question,
                        original_task=self._current_task,
                    )
                else:
                    return f"Task partially completed. {decomposition.to_prompt()}"

        # No planning or decomposition - run directly
        self.session.append(Message(role=Role.USER, content=user_message))
        return await self._run_inner(
            user_message,
            emit,
            on_confirmation,
            on_user_question=on_user_question,
            requested_mode=self._requested_workflow_mode(use_plan),
            original_task=self._current_task,
        )

    async def _run_inner(
        self,
        task: str,
        emit: Callable[[AgentEvent], Awaitable[None]],
        on_confirmation: Callable[[str, str, str], Awaitable[bool]] | None = None,
        on_user_question: Callable[[str, list[str] | None], Awaitable[str]] | None = None,
        requested_mode: str | None = None,
        original_task: str | None = None,
    ) -> str:
        """Inner execution loop without planning."""

        runtime = ConversationRuntime(self)
        self.last_turn_summary = await runtime.run_turn(
            task,
            emit,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            requested_mode=requested_mode,
            original_task=original_task,
        )
        return self.last_turn_summary.final_response

    def _requested_workflow_mode(self, use_plan: bool | None) -> str | None:
        """Resolve the explicit workflow-mode override for the current turn."""

        if use_plan is True:
            return WorkflowMode.PLAN.value
        if use_plan is False:
            return WorkflowMode.EXECUTE.value
        if self.config.workflow_mode_override:
            return self.config.workflow_mode_override
        if self.config.auto_plan:
            return WorkflowMode.PLAN.value
        return None

    async def run_streaming(
        self,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent with streaming output from the primary runtime path."""

        queue: asyncio.Queue[AgentEvent | BaseException | None] = asyncio.Queue()

        async def on_event(event: AgentEvent) -> None:
            await queue.put(event)

        async def run_agent() -> None:
            try:
                await self.run(user_message, on_event=on_event)
            except BaseException as exc:  # pragma: no cover - propagated below
                await queue.put(exc)
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
            await task
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def run_explore(
        self,
        user_message: str,
        on_event: Callable[[AgentEvent], None] | Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> str:
        """Run one read-only explore query outside the main workflow runtime."""

        import inspect

        async def emit(event: AgentEvent) -> None:
            if on_event:
                result = on_event(event)
                if inspect.iscoroutine(result):
                    await result

        runtime = ExploreRuntime(self)
        self.last_turn_summary = await runtime.run_query(user_message, emit)
        return self.last_turn_summary.final_response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.prompt_format = None
        self.prompt_sections = []
        self.session = self._create_session(messages=self.messages)
        self._current_task = None
        self.last_turn_summary = None
        self.workflow_mode = WorkflowMode.EXECUTE.value
        self._system_message = None
        self.safeguards.reset()  # Reset all runtime safeguards
