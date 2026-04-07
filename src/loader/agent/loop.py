"""The main agent loop."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from ..context.project import ProjectContext, detect_project
from ..llm.base import LLMBackend, Message, Role, ToolCall
from ..runtime.capabilities import resolve_backend_capability_profile
from ..runtime.conversation import ConversationRuntime
from ..runtime.dod import DefinitionOfDoneStore
from ..runtime.events import AgentEvent, TurnSummary
from ..runtime.explore import ExploreRuntime
from ..runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from ..runtime.session import ConversationSession
from ..runtime.workflow import WorkflowMode
from ..tools.base import ToolRegistry, create_default_registry
from .planner import (
    PLANNING_PROMPT,
    SHOULD_PLAN_PROMPT,
    Plan,
    parse_plan,
    should_plan,
)
from .prompts import build_system_prompt_result
from .reasoning import (
    CONFIDENCE_PROMPT,
    DECOMPOSITION_PROMPT,
    SELF_CRITIQUE_PROMPT,
    VERIFICATION_PROMPT,
    ActionVerification,
    ConfidenceAssessment,
    ConfidenceLevel,
    SelfCritique,
    TaskDecomposition,
    estimate_confidence_quick,
    is_conversational,
    parse_confidence,
    parse_decomposition,
    parse_self_critique,
    parse_verification,
    quick_verify,
    should_decompose,
)
from .recovery import RecoveryContext
from .safeguards import RuntimeSafeguards


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

        # Recovery tracking
        self._recovery_context: RecoveryContext | None = None

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

    async def _should_plan(self, task: str) -> bool:
        """Ask LLM if this task needs planning."""
        if not self.config.auto_plan:
            return False

        prompt = SHOULD_PLAN_PROMPT.format(task=task)
        response = await self.backend.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            tools=None,
            temperature=0.3,
            max_tokens=20,
        )
        return should_plan(response.content)

    async def _create_plan(self, task: str) -> Plan:
        """Generate a plan for the task."""
        prompt = PLANNING_PROMPT.format(task=task)
        response = await self.backend.complete(
            messages=[
                self._get_system_message(),
                Message(role=Role.USER, content=prompt),
            ],
            tools=None,
            temperature=0.5,
            max_tokens=500,
        )
        return parse_plan(response.content, goal=task)

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

    async def _self_critique(self, response: str, context: str) -> SelfCritique:
        """Perform self-critique on a response."""
        prompt = SELF_CRITIQUE_PROMPT.format(response=response, context=context)
        critique_response = await self.backend.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            tools=None,
            temperature=0.3,
            max_tokens=500,
        )
        return parse_self_critique(critique_response.content, response)

    async def _assess_confidence(
        self,
        tool_name: str,
        tool_args: dict,
        context: str = "",
    ) -> ConfidenceAssessment:
        """Assess confidence in a tool action."""
        cfg = self.config.reasoning

        # Try quick heuristic first
        if cfg.use_quick_confidence:
            quick_level = estimate_confidence_quick(tool_name, tool_args, context)
            # Only call LLM if quick estimate is low
            if quick_level.value >= ConfidenceLevel.MEDIUM.value:
                return ConfidenceAssessment(
                    action=f"{tool_name} with {tool_args}",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    level=quick_level,
                    reasoning="Quick heuristic assessment",
                )

        # Full LLM assessment
        action = f"Call {tool_name} with arguments: {tool_args}"
        prompt = CONFIDENCE_PROMPT.format(
            action=action,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context[-2000:] if context else "No prior context",
        )
        response = await self.backend.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            tools=None,
            temperature=0.3,
            max_tokens=300,
        )
        return parse_confidence(response.content, tool_name, tool_args)

    async def _verify_action(
        self,
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        """Verify that an action produced the expected result."""
        cfg = self.config.reasoning

        # Try quick verification first
        if cfg.use_quick_verification:
            quick_result = quick_verify(tool_name, tool_args, result)
            if quick_result:
                return ActionVerification(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    expected_outcome=expected or "Success",
                    actual_result=result[:500],
                    verified=True,
                    verification_method="quick_heuristic",
                )

        # Full LLM verification
        prompt = VERIFICATION_PROMPT.format(
            tool_name=tool_name,
            tool_args=tool_args,
            expected=expected or "The action should complete successfully",
            result=result[:2000],  # Truncate long results
        )
        response = await self.backend.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            tools=None,
            temperature=0.3,
            max_tokens=300,
        )
        return parse_verification(response.content, tool_name, tool_args, expected, result)

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

    def _contains_unexecuted_code(self, content: str) -> bool:
        """Detect if response contains code blocks that should be tool calls.

        Returns True if the response looks like chatbot-style advice with
        code blocks, rather than an actual final answer.
        """
        import re

        # Check for raw JSON tool call attempts (model outputting tool calls as text)
        # This happens when small models try to call tools but output JSON instead
        json_tool_patterns = [
            r'\{"name"\s*:\s*"(write|read|edit|bash|glob|grep)"',  # Tool call JSON
            r'"name"\s*:\s*"(write|read|edit|bash|glob|grep)".*"(?:parameters|arguments)"',
        ]
        for pattern in json_tool_patterns:
            if re.search(pattern, content):
                return True

        # Check for bracket format: [calls bash tool with: ...] or [USE write tool: ...]
        bracket_patterns = [
            r'\[calls?\s+\w+\s+tool\s+with:',
            r'\[USE\s+\w+\s+tool:',
        ]
        for pattern in bracket_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        # Check for hallucinated/narrated tool uses - model DESCRIBES using tools
        # but doesn't actually call them (past tense narration)
        hallucination_patterns = [
            r'used\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool',  # "Used bash tool..."
            r'used\s+the\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool',  # "Used the bash tool..."
            r'using\s+the\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool',  # "...using the write tool"
            r'with\s+file_path\s*=\s*[`\'"]',  # "with file_path=`..." (narrated parameter)
            r'with\s+command\s*[`\'"]',  # "with command `..." (narrated bash)
            r'i\s+(ran|executed|created|wrote|read)\s+(the\s+)?(command|file)',  # "I ran the command"
            r'\*\s*used\s+`',  # "* Used `bash`..." (bullet point narration)
            r'here\s+is\s+what\s+i\s+did:',  # "Here is what I did:"
        ]
        for pattern in hallucination_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        # Look for markdown code blocks
        code_blocks = re.findall(r'```(\w*)\n(.*?)```', content, re.DOTALL)

        if not code_blocks:
            return False

        # Check if any code blocks look like commands or file contents
        action_indicators = [
            'bash', 'sh', 'shell', 'cmd', 'powershell',  # Shell code
            'mkdir', 'cd ', 'npm ', 'pip ', 'git ',  # Commands in code
            'python', 'html', 'css', 'javascript', 'js', 'ts',  # File content
        ]

        chatbot_phrases = [
            'you can run', 'you can create', 'you can use',
            'run this', 'create this', 'save this',
            'here\'s how', 'here is', 'copy this',
            'execute this', 'paste this',
        ]

        # Tutorial/instruction patterns
        tutorial_patterns = [
            r'^\s*\d+\.\s+(open|create|navigate|run|execute|make)',  # Numbered instructions
            r'(first|second|third|next|then),?\s+(open|create|navigate)',  # Sequenced steps
            r'open your (terminal|command|shell)',  # Tutorial starter
            r'navigate to (the|your|~/)',  # Navigation instruction
            r'here\'s how you can (quickly|easily)?',  # How-to preamble
            r'you can (start by|begin by|follow these)',  # Tutorial start
        ]

        content_lower = content.lower()

        # Check for tutorial patterns
        for pattern in tutorial_patterns:
            if re.search(pattern, content_lower, re.MULTILINE | re.IGNORECASE):
                return True

        # If chatbot phrases present with code blocks, it's describing not doing
        for phrase in chatbot_phrases:
            if phrase in content_lower:
                return True

        # Check code block languages that suggest action needed
        for lang, _ in code_blocks:
            if lang.lower() in action_indicators:
                return True

        return False

    def _extract_raw_json_tool_calls(self, content: str) -> list[ToolCall]:
        """Try to extract tool calls from raw JSON or bracket format in content.

        Some small models output tool calls as raw JSON text or bracket format
        instead of using the proper tool calling API. This method tries to
        parse and recover them.
        """
        import json
        import os
        import re

        tool_calls = []
        tool_names = ["write", "read", "edit", "bash", "glob", "grep"]

        # Debug log
        def debug(msg):
            try:
                with open("/tmp/loader_debug.log", "a") as f:
                    f.write(f"[extract] {msg}\n")
            except Exception:
                pass

        debug(f"checking content len={len(content)}")

        # First, try to extract bracket format: [calls bash tool with: ...]
        # or [USE bash tool: ...] or similar variations
        # Note: Using (.+?) with re.DOTALL to capture content that may span patterns
        # The ] at end acts as anchor, but we need to handle ] inside content
        # Also handle formats without colon: [calls bash tool with command="..."]
        bracket_patterns = [
            # With colon after "with"
            r'\[calls?\s+(\w+)\s+tool\s+with:\s*(.+?)\](?=\s*(?:\n|$|[A-Z]|Done|Created|Error))',
            r'\[USE\s+(\w+)\s+tool:\s*(.+?)\](?=\s*(?:\n|$|[A-Z]|Done|Created|Error))',
            r'\[calls?\s+(\w+)\s+tool\s+with:\s*([^\]]+)\]',
            r'\[USE\s+(\w+)\s+tool:\s*([^\]]+)\]',
            # Without colon - direct key=value format: [calls bash tool with command="..."]
            r'\[calls?\s+(\w+)\s+tool\s+with\s+(\w+\s*=.+?)\](?=\s*(?:\n|$|[A-Z]|Done|Created|Error|Directly))',
            r'\[calls?\s+(\w+)\s+tool\s+with\s+([^\]]+)\]',
            # Inline format: [calls write tool with file_path="..." and inline content "..."]
            r'\[calls?\s+(\w+)\s+tool\s+with\s+(.+?)\](?=\s*(?:\n|$|Directly|Done))',
        ]

        for pattern in bracket_patterns:
            debug(f"trying pattern: {pattern}")
            for match in re.finditer(pattern, content, re.IGNORECASE):
                tool_name = match.group(1).lower()
                args_str = match.group(2).strip()
                debug(f"  matched: tool={tool_name}, args={args_str[:50]}...")

                if tool_name not in tool_names:
                    debug(f"  skipping - tool_name '{tool_name}' not in tool_names")
                    continue

                # Skip if we already have a tool call at this position (avoid duplicates)
                match_start = match.start()
                if any(tc.id.endswith(f"_pos{match_start}") for tc in tool_calls):
                    debug(f"  skipping - already extracted at position {match_start}")
                    continue

                try:
                    # Parse the arguments based on tool type
                    if tool_name == "bash":
                        # bash tool: extract command, handling various formats
                        # Model might output: "mkdir -p /foo" or "command='mkdir -p /foo'"
                        cmd = args_str
                        # If it has command= prefix, extract just the command value
                        cmd_match = re.search(r'command\s*[=:]\s*["\']?([^"\']+)["\']?', args_str)
                        if cmd_match:
                            cmd = cmd_match.group(1).strip()
                        # Also handle case where model outputs "cmd, command='cmd'" - take first part
                        elif ',' in args_str and 'command=' in args_str:
                            cmd = args_str.split(',')[0].strip()
                        # Expand ~ in command
                        cmd = os.path.expanduser(cmd)
                        tool_calls.append(ToolCall(
                            id=f"bracket_{tool_name}_{len(tool_calls)}_pos{match_start}",
                            name=tool_name,
                            arguments={"command": cmd},
                        ))
                    elif tool_name == "write":
                        # write tool: file_path=..., content="..."
                        # Handle quoted file paths
                        file_path_match = re.search(r'file_path[=:]\s*["\']?([^"\'`,\s]+)["\']?', args_str)

                        # For content, find the content= part and extract everything after it
                        # Handle both quoted and unquoted content
                        # Also handle "inline content" format: and inline content "..."
                        content_start = re.search(r'(?:inline\s+)?content[=:]\s*', args_str, re.IGNORECASE)
                        if not content_start:
                            # Also try: and inline content "..."
                            content_start = re.search(r'and\s+inline\s+content\s+', args_str, re.IGNORECASE)

                        file_content = ""
                        if content_start:
                            rest = args_str[content_start.end():]
                            # Check if content starts with a quote
                            if rest.startswith('"'):
                                # Find matching end quote (handle escaped quotes)
                                end_idx = len(rest) - 1
                                # Walk backward to find the last quote
                                while end_idx > 0 and rest[end_idx] != '"':
                                    end_idx -= 1
                                if end_idx > 0:
                                    file_content = rest[1:end_idx]
                            elif rest.startswith("'"):
                                end_idx = len(rest) - 1
                                while end_idx > 0 and rest[end_idx] != "'":
                                    end_idx -= 1
                                if end_idx > 0:
                                    file_content = rest[1:end_idx]
                            else:
                                # No quotes - take everything
                                file_content = rest.strip()

                        debug(f"  write: file_path={file_path_match.group(1) if file_path_match else None}, content_len={len(file_content)}")

                        if file_path_match:
                            file_path = file_path_match.group(1).strip('"\'')
                            file_path = os.path.expanduser(file_path)  # Expand ~
                            tool_calls.append(ToolCall(
                                id=f"bracket_{tool_name}_{len(tool_calls)}_pos{match_start}",
                                name=tool_name,
                                arguments={"file_path": file_path, "content": file_content},
                            ))
                    elif tool_name == "read":
                        # read tool: file_path
                        file_path = args_str.split(',')[0].split('=')[-1].strip().strip('"\'')
                        file_path = os.path.expanduser(file_path)
                        tool_calls.append(ToolCall(
                            id=f"bracket_{tool_name}_{len(tool_calls)}_pos{match_start}",
                            name=tool_name,
                            arguments={"file_path": file_path},
                        ))
                    elif tool_name == "edit":
                        # edit tool: file_path=..., old_string="...", new_string="..."
                        file_path_match = re.search(r'file_path[=:]\s*["\']?([^"\'`,]+)["\']?', args_str)
                        old_match = re.search(r'old_string[=:]\s*["\'](.+?)["\']', args_str)
                        new_match = re.search(r'new_string[=:]\s*["\'](.+?)["\']', args_str)

                        if file_path_match and old_match and new_match:
                            file_path = os.path.expanduser(file_path_match.group(1).strip('"\''))
                            tool_calls.append(ToolCall(
                                id=f"bracket_{tool_name}_{len(tool_calls)}_pos{match_start}",
                                name=tool_name,
                                arguments={
                                    "file_path": file_path,
                                    "old_string": old_match.group(1),
                                    "new_string": new_match.group(1),
                                },
                            ))
                    elif tool_name in ("glob", "grep"):
                        # glob/grep: pattern - expand ~ if it looks like a path
                        pattern = args_str
                        if '~' in pattern:
                            pattern = os.path.expanduser(pattern)
                        tool_calls.append(ToolCall(
                            id=f"bracket_{tool_name}_{len(tool_calls)}_pos{match_start}",
                            name=tool_name,
                            arguments={"pattern": pattern},
                        ))
                except Exception:
                    continue

        # If we found bracket-format calls, return them
        if tool_calls:
            return tool_calls

        # Otherwise, try to find JSON objects starting with {"name": "tool_name"
        # This is tricky because the content field may contain arbitrary text

        for tool_name in tool_names:
            # Look for the start of a tool call JSON
            pattern = rf'\{{\s*"name"\s*:\s*"{tool_name}"\s*,\s*"(?:parameters|arguments)"\s*:\s*\{{'
            for match in re.finditer(pattern, content):
                start = match.start()

                # Try to find the matching closing braces by parsing
                # Start from the beginning of the JSON object
                try:
                    # Find the complete JSON by tracking brace depth
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    end = start

                    for i, char in enumerate(content[start:], start):
                        if escape_next:
                            escape_next = False
                            continue

                        if char == '\\' and in_string:
                            escape_next = True
                            continue

                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue

                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end = i + 1
                                    break

                    if brace_count == 0 and end > start:
                        json_str = content[start:end]
                        try:
                            # Try to parse as-is first
                            data = json.loads(json_str)
                        except json.JSONDecodeError:
                            # Model may have output literal newlines in strings
                            # Escape them so JSON parser accepts it
                            try:
                                fixed = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                                data = json.loads(fixed)
                            except json.JSONDecodeError:
                                continue

                        if "name" in data and ("parameters" in data or "arguments" in data):
                            args = data.get("arguments") or data.get("parameters", {})
                            tool_calls.append(ToolCall(
                                id=f"raw_{data['name']}_{len(tool_calls)}",
                                name=data["name"],
                                arguments=args,
                            ))

                except Exception:
                    continue

        return tool_calls

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.prompt_format = None
        self.prompt_sections = []
        self.session = self._create_session(messages=self.messages)
        self._recovery_context = None
        self._current_task = None
        self.last_turn_summary = None
        self.workflow_mode = WorkflowMode.EXECUTE.value
        self._system_message = None
        self.safeguards.reset()  # Reset all runtime safeguards
