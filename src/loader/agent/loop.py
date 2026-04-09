"""The main agent loop."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from ..context.project import ProjectContext, detect_project
from ..llm.base import LLMBackend, Message
from ..runtime.bootstrap import build_runtime_bootstrap_source
from ..runtime.capabilities import resolve_backend_capability_profile
from ..runtime.events import AgentEvent, TurnSummary
from ..runtime.launcher import build_runtime_launcher
from ..runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from ..runtime.public_shell import (
    RuntimeSessionInstall,
    SteeringMailbox,
    build_event_emitter,
    build_runtime_few_shot_examples,
    build_runtime_system_message,
    create_runtime_session_install,
    load_runtime_session_install,
    refresh_runtime_capability_state,
)
from ..runtime.safeguards import RuntimeSafeguards
from ..runtime.workflow import WorkflowMode
from ..tools.base import ToolRegistry, create_default_registry


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
        self._system_message: Message | None = None
        self._use_react: bool | None = None
        self.capability_profile = resolve_backend_capability_profile(self.backend)
        self.last_turn_summary: TurnSummary | None = None
        self.steering = SteeringMailbox()
        self.session = create_runtime_session_install(
            project_root=self.project_root,
            messages=self.messages,
            permission_policy=self.permission_policy,
            permission_config_status=self.permission_config_status,
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
            system_message_factory=self._get_system_message,
            few_shot_factory=self._get_few_shot_examples,
        ).session

        # Track original task for multi-turn conversations
        self._current_task: str | None = None

        # Runtime safeguards for filtering, steering, and deduplication
        self.safeguards = RuntimeSafeguards()

        # Load project context if enabled
        self.project_context: ProjectContext | None = None
        if self.config.auto_context:
            self.project_context = detect_project(self.project_root)

    def _install_runtime_session(self, install: RuntimeSessionInstall) -> None:
        """Install one restored runtime session into the agent shell."""

        self.steering.clear()
        self.session = install.session
        self.messages = install.restored.messages
        self._current_task = install.restored.current_task
        self.set_workflow_mode(install.restored.workflow_mode)
        self.permission_policy.active_mode = PermissionMode.from_str(
            install.restored.permission_mode
        )
        self.prompt_format = install.restored.prompt_format
        self.prompt_sections = list(install.restored.prompt_sections)
        self.last_turn_summary = install.restored.last_turn_summary
        self._system_message = None

    def _build_fresh_session_install(
        self,
        *,
        messages: list[Message] | None = None,
        workflow_mode: str | None = None,
    ) -> RuntimeSessionInstall:
        """Build a fresh runtime session plus its restored shell view."""

        return create_runtime_session_install(
            project_root=self.project_root,
            messages=messages,
            permission_policy=self.permission_policy,
            permission_config_status=self.permission_config_status,
            prompt_format=self.prompt_format,
            prompt_sections=list(self.prompt_sections),
            workflow_mode=workflow_mode or self.workflow_mode,
            rotate_after_bytes=self.config.session_rotate_after_bytes,
            auto_compaction_input_tokens_threshold=(
                self.config.session_auto_compaction_input_tokens_threshold
            ),
            compaction_keep_last_messages=(
                self.config.session_compaction_keep_last_messages
            ),
            system_message_factory=self._get_system_message,
            few_shot_factory=self._get_few_shot_examples,
        )

    def resume_session(self, session_id: str | None = None) -> bool:
        """Resume the latest or named persisted session."""

        loaded = load_runtime_session_install(
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
        self._install_runtime_session(loaded)
        return True

    def steer(self, message: str) -> bool:
        """Send a steering message to the agent during execution.

        Returns True if the agent is running and the message was queued,
        False if the agent is not running.
        """
        return self.steering.steer(message)

    @property
    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self.steering.is_running

    @property
    def current_task(self) -> str | None:
        """Expose the current top-level task through the bootstrap contract."""

        return self._current_task

    @current_task.setter
    def current_task(self, value: str | None) -> None:
        self._current_task = value

    @property
    def active_permission_mode(self) -> str:
        """Return the current runtime permission mode."""
        return self.permission_policy.active_mode.as_str()

    @property
    def active_permission_rule_counts(self) -> dict[str, int]:
        """Return rule counts for the active permission policy."""

        return self.permission_policy.rule_counts()

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
            prompt_state = build_runtime_system_message(
                registry=self.registry,
                use_react=self.use_react,
                project_context=self.project_context,
                workflow_mode=self.workflow_mode,
                permission_mode=self.active_permission_mode,
                cwd=self.project_root,
                current_task=self._current_task,
                session=self.session,
            )
            self.prompt_format = prompt_state.prompt_format
            self.prompt_sections = list(prompt_state.prompt_sections)
            self._system_message = prompt_state.system_message
        return self._system_message

    def set_workflow_mode(self, workflow_mode: str) -> None:
        """Update the active workflow mode used by the system prompt."""

        if workflow_mode == self.workflow_mode:
            return
        self.workflow_mode = workflow_mode
        self._system_message = None

    def refresh_capability_profile(self) -> None:
        """Refresh the runtime capability profile from the current backend."""
        refresh = refresh_runtime_capability_state(
            backend=self.backend,
            current_profile=self.capability_profile,
        )
        self.capability_profile = refresh.capability_profile
        if refresh.prompt_reset_required:
            self._system_message = None
        self._use_react = None

    def queue_steering_message(self, message: str) -> None:
        """Queue one runtime steering message."""

        self.steering.queue(message)

    def build_runtime_source(self):
        """Build the explicit runtime bootstrap source for public entrypoints."""

        return build_runtime_bootstrap_source(self)

    def drain_steering_messages(self) -> list[str]:
        """Drain queued runtime steering messages."""

        return self.steering.drain()

    def _get_few_shot_examples(self) -> list[Message]:
        """Get few-shot examples demonstrating proper tool use."""
        return build_runtime_few_shot_examples(use_react=self.use_react)

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
        emit = build_event_emitter(on_event)

        # Mark agent as running (enables steering)
        self.steering.mark_running()
        try:
            launcher = build_runtime_launcher(self.build_runtime_source())
            return await launcher.run_user_message(
                user_message,
                emit,
                on_confirmation=on_confirmation,
                on_user_question=on_user_question,
                use_plan=use_plan,
            )
        finally:
            self.steering.mark_idle()

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
        *,
        fresh: bool = False,
    ) -> str:
        """Run one read-only explore query outside the main workflow runtime."""
        emit = build_event_emitter(on_event)

        launcher = build_runtime_launcher(self.build_runtime_source())
        self.last_turn_summary = await launcher.run_explore(
            user_message,
            emit,
            fresh=fresh,
        )
        return self.last_turn_summary.final_response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.prompt_format = None
        self.prompt_sections = []
        self._current_task = None
        self.last_turn_summary = None
        self.set_workflow_mode(WorkflowMode.EXECUTE.value)
        self._install_runtime_session(
            self._build_fresh_session_install(
                messages=self.messages,
                workflow_mode=WorkflowMode.EXECUTE.value,
            )
        )
        self._system_message = None
        self.safeguards.reset()  # Reset all runtime safeguards
