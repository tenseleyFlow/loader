"""The main agent loop."""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from ..context.project import ProjectContext, detect_project
from ..llm.base import LLMBackend, Message
from ..runtime.capabilities import resolve_backend_capability_profile
from ..runtime.events import AgentEvent, TurnSummary
from ..runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from ..runtime.public_shell import (
    SteeringMailbox,
    build_fresh_runtime_session_install,
    clear_runtime_shell_history,
    get_runtime_shell_few_shot_examples,
    get_runtime_shell_system_message,
    refresh_runtime_shell_capability_profile,
    resolve_runtime_shell_use_react,
    resume_runtime_shell_session,
    run_runtime_shell,
    run_runtime_shell_explore,
    set_runtime_shell_workflow_mode,
    stream_runtime_shell,
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

        # Track original task for multi-turn conversations
        self._current_task: str | None = None

        # Runtime safeguards for filtering, steering, and deduplication
        self.safeguards = RuntimeSafeguards()

        self.session = build_fresh_runtime_session_install(self).session

        # Load project context if enabled
        self.project_context: ProjectContext | None = None
        if self.config.auto_context:
            self.project_context = detect_project(self.project_root)

    def resume_session(self, session_id: str | None = None) -> bool:
        """Resume the latest or named persisted session."""
        return resume_runtime_shell_session(self, session_id=session_id)

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
        return resolve_runtime_shell_use_react(self)

    def _get_system_message(self) -> Message:
        """Get the system message with current context."""
        return get_runtime_shell_system_message(self)

    def set_workflow_mode(self, workflow_mode: str) -> None:
        """Update the active workflow mode used by the system prompt."""
        set_runtime_shell_workflow_mode(self, workflow_mode)

    def refresh_capability_profile(self) -> None:
        """Refresh the runtime capability profile from the current backend."""
        refresh_runtime_shell_capability_profile(self)

    def queue_steering_message(self, message: str) -> None:
        """Queue one runtime steering message."""

        self.steering.queue(message)

    def drain_steering_messages(self) -> list[str]:
        """Drain queued runtime steering messages."""

        return self.steering.drain()

    def _get_few_shot_examples(self) -> list[Message]:
        """Get few-shot examples demonstrating proper tool use."""
        return get_runtime_shell_few_shot_examples(self)

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
        return await run_runtime_shell(
            self,
            user_message,
            on_event=on_event,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            use_plan=use_plan,
        )

    async def run_streaming(
        self,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent with streaming output from the primary runtime path."""
        async for event in stream_runtime_shell(self, user_message):
            yield event

    async def run_explore(
        self,
        user_message: str,
        on_event: Callable[[AgentEvent], None] | Callable[[AgentEvent], Awaitable[None]] | None = None,
        *,
        fresh: bool = False,
    ) -> str:
        """Run one read-only explore query outside the main workflow runtime."""
        return await run_runtime_shell_explore(
            self,
            user_message,
            on_event=on_event,
            fresh=fresh,
        )

    def clear_history(self) -> None:
        """Clear conversation history."""
        clear_runtime_shell_history(self)
