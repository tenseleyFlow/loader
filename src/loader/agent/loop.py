"""The main agent loop."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from ..llm.base import LLMBackend, Message, Role, ToolCall
from ..tools.base import ToolRegistry, create_default_registry, ConfirmationRequired
from ..context.project import ProjectContext, detect_project
from .prompts import build_system_prompt
from .parsing import parse_tool_calls, format_tool_result
from .planner import Plan, parse_plan, should_plan, format_step_prompt, PLANNING_PROMPT, SHOULD_PLAN_PROMPT
from .recovery import RecoveryContext, format_recovery_prompt, format_failure_message
from .reasoning import (
    TaskDecomposition,
    Subtask,
    SelfCritique,
    ConfidenceAssessment,
    ActionVerification,
    ConfidenceLevel,
    TaskCompletionCheck,
    RollbackPlan,
    RollbackAction,
    RollbackType,
    DECOMPOSITION_PROMPT,
    SELF_CRITIQUE_PROMPT,
    CONFIDENCE_PROMPT,
    VERIFICATION_PROMPT,
    COMPLETION_CHECK_PROMPT,
    parse_decomposition,
    parse_self_critique,
    parse_confidence,
    parse_verification,
    parse_completion_check,
    should_decompose,
    should_self_critique,
    estimate_confidence_quick,
    quick_verify,
    detect_premature_completion,
    get_continuation_prompt,
    is_destructive_tool,
    create_rollback_plan_for_action,
    is_conversational,
    estimate_complexity,
    get_token_budget,
)
from .safeguards import RuntimeSafeguards, ValidationResult


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
    temperature: float = 0.5  # Lower = faster, more focused
    max_tokens: int = 2048  # Reduced from 4096, most responses are shorter
    force_react: bool = False  # Force ReAct even if model supports native tools
    auto_context: bool = True  # Auto-detect project context on startup
    auto_plan: bool = False  # Auto-plan complex tasks (disabled by default - confuses smaller models)
    auto_recover: bool = True  # Auto-recover from tool errors
    max_recovery_attempts: int = 2  # Reduced from 3
    stream: bool = True  # Stream LLM responses for real-time output

    # Reasoning stages configuration
    reasoning: ReasoningConfig = None  # type: ignore

    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning = ReasoningConfig()


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    # Event types: thinking, tool_call, tool_result, response, error, plan, step,
    # recovery, stream, confirmation, steering, decomposition, subtask, critique,
    # confidence, verification
    type: str
    content: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    step_info: str | None = None  # For step progress like "[2/5] Doing X"
    recovery_attempt: int | None = None  # For recovery events
    is_stream_end: bool = False  # For stream events - indicates final chunk
    confirm_message: str | None = None  # For confirmation events
    confirm_details: str | None = None  # For confirmation events
    is_error: bool = False  # For tool_result events

    # Reasoning events
    decomposition: TaskDecomposition | None = None  # For decomposition events
    subtask: Subtask | None = None  # For subtask events
    critique: SelfCritique | None = None  # For critique events
    confidence: ConfidenceAssessment | None = None  # For confidence events
    verification: ActionVerification | None = None  # For verification events
    completion_check: TaskCompletionCheck | None = None  # For completion events
    rollback_plan: RollbackPlan | None = None  # For rollback events
    rollback_action: RollbackAction | None = None  # For individual rollback action


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
        self.registry = registry or create_default_registry()
        self.config = config or AgentConfig()
        self.messages: list[Message] = []
        self._system_message: Message | None = None
        self._use_react: bool | None = None

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
            self.project_context = detect_project(project_root)

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

        # Check if backend supports native tools
        if hasattr(self.backend, "supports_native_tools"):
            supports_native = self.backend.supports_native_tools()
            self._use_react = not supports_native
            # Debug log
            try:
                with open("/tmp/loader_debug.log", "a") as f:
                    f.write(f"[loop] use_react: supports_native={supports_native}, use_react={self._use_react}\n")
            except Exception:
                pass
        else:
            # Default to ReAct for unknown backends
            self._use_react = True

        return self._use_react

    def _get_system_message(self) -> Message:
        """Get the system message with current context."""
        if self._system_message is None:
            tool_schemas = self.registry.get_schemas()

            # Pass ProjectContext directly for project-specific tips
            content = build_system_prompt(
                tools=tool_schemas,
                use_react=self.use_react,
                project_context=self.project_context,
            )
            self._system_message = Message(
                role=Role.SYSTEM,
                content=content,
            )
        return self._system_message

    def _build_messages(self) -> list[Message]:
        """Build the full message list for the LLM."""
        return [self._get_system_message()] + self.messages

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
        self.messages.append(Message(role=Role.USER, content=user_message))

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
        self.messages.append(Message(role=Role.ASSISTANT, content=full_content))

        await emit(AgentEvent(type="response", content=full_content))
        return full_content

    async def run(
        self,
        user_message: str,
        on_event: Callable[[AgentEvent], None] | Callable[[AgentEvent], Awaitable[None]] | None = None,
        on_confirmation: Callable[[str, str, str], Awaitable[bool]] | None = None,
        use_plan: bool | None = None,
    ) -> str:
        """Run the agent with a user message.

        Args:
            user_message: The user's input
            on_event: Optional callback for streaming events (sync or async)
            on_confirmation: Optional callback for tool confirmation. Takes (tool_name, message, details) and returns True to confirm.
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
            return await self._run_with_steering(user_message, emit, on_confirmation, use_plan)
        finally:
            self._is_running = False

    async def _run_with_steering(
        self,
        user_message: str,
        emit: Callable[[AgentEvent], Awaitable[None]],
        on_confirmation: Callable[[str, str, str], Awaitable[bool]] | None,
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
                    self.messages.append(Message(
                        role=Role.USER,
                        content=f"Execute this subtask: {subtask.description}\n\n"
                                f"Verification: {subtask.verification}",
                    ))
                    subtask_response = await self._run_inner(
                        subtask.description, emit, on_confirmation,
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
                    self.messages.append(Message(role=Role.USER, content=summary_prompt))
                    return await self._run_inner(
                        summary_prompt, emit, on_confirmation,
                        original_task=self._current_task,
                    )
                else:
                    return f"Task partially completed. {decomposition.to_prompt()}"

        # Check if we should use planning
        should_use_plan = use_plan
        if should_use_plan is None and self.config.auto_plan:
            await emit(AgentEvent(type="thinking"))
            should_use_plan = await self._should_plan(user_message)

        # If planning, create and execute plan
        if should_use_plan:
            plan = await self._create_plan(user_message)
            if plan.steps:
                await emit(AgentEvent(type="plan", content=plan.to_prompt()))

                # Execute each step
                while not plan.is_complete():
                    step = plan.next_step()
                    if not step:
                        break

                    await emit(AgentEvent(
                        type="step",
                        step_info=f"{plan.progress_str()} {step.description}",
                    ))

                    # Run the step
                    step_prompt = format_step_prompt(plan, step)
                    step_response = await self._run_inner(
                        step_prompt, emit, on_confirmation,
                        original_task=self._current_task,
                    )

                    plan.complete_current()

                # Final summary
                self.messages.append(Message(role=Role.USER, content=user_message))
                summary_prompt = f"I've completed the plan. Summarize what was done:\n{plan.to_prompt()}"
                return await self._run_inner(
                    summary_prompt, emit, on_confirmation,
                    original_task=self._current_task,
                )

        # No planning or decomposition - run directly
        self.messages.append(Message(role=Role.USER, content=user_message))
        return await self._run_inner(
            user_message, emit, on_confirmation,
            original_task=self._current_task,
        )

    async def _run_inner(
        self,
        task: str,
        emit: Callable[[AgentEvent], Awaitable[None]],
        on_confirmation: Callable[[str, str, str], Awaitable[bool]] | None = None,
        original_task: str | None = None,
    ) -> str:
        """Inner execution loop without planning."""
        iterations = 0
        final_response = ""
        actions_taken: list[str] = []  # Track what we've done
        continuation_count = 0  # How many times we've nudged to continue
        empty_retry_count = 0  # How many times we've retried on empty response
        MAX_EMPTY_RETRIES = 5  # More retries before giving up - small models need patience
        extracted_iterations = 0  # How many times we've extracted bracket-format tool calls
        MAX_EXTRACTED_ITERATIONS = 3  # Limit extracted tool call loops
        consecutive_errors = 0  # Track consecutive tool errors

        # Adaptive token budgeting based on task complexity
        complexity = estimate_complexity(task)
        max_tokens, _ = get_token_budget(complexity)
        # Use configured max_tokens as ceiling, complexity as floor
        effective_max_tokens = min(self.config.max_tokens, max(max_tokens, 512))

        # Rollback planning
        rollback_plan = RollbackPlan() if self.config.reasoning.rollback else None

        while iterations < self.config.max_iterations:
            iterations += 1

            # Check for steering messages from user
            steering_messages = self._drain_steering_queue()
            for steer_msg in steering_messages:
                await emit(AgentEvent(type="steering", content=steer_msg))
                self.messages.append(Message(
                    role=Role.USER,
                    content=f"[USER INTERRUPTION]: {steer_msg}",
                ))

            # Get completion from LLM
            await emit(AgentEvent(type="thinking"))

            # Reset code block filter state for this LLM call
            self.safeguards.code_filter.reset()

            # Pass tools only for native tool calling
            tools = None if self.use_react else self.registry.get_schemas()

            # Use streaming or regular completion
            pending_tool_calls_seen: set[str] = set()  # Track IDs of pending tool calls shown
            if self.config.stream:
                full_content = ""
                full_content_unfiltered = ""  # Keep original for history
                tool_calls: list[ToolCall] = []

                async for chunk in self.backend.stream(
                    messages=self._build_messages(),
                    tools=tools,
                    temperature=self.config.temperature,
                    max_tokens=effective_max_tokens,
                ):
                    # Filter content through safeguards (removes code blocks)
                    filtered_content = ""
                    if chunk.content:
                        filtered_content = self.safeguards.filter_stream_chunk(chunk.content)
                        full_content_unfiltered += chunk.content

                    # Emit stream events for filtered content OR for final chunk (to signal end)
                    if filtered_content or chunk.is_done:
                        await emit(AgentEvent(
                            type="stream",
                            content=filtered_content,
                            is_stream_end=chunk.is_done,
                        ))

                    # Check if we should inject steering (bad patterns detected)
                    if self.safeguards.should_steer():
                        steering_msg = self.safeguards.get_steering_message()
                        if steering_msg:
                            # Queue steering for next iteration
                            self._steering_queue.put_nowait(steering_msg)

                    # Show pending tool calls as they're detected (ReAct mode interleaving)
                    if chunk.pending_tool_call and chunk.pending_tool_call.id not in pending_tool_calls_seen:
                        pending_tool_calls_seen.add(chunk.pending_tool_call.id)
                        await emit(AgentEvent(
                            type="tool_call",
                            tool_name=chunk.pending_tool_call.name,
                            tool_args=chunk.pending_tool_call.arguments,
                        ))
                    if chunk.is_done:
                        full_content = chunk.full_content or full_content_unfiltered
                        tool_calls = chunk.tool_calls
                        # Debug log
                        try:
                            with open("/tmp/loader_debug.log", "a") as f:
                                f.write(f"[loop] chunk.is_done: got {len(tool_calls)} tool_calls\n")
                        except Exception:
                            pass

                content = full_content
                response_content = full_content
            else:
                response = await self.backend.complete(
                    messages=self._build_messages(),
                    tools=tools,
                    temperature=self.config.temperature,
                    max_tokens=effective_max_tokens,
                )
                # Filter content through safeguards (removes code blocks)
                response_content = response.content  # Keep original for history
                content = self.safeguards.filter_complete_content(response.content)
                tool_calls = response.tool_calls if not self.use_react else []

                # Check if we should inject steering (bad patterns detected)
                if self.safeguards.should_steer():
                    steering_msg = self.safeguards.get_steering_message()
                    if steering_msg:
                        self._steering_queue.put_nowait(steering_msg)

            # Handle empty responses (common with small models after clarifications)
            if not content.strip():
                empty_retry_count += 1
                if empty_retry_count <= MAX_EMPTY_RETRIES:
                    # Use progressively more direct prompts
                    task_context = original_task or task
                    retry_prompts = [
                        # Retry 1: Gentle nudge with action focus
                        f"Great! Now let me proceed with the task. I'll start by using my tools.",
                        # Retry 2: More explicit about what to do
                        f"I understand. Let me create that now using my tools (write, bash, etc.).",
                        # Retry 3: Very direct instruction
                        f"Proceeding with: {task_context[:80]}. I'll use the write tool to create the files.",
                        # Retry 4: Action-first prompt
                        f"Starting now. First step: create the necessary files and directories.",
                        # Retry 5: Last attempt with full context
                        f"Let me complete this task step by step. The goal is: {task_context[:100]}",
                    ]
                    prompt = retry_prompts[min(empty_retry_count - 1, len(retry_prompts) - 1)]
                    self.messages.append(Message(
                        role=Role.ASSISTANT,
                        content=prompt,  # Add as assistant message to give model a "running start"
                    ))
                    continue
                else:
                    # Give up after max retries - but make the message less alarming
                    await emit(AgentEvent(
                        type="response",
                        content="I need a bit more direction. What specifically would you like me to create or do?",
                    ))
                    break

            # Get tool calls - either native or parsed from text
            if self.use_react:
                # Parse tool calls from text (ReAct mode)
                parsed = parse_tool_calls(content)
                tool_calls = parsed.tool_calls
                content = parsed.content

                # Check if this is a final answer
                if parsed.is_final_answer and not tool_calls:
                    final_response = content
                    self.messages.append(Message(
                        role=Role.ASSISTANT,
                        content=response_content,  # Keep original for history
                    ))
                    await emit(AgentEvent(type="response", content=final_response))
                    break

            # If there are tool calls, execute them
            if tool_calls:
                # Debug log
                try:
                    with open("/tmp/loader_debug.log", "a") as f:
                        f.write(f"[loop] executing {len(tool_calls)} tool_calls\n")
                        for tc in tool_calls:
                            f.write(f"[loop]   - {tc.name}: id={tc.id}, args_keys={list(tc.arguments.keys())}\n")
                except Exception:
                    pass

                # Add assistant message with tool calls
                self.messages.append(Message(
                    role=Role.ASSISTANT,
                    content=response_content,
                    tool_calls=tool_calls,
                ))

                # Execute each tool (with recovery logic)
                for tool_call in tool_calls:
                    cfg = self.config.reasoning

                    # Confidence scoring before execution
                    if cfg.confidence_scoring:
                        context = "\n".join(
                            m.content[:500] for m in self.messages[-5:]
                            if m.content
                        )
                        confidence = await self._assess_confidence(
                            tool_call.name,
                            tool_call.arguments,
                            context,
                        )
                        await emit(AgentEvent(
                            type="confidence",
                            content=f"Confidence: {confidence.level.name} ({confidence.score}/5)",
                            confidence=confidence,
                            tool_name=tool_call.name,
                        ))

                        # If confidence is too low, ask LLM to reconsider
                        if confidence.score < cfg.min_confidence_for_action:
                            low_conf_msg = (
                                f"[LOW CONFIDENCE WARNING] The planned action has low confidence "
                                f"({confidence.level.name}).\n"
                                f"Reasoning: {confidence.reasoning}\n"
                                f"Risks: {', '.join(confidence.risks)}\n"
                                f"Consider an alternative approach or gather more information first."
                            )
                            self.messages.append(Message(
                                role=Role.USER,
                                content=low_conf_msg,
                            ))
                            continue  # Skip this tool call, let LLM reconsider

                    # Only emit tool_call if not already shown during streaming
                    if tool_call.id not in pending_tool_calls_seen:
                        try:
                            with open("/tmp/loader_debug.log", "a") as f:
                                f.write(f"[loop] emitting tool_call event for {tool_call.name}\n")
                        except Exception:
                            pass
                        await emit(AgentEvent(
                            type="tool_call",
                            tool_name=tool_call.name,
                            tool_args=tool_call.arguments,
                        ))
                    else:
                        try:
                            with open("/tmp/loader_debug.log", "a") as f:
                                f.write(f"[loop] SKIPPING tool_call event for {tool_call.name} (already in pending_seen)\n")
                        except Exception:
                            pass

                    # Track this action for completion checking
                    action_desc = f"{tool_call.name}: {str(tool_call.arguments)[:100]}"
                    actions_taken.append(action_desc)

                    # Check for duplicate actions using safeguards
                    is_dup, dup_reason = self.safeguards.check_duplicate(
                        tool_call.name, tool_call.arguments
                    )
                    if is_dup:
                        try:
                            with open("/tmp/loader_debug.log", "a") as f:
                                f.write(f"[loop] SKIPPING duplicate: {dup_reason}\n")
                        except Exception:
                            pass
                        # Add a tool result indicating skip
                        self.messages.append(Message(
                            role=Role.TOOL,
                            content=f"[Skipped - duplicate action: {dup_reason}]",
                            tool_call_id=tool_call.id,
                        ))
                        continue  # Skip to next tool call

                    # Pre-action validation
                    validation = self.safeguards.validate_action(
                        tool_call.name, tool_call.arguments
                    )
                    if not validation.valid:
                        try:
                            with open("/tmp/loader_debug.log", "a") as f:
                                f.write(f"[loop] BLOCKED by validation: {validation.reason}\n")
                        except Exception:
                            pass
                        # Add a tool result with the validation error
                        error_msg = f"[Blocked - {validation.reason}]"
                        if validation.suggestion:
                            error_msg += f" Suggestion: {validation.suggestion}"
                        self.messages.append(Message(
                            role=Role.TOOL,
                            content=error_msg,
                            tool_call_id=tool_call.id,
                        ))
                        await emit(AgentEvent(
                            type="tool_result",
                            content=error_msg,
                            tool_name=tool_call.name,
                            is_error=True,
                        ))
                        continue  # Skip to next tool call

                    # Rollback planning: create rollback action before destructive ops
                    if rollback_plan and is_destructive_tool(tool_call.name, tool_call.arguments):
                        async def read_file_for_backup(path: str) -> str:
                            """Read file contents for backup."""
                            result = await self.registry.execute("read", file_path=path)
                            return result.output if not result.is_error else ""

                        rollback_action = await create_rollback_plan_for_action(
                            tool_call.name,
                            tool_call.arguments,
                            read_file_for_backup,
                        )
                        if rollback_action:
                            rollback_plan.actions.append(rollback_action)
                            if self.config.reasoning.show_rollback_plan:
                                await emit(AgentEvent(
                                    type="rollback",
                                    content=f"Rollback tracked: {rollback_action.description}",
                                    rollback_action=rollback_action,
                                ))

                    # Try to execute, handling confirmation if needed
                    try:
                        result = await self.registry.execute(
                            tool_call.name,
                            **tool_call.arguments,
                        )
                    except ConfirmationRequired as conf:
                        # Emit confirmation event
                        await emit(AgentEvent(
                            type="confirmation",
                            tool_name=conf.tool_name,
                            confirm_message=conf.message,
                            confirm_details=conf.details,
                        ))

                        # If we have a confirmation callback, ask user
                        if on_confirmation:
                            confirmed = await on_confirmation(
                                conf.tool_name,
                                conf.message,
                                conf.details,
                            )
                            if confirmed:
                                # Re-execute with skip_confirmation
                                old_skip = self.registry.skip_confirmation
                                self.registry.skip_confirmation = True
                                try:
                                    result = await self.registry.execute(
                                        tool_call.name,
                                        **tool_call.arguments,
                                    )
                                finally:
                                    self.registry.skip_confirmation = old_skip
                            else:
                                # User declined - create a skip result
                                from ..tools.base import ToolResult
                                result = ToolResult(
                                    output=f"Tool {tool_call.name} was declined by user",
                                    is_error=False,
                                )
                        else:
                            # No callback - treat as auto-confirmed (for non-TUI mode)
                            old_skip = self.registry.skip_confirmation
                            self.registry.skip_confirmation = True
                            try:
                                result = await self.registry.execute(
                                    tool_call.name,
                                    **tool_call.arguments,
                                )
                            finally:
                                self.registry.skip_confirmation = old_skip

                    # Handle errors with recovery
                    if result.is_error and self.config.auto_recover:
                        # Initialize or update recovery context
                        if self._recovery_context is None:
                            self._recovery_context = RecoveryContext(
                                original_tool=tool_call.name,
                                original_args=tool_call.arguments,
                                max_retries=self.config.max_recovery_attempts,
                            )

                        # Check if this or a similar call was already tried (loop detection)
                        if self._recovery_context.is_similar_attempt(tool_call.name, tool_call.arguments):
                            await emit(AgentEvent(
                                type="error",
                                content=f"Loop detected: already tried a similar command. Try a DIFFERENT approach (e.g., read a config file first).",
                                tool_name=tool_call.name,
                            ))
                        else:
                            # Record this attempt
                            self._recovery_context.add_attempt(
                                tool_call.name,
                                tool_call.arguments,
                                result.output,
                            )

                        # Can we retry?
                        if self._recovery_context.can_retry():
                            attempt_num = len(self._recovery_context.attempts)
                            await emit(AgentEvent(
                                type="recovery",
                                content=f"Tool failed, attempting recovery ({attempt_num}/{self._recovery_context.max_retries})",
                                tool_name=tool_call.name,
                                recovery_attempt=attempt_num,
                            ))

                            # Add recovery prompt for LLM
                            recovery_prompt = format_recovery_prompt(
                                self._recovery_context,
                                tool_call.name,
                                tool_call.arguments,
                                result.output,
                            )
                            self.messages.append(Message(
                                role=Role.TOOL,
                                content=recovery_prompt,
                            ))

                            # Continue to let LLM try an alternative
                            continue
                        else:
                            # Max retries exceeded
                            failure_msg = format_failure_message(self._recovery_context)
                            await emit(AgentEvent(
                                type="error",
                                content=failure_msg,
                                tool_name=tool_call.name,
                            ))
                            self._recovery_context = None

                            # Add the final error result
                            result_text = format_tool_result(
                                tool_call.name,
                                failure_msg,
                                is_error=True,
                            )
                            self.messages.append(Message(
                                role=Role.TOOL,
                                content=result_text,
                            ))
                            continue
                    else:
                        # Success or no auto-recover - clear recovery context
                        if not result.is_error:
                            self._recovery_context = None
                            # Record successful action to prevent duplicates
                            self.safeguards.record_action(tool_call.name, tool_call.arguments)

                            # Check for repetitive loop pattern
                            is_loop, loop_desc = self.safeguards.detect_loop()
                            if is_loop:
                                await emit(AgentEvent(
                                    type="error",
                                    content=f"Loop detected: {loop_desc}. Stopping to prevent repetitive behavior.",
                                ))
                                final_response = "I noticed I was repeating the same actions. Let me know what you'd like me to do differently."
                                self.messages.append(Message(
                                    role=Role.ASSISTANT,
                                    content=final_response,
                                ))
                                await emit(AgentEvent(type="response", content=final_response))
                                return final_response

                    await emit(AgentEvent(
                        type="tool_result",
                        content=result.output,
                        tool_name=tool_call.name,
                        is_error=result.is_error,
                    ))

                    # Post-action verification
                    if cfg.verification and not result.is_error:
                        verification = await self._verify_action(
                            tool_call.name,
                            tool_call.arguments,
                            result.output,
                        )
                        await emit(AgentEvent(
                            type="verification",
                            content=f"Verified: {verification.verified}",
                            verification=verification,
                            tool_name=tool_call.name,
                        ))

                        if not verification.verified and verification.needs_correction:
                            # Add correction suggestion for LLM
                            correction_msg = (
                                f"[VERIFICATION FAILED] The action did not produce expected results.\n"
                                f"Discrepancies: {', '.join(verification.discrepancies)}\n"
                                f"Suggestion: {verification.correction_suggestion}"
                            )
                            self.messages.append(Message(
                                role=Role.USER,
                                content=correction_msg,
                            ))
                            # Don't add the tool result - let LLM try correction
                            continue

                    # Add tool result message
                    result_text = format_tool_result(
                        tool_call.name,
                        result.output,
                        result.is_error,
                    )
                    self.messages.append(Message(
                        role=Role.TOOL,
                        content=result_text,
                    ))

                # Continue the loop to get next response
                continue

            # No tool calls - check if model outputted raw JSON tool calls as text
            # Some small models do this instead of using the proper API
            if not tool_calls:
                try:
                    with open("/tmp/loader_debug.log", "a") as f:
                        f.write(f"[loop] no tool_calls, checking for raw JSON/bracket format in content (len={len(content)})\n")
                except Exception:
                    pass
                raw_tool_calls = self._extract_raw_json_tool_calls(content)
                try:
                    with open("/tmp/loader_debug.log", "a") as f:
                        f.write(f"[loop] _extract_raw_json_tool_calls returned {len(raw_tool_calls)} calls\n")
                        for tc in raw_tool_calls:
                            f.write(f"[loop]   - {tc.name}: {list(tc.arguments.keys())}\n")
                except Exception:
                    pass
                if raw_tool_calls:
                    # Successfully extracted tool calls from raw JSON - use them
                    tool_calls = raw_tool_calls
                    # Clear the streamed content (it was raw JSON, looks ugly)
                    await emit(AgentEvent(type="clear_stream"))

            # If we now have tool calls (from raw JSON extraction), execute them
            if tool_calls:
                extracted_iterations += 1

                # Check if we've exceeded extraction limits
                if extracted_iterations > MAX_EXTRACTED_ITERATIONS:
                    # Model keeps outputting bracket-format calls - stop and let user continue
                    final_response = content
                    self.messages.append(Message(role=Role.ASSISTANT, content=response_content))
                    await emit(AgentEvent(
                        type="response",
                        content=final_response + "\n\nLet me know if you'd like me to continue or make changes."
                    ))
                    break

                try:
                    with open("/tmp/loader_debug.log", "a") as f:
                        f.write(f"[loop] executing {len(tool_calls)} extracted tool calls (iteration {extracted_iterations})\n")
                except Exception:
                    pass

                # Track errors in this batch
                batch_errors = 0

                # This duplicates the tool execution logic above, but that's intentional
                # to handle the case where raw JSON tool calls are extracted
                for i, tc in enumerate(tool_calls):
                    # Skip browser/display commands that don't work in terminal
                    if tc.name == "bash":
                        cmd = tc.arguments.get("command", "")
                        if any(x in cmd for x in ["xdg-open", "open ", "firefox", "chrome", "browser"]):
                            try:
                                with open("/tmp/loader_debug.log", "a") as f:
                                    f.write(f"[loop] skipping browser command: {cmd[:50]}\n")
                            except Exception:
                                pass
                            continue

                    # Use safeguards for duplicate checking
                    is_dup, dup_reason = self.safeguards.check_duplicate(tc.name, tc.arguments)
                    if is_dup:
                        try:
                            with open("/tmp/loader_debug.log", "a") as f:
                                f.write(f"[loop] skipping duplicate: {dup_reason}\n")
                        except Exception:
                            pass
                        continue

                    # Pre-action validation
                    validation = self.safeguards.validate_action(tc.name, tc.arguments)
                    if not validation.valid:
                        try:
                            with open("/tmp/loader_debug.log", "a") as f:
                                f.write(f"[loop] BLOCKED by validation: {validation.reason}\n")
                        except Exception:
                            pass
                        error_msg = f"[Blocked - {validation.reason}]"
                        if validation.suggestion:
                            error_msg += f" Suggestion: {validation.suggestion}"
                        await emit(AgentEvent(
                            type="tool_result",
                            content=error_msg,
                            tool_name=tc.name,
                            is_error=True,
                        ))
                        self.messages.append(Message(
                            role=Role.TOOL,
                            content=error_msg,
                        ))
                        batch_errors += 1
                        continue

                    # Small delay between tool executions for better UX
                    if i > 0:
                        await asyncio.sleep(0.4)
                    try:
                        with open("/tmp/loader_debug.log", "a") as f:
                            f.write(f"[loop] executing extracted tool: {tc.name} args={tc.arguments}\n")
                    except Exception:
                        pass
                    actions_taken.append(f"{tc.name}: {str(tc.arguments)[:50]}...")
                    await emit(AgentEvent(
                        type="tool_call",
                        tool_name=tc.name,
                        tool_args=tc.arguments,
                    ))

                    # Execute the tool
                    is_error = False
                    try:
                        result = await self.registry.execute(tc.name, **tc.arguments)
                        result_text = result.output
                        is_error = result.is_error
                    except ConfirmationRequired as e:
                        # Emit confirmation event
                        await emit(AgentEvent(
                            type="confirmation",
                            tool_name=e.tool_name,
                            confirm_message=e.message,
                            confirm_details=e.details,
                        ))
                        if on_confirmation:
                            confirmed = await on_confirmation(tc.name, e.message, e.details)
                            if confirmed:
                                # Re-execute with skip_confirmation
                                old_skip = self.registry.skip_confirmation
                                self.registry.skip_confirmation = True
                                try:
                                    result = await self.registry.execute(tc.name, **tc.arguments)
                                    result_text = result.output
                                    is_error = result.is_error
                                finally:
                                    self.registry.skip_confirmation = old_skip
                            else:
                                result_text = "Tool execution cancelled by user."
                        else:
                            # No callback - auto-confirm for extracted tool calls
                            old_skip = self.registry.skip_confirmation
                            self.registry.skip_confirmation = True
                            try:
                                result = await self.registry.execute(tc.name, **tc.arguments)
                                result_text = result.output
                                is_error = result.is_error
                            finally:
                                self.registry.skip_confirmation = old_skip
                    except Exception as e:
                        result_text = f"Error: {e}"
                        is_error = True

                    # Track errors
                    if is_error:
                        batch_errors += 1
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0  # Reset on success
                        # Record successful action to prevent duplicates
                        self.safeguards.record_action(tc.name, tc.arguments)

                        # Check for repetitive loop pattern
                        is_loop, loop_desc = self.safeguards.detect_loop()
                        if is_loop:
                            await emit(AgentEvent(
                                type="error",
                                content=f"Loop detected: {loop_desc}. Stopping to prevent repetitive behavior.",
                            ))
                            final_response = "I noticed I was repeating the same actions. Let me know what you'd like me to do differently."
                            self.messages.append(Message(
                                role=Role.ASSISTANT,
                                content=final_response,
                            ))
                            await emit(AgentEvent(type="response", content=final_response))
                            return final_response

                    await emit(AgentEvent(
                        type="tool_result",
                        content=result_text,
                        tool_name=tc.name,
                        is_error=is_error,
                    ))

                    self.messages.append(Message(
                        role=Role.ASSISTANT,
                        content=response_content,
                    ))
                    self.messages.append(Message(
                        role=Role.TOOL,
                        content=result_text,
                    ))

                # After executing batch, check if we should stop
                # Stop if: all tools in batch failed, or we have many consecutive errors
                if batch_errors == len(tool_calls) or consecutive_errors >= 3:
                    # All failed or too many consecutive errors - stop trying
                    final_response = "I ran into some issues. Let me know if you'd like me to try a different approach."
                    await emit(AgentEvent(type="response", content=final_response))
                    break

                continue

            # No tool calls - check if model is describing instead of acting
            # IMPORTANT: Check ORIGINAL content before safeguards filtered it!
            # Debug log
            try:
                has_unexecuted = self._contains_unexecuted_code(response_content)
                with open("/tmp/loader_debug.log", "a") as f:
                    f.write(f"[chatbot-check] iterations={iterations}, has_unexecuted={has_unexecuted}\n")
                    f.write(f"[chatbot-check] response_content (first 200): {response_content[:200]}\n")
                    f.write(f"[chatbot-check] filtered content (first 200): {content[:200]}\n")
            except Exception:
                pass

            if self._contains_unexecuted_code(response_content) and iterations < self.config.max_iterations - 1:
                # Model outputted code blocks without using tools - nudge it
                try:
                    with open("/tmp/loader_debug.log", "a") as f:
                        f.write(f"[chatbot-check] TRIGGERING chatbot recovery\n")
                except Exception:
                    pass
                await emit(AgentEvent(
                    type="error",
                    content="⚠ Chatbot mode detected - steering agent to use tools instead of giving instructions",
                ))
                self.messages.append(Message(
                    role=Role.ASSISTANT,
                    content=response_content,
                ))
                self.messages.append(Message(
                    role=Role.USER,
                    content="CRITICAL ERROR: You are PRETENDING to use tools instead of actually using them.\n\n"
                            "DO NOT write:\n"
                            "- 'Used bash tool with command...' (THIS IS FAKE)\n"
                            "- 'Created a file using the write tool...' (THIS IS FAKE)\n"
                            "- 'Here is what I did:' followed by descriptions\n"
                            "- Numbered steps or instructions\n"
                            "- Code blocks for me to copy\n\n"
                            "Your tool calls MUST go through the proper tool interface.\n"
                            "Writing 'Used bash tool...' does NOT execute anything!\n\n"
                            "ACTUALLY call the tools using the tool_call mechanism.\n"
                            "DO IT NOW - stop narrating and start executing.",
                ))
                continue

            # No tool calls and early in the task - MAY be giving up too soon
            # But only intervene if we haven't done ANY work yet
            if not self.use_react and len(actions_taken) == 0 and iterations < self.config.max_iterations - 2:
                # Check if response looks like deflection without having done anything
                deflection_phrases = ["you can", "you should", "you could", "try running"]
                looks_like_deflection = any(p in content.lower() for p in deflection_phrases)

                if looks_like_deflection:
                    self.messages.append(Message(
                        role=Role.ASSISTANT,
                        content=response_content,
                    ))
                    self.messages.append(Message(
                        role=Role.USER,
                        content="Please use your tools to execute the task rather than telling me what to do.",
                    ))
                    continue

            # Self-critique before finalizing (if enabled and response has substance)
            cfg = self.config.reasoning
            if cfg.self_critique and len(content) > 100:
                # Check if we should critique this response
                is_code_response = "```" in content or any(
                    keyword in content.lower()
                    for keyword in ["def ", "function ", "class ", "import "]
                )
                if should_self_critique(content, is_code=is_code_response):
                    context = task
                    critique = await self._self_critique(content, context)

                    await emit(AgentEvent(
                        type="critique",
                        content=f"Self-critique: {len(critique.issues_found)} issues found",
                        critique=critique,
                    ))

                    if critique.can_revise():
                        # Ask for revision
                        revision_msg = (
                            f"[SELF-CRITIQUE] Review your response:\n"
                            f"Issues found: {', '.join(critique.issues_found)}\n"
                            f"Suggestions: {', '.join(critique.suggestions)}\n\n"
                            "Please provide an improved response addressing these issues."
                        )
                        self.messages.append(Message(
                            role=Role.ASSISTANT,
                            content=response_content,
                        ))
                        self.messages.append(Message(
                            role=Role.USER,
                            content=revision_msg,
                        ))
                        critique.revision_count += 1
                        continue  # Loop to get revised response

            # Check for text loop (agent repeating the same response)
            is_text_loop, text_loop_desc = self.safeguards.detect_text_loop(content)
            if is_text_loop:
                await emit(AgentEvent(
                    type="error",
                    content=f"Text loop detected: {text_loop_desc}. Stopping.",
                ))
                final_response = "I seem to be repeating myself. Let me know if you'd like me to try a different approach."
                self.messages.append(Message(
                    role=Role.ASSISTANT,
                    content=final_response,
                ))
                await emit(AgentEvent(type="response", content=final_response))
                return final_response

            # Record response for future loop detection
            self.safeguards.record_response(content)

            # Task completion check - don't give up too early!
            # Use original_task if available (for multi-turn conversations)
            effective_task = original_task or task
            if cfg.completion_check and continuation_count < cfg.max_continuation_prompts:
                # Quick heuristic check first
                if cfg.use_quick_completion:
                    is_premature = detect_premature_completion(effective_task, content, actions_taken)
                else:
                    is_premature = False

                if is_premature:
                    continuation_count += 1
                    continuation_prompt = get_continuation_prompt(effective_task, actions_taken, content)

                    await emit(AgentEvent(
                        type="completion_check",
                        content=f"Task may be incomplete ({len(actions_taken)} actions taken)",
                        completion_check=TaskCompletionCheck(
                            original_task=effective_task,
                            is_complete=False,
                            accomplished=[a.split(":")[0] for a in actions_taken],
                            continuation_prompt=continuation_prompt,
                        ),
                    ))

                    # Add the assistant's response and nudge to continue
                    self.messages.append(Message(
                        role=Role.ASSISTANT,
                        content=response_content,
                    ))
                    self.messages.append(Message(
                        role=Role.USER,
                        content=continuation_prompt,
                    ))
                    continue  # Loop to get continuation

            # This is the final response
            final_response = content

            # If we completed actions, add follow-up question to encourage continued conversation
            if actions_taken and final_response.strip():
                # Only add if the response doesn't already end with a question
                if not final_response.rstrip().endswith('?'):
                    final_response = final_response.rstrip() + "\n\nWould you like me to make any changes or additions?"

            self.messages.append(Message(
                role=Role.ASSISTANT,
                content=response_content,
            ))

            # Emit rollback plan summary if we tracked any actions
            if rollback_plan and rollback_plan.actions:
                await emit(AgentEvent(
                    type="rollback_summary",
                    content=f"Rollback plan: {len(rollback_plan.actions)} action(s) tracked",
                    rollback_plan=rollback_plan,
                ))

            await emit(AgentEvent(type="response", content=final_response))
            break

        return final_response

    async def run_streaming(
        self,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent with streaming output."""
        # Add user message
        self.messages.append(Message(role=Role.USER, content=user_message))

        iterations = 0
        tools = None if self.use_react else self.registry.get_schemas()

        while iterations < self.config.max_iterations:
            iterations += 1

            yield AgentEvent(type="thinking")

            # Stream the response
            full_content = ""
            tool_calls: list[ToolCall] = []

            async for chunk in self.backend.stream(
                messages=self._build_messages(),
                tools=tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ):
                if chunk.content:
                    full_content += chunk.content
                    yield AgentEvent(type="response", content=chunk.content)

                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls

            # In ReAct mode, parse tool calls from text
            if self.use_react:
                parsed = parse_tool_calls(full_content)
                tool_calls = parsed.tool_calls

                if parsed.is_final_answer and not tool_calls:
                    self.messages.append(Message(
                        role=Role.ASSISTANT,
                        content=full_content,
                    ))
                    break

            # If there are tool calls, execute them
            if tool_calls:
                self.messages.append(Message(
                    role=Role.ASSISTANT,
                    content=full_content,
                    tool_calls=tool_calls,
                ))

                for tool_call in tool_calls:
                    yield AgentEvent(
                        type="tool_call",
                        tool_name=tool_call.name,
                        tool_args=tool_call.arguments,
                    )

                    result = await self.registry.execute(
                        tool_call.name,
                        **tool_call.arguments,
                    )

                    yield AgentEvent(
                        type="tool_result",
                        content=result.output,
                        tool_name=tool_call.name,
                    )

                    result_text = format_tool_result(
                        tool_call.name,
                        result.output,
                        result.is_error,
                    )
                    self.messages.append(Message(
                        role=Role.TOOL,
                        content=result_text,
                    ))

                continue

            # No tool calls - done
            self.messages.append(Message(
                role=Role.ASSISTANT,
                content=full_content,
            ))
            break

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
        import re
        import json
        import os

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
        self._recovery_context = None
        self._current_task = None
        self._executed_commands = set()  # Clear command dedup tracking
        self.safeguards.reset()  # Reset all runtime safeguards
