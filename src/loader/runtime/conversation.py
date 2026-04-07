"""Typed turn engine for Loader runtime execution."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..agent.parsing import parse_tool_calls
from ..agent.reasoning import (
    RollbackPlan,
    TaskCompletionCheck,
    detect_premature_completion,
    estimate_complexity,
    get_continuation_prompt,
    get_token_budget,
    should_self_critique,
)
from ..llm.base import Message, Role, ToolCall
from .assistant_turns import AssistantTurnRequester
from .dod import DefinitionOfDone, DefinitionOfDoneStore
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .finalization import TurnFinalizer, merge_usage
from .hooks import build_default_tool_hooks
from .tool_batches import ToolBatchRunner
from .tracing import RuntimeTracer
from .workflow import (
    VERIFICATION_SEPARATOR,
    ClarifyBrief,
    ModeRouter,
    PlanningArtifacts,
    WorkflowArtifactStore,
    WorkflowMode,
    build_execute_bridge,
    sync_todos_to_definition_of_done,
)

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = Callable[[str, str, str], Awaitable[bool]] | None
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None


class ConversationRuntime:
    """Runs one explicit conversation turn against the current session."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.tracer = RuntimeTracer()
        self.executor: ToolExecutor | None = None
        self.dod_store = DefinitionOfDoneStore(agent.project_root)
        self.router = ModeRouter()
        self.artifact_store = WorkflowArtifactStore(agent.project_root)
        self.turn_requester = AssistantTurnRequester(agent, self.tracer)
        self.tool_batches = ToolBatchRunner(agent, self.dod_store)
        self.finalizer = TurnFinalizer(
            agent,
            self.tracer,
            self.dod_store,
            self._set_workflow_mode,
        )

    async def run_turn(
        self,
        task: str,
        emit: EventSink,
        on_confirmation: ConfirmationHandler = None,
        on_user_question: UserQuestionHandler = None,
        requested_mode: str | None = None,
        original_task: str | None = None,
    ) -> TurnSummary:
        """Run one task turn and return a structured summary."""

        await self._prepare_runtime_capabilities()

        iterations = 0
        final_response = ""
        actions_taken: list[str] = []
        continuation_count = 0
        empty_retry_count = 0
        max_empty_retries = 5
        extracted_iterations = 0
        max_extracted_iterations = 3
        consecutive_errors = 0

        complexity = estimate_complexity(task)
        max_tokens, _ = get_token_budget(complexity)
        effective_max_tokens = min(self.agent.config.max_tokens, max(max_tokens, 512))

        rollback_plan = RollbackPlan() if self.agent.config.reasoning.rollback else None
        self.executor = ToolExecutor(
            self.agent.registry,
            self.tracer,
            self.agent.permission_policy,
            hooks=build_default_tool_hooks(
                action_tracker=self.agent.safeguards.action_tracker,
                validator=self.agent.safeguards.validator,
                registry=self.agent.registry,
                rollback_plan=rollback_plan,
            ),
        )
        summary = TurnSummary(final_response="")
        summary.session_id = self.agent.session.session_id
        dod = self.dod_store.create_or_resume(
            original_task or task,
            retry_budget=self.agent.config.verification_retry_budget,
        )
        summary.definition_of_done = dod
        self.agent.session.update_runtime_state(
            active_dod_path=dod.storage_path,
            current_task=original_task or task,
            workflow_mode=self.agent.workflow_mode,
            permission_mode=self.agent.active_permission_mode,
            permission_prompting_enabled=self.agent.permission_policy.prompting_enabled,
            permission_rule_counts=self.agent.active_permission_rule_counts,
            permission_rules_source=str(self.agent.permission_config_status.source_path),
        )
        await self.finalizer.emit_dod_status(emit, dod)

        task = await self._prepare_workflow(
            task=task,
            dod=dod,
            emit=emit,
            summary=summary,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            requested_mode=requested_mode,
        )

        while iterations < self.agent.config.max_iterations:
            iterations += 1
            summary.iterations = iterations
            self.tracer.record("turn.iteration_started", iteration=iterations)

            if iterations == 1 and len(self.agent.messages) == 1:
                task_lower = task.lower()
                action_keywords = [
                    "create",
                    "write",
                    "make",
                    "run",
                    "execute",
                    "build",
                    "install",
                    "delete",
                    "remove",
                    "add",
                    "edit",
                    "modify",
                    "update",
                    "fix",
                ]
                if any(keyword in task_lower for keyword in action_keywords):
                    self.agent.session.append(Message(role=Role.ASSISTANT, content="["))

            steering_messages = self.agent._drain_steering_queue()
            for steering_message in steering_messages:
                await emit(AgentEvent(type="steering", content=steering_message))
                self.agent.session.append(
                    Message(
                        role=Role.USER,
                        content=f"[USER INTERRUPTION]: {steering_message}",
                    )
                )

            await emit(AgentEvent(type="thinking"))
            assistant_turn = await self.turn_requester.request_turn(
                emit=emit,
                max_tokens=effective_max_tokens,
            )
            merge_usage(summary.usage, assistant_turn.usage)

            content = assistant_turn.content
            response_content = assistant_turn.response_content
            tool_calls = list(assistant_turn.tool_calls)
            pending_tool_calls_seen = set(assistant_turn.pending_tool_calls_seen)

            if not content.strip():
                empty_retry_count += 1
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
                    prompt = retry_prompts[min(empty_retry_count - 1, len(retry_prompts) - 1)]
                    self.agent.session.append(Message(role=Role.ASSISTANT, content=prompt))
                    continue

                final_response = (
                    "I need a bit more direction. "
                    "What specifically would you like me to create or do?"
                )
                summary.final_response = final_response
                summary.failures.append("assistant returned empty output repeatedly")
                await emit(AgentEvent(type="response", content=final_response))
                break

            if self.agent.use_react:
                parsed = parse_tool_calls(content)
                tool_calls = parsed.tool_calls
                content = parsed.content

                if parsed.is_final_answer and not tool_calls:
                    assistant_message = Message(role=Role.ASSISTANT, content=response_content)
                    self.agent.session.append(assistant_message)
                    summary.assistant_messages.append(assistant_message)
                    final_response = content
                    summary.final_response = final_response
                    self.tracer.record("turn.completed", reason="final_answer")
                    await emit(AgentEvent(type="response", content=final_response))
                    break

            tool_source = "native"
            if not tool_calls:
                raw_tool_calls = self.agent._extract_raw_json_tool_calls(response_content)
                if raw_tool_calls:
                    tool_calls = raw_tool_calls
                    tool_source = "raw_text"
                    await emit(AgentEvent(type="clear_stream"))

            if tool_calls:
                if tool_source == "raw_text":
                    extracted_iterations += 1
                    if extracted_iterations > max_extracted_iterations:
                        final_response = (
                            content
                            + "\n\nLet me know if you'd like me to continue or make changes."
                        )
                        assistant_message = Message(role=Role.ASSISTANT, content=response_content)
                        self.agent.session.append(assistant_message)
                        summary.assistant_messages.append(assistant_message)
                        summary.final_response = final_response
                        summary.failures.append("raw tool extraction exceeded iteration budget")
                        await emit(AgentEvent(type="response", content=final_response))
                        break

                assistant_message = Message(
                    role=Role.ASSISTANT,
                    content=response_content,
                    tool_calls=tool_calls,
                )
                self.agent.session.append(assistant_message)
                summary.assistant_messages.append(assistant_message)
                self.tracer.record(
                    "assistant.tool_batch",
                    tool_count=len(tool_calls),
                    source=tool_source,
                )

                batch_result = await self.tool_batches.execute_batch(
                    tool_calls=tool_calls,
                    tool_source=tool_source,
                    pending_tool_calls_seen=pending_tool_calls_seen,
                    emit=emit,
                    summary=summary,
                    dod=dod,
                    executor=self.executor,
                    on_confirmation=on_confirmation,
                    on_user_question=on_user_question,
                    emit_confirmation=self._emit_confirmation(emit),
                    consecutive_errors=consecutive_errors,
                )
                actions_taken.extend(batch_result.actions_taken)
                consecutive_errors = batch_result.consecutive_errors
                if batch_result.halted:
                    return self.finalizer.finalize_summary(summary)

                continue

            if self.agent._contains_unexecuted_code(response_content):
                if iterations < self.agent.config.max_iterations - 1:
                    self.agent.session.append(
                        Message(role=Role.ASSISTANT, content=response_content)
                    )
                    self.agent.session.append(
                        Message(
                            role=Role.USER,
                            content=(
                                "CRITICAL ERROR: You are PRETENDING to use tools "
                                "instead of actually "
                                "using them.\n\n"
                                "DO NOT write:\n"
                                "- 'Used bash tool with command...' (THIS IS FAKE)\n"
                                "- 'Created a file using the write tool...' (THIS IS FAKE)\n"
                                "- 'Here is what I did:' followed by descriptions\n"
                                "- Numbered steps or instructions\n"
                                "- Code blocks for me to copy\n\n"
                                "Your tool calls MUST go through the proper tool interface.\n"
                                "Writing 'Used bash tool...' does NOT execute anything!\n\n"
                                "ACTUALLY call the tools using the tool_call mechanism.\n"
                                "DO IT NOW - stop narrating and start executing."
                            ),
                        )
                    )
                    continue

            if (
                not self.agent.use_react
                and len(actions_taken) == 0
                and iterations < self.agent.config.max_iterations - 2
            ):
                deflection_phrases = ["you can", "you should", "you could", "try running"]
                if any(phrase in content.lower() for phrase in deflection_phrases):
                    self.agent.session.append(
                        Message(role=Role.ASSISTANT, content=response_content)
                    )
                    self.agent.session.append(
                        Message(
                            role=Role.USER,
                            content=(
                                "Please use your tools to execute the task "
                                "rather than telling me what to do."
                            ),
                        )
                    )
                    continue

            cfg = self.agent.config.reasoning
            if cfg.self_critique and len(content) > 100:
                is_code_response = "```" in content or any(
                    keyword in content.lower()
                    for keyword in ["def ", "function ", "class ", "import "]
                )
                if should_self_critique(content, is_code=is_code_response):
                    critique = await self.agent._self_critique(content, task)
                    await emit(
                        AgentEvent(
                            type="critique",
                            content=f"Self-critique: {len(critique.issues_found)} issues found",
                            critique=critique,
                        )
                    )
                    if critique.can_revise():
                        revision_message = (
                            "[SELF-CRITIQUE] Review your response:\n"
                            f"Issues found: {', '.join(critique.issues_found)}\n"
                            f"Suggestions: {', '.join(critique.suggestions)}\n\n"
                            "Please provide an improved response addressing these issues."
                        )
                        self.agent.session.append(
                            Message(role=Role.ASSISTANT, content=response_content)
                        )
                        self.agent.session.append(Message(role=Role.USER, content=revision_message))
                        critique.revision_count += 1
                        continue

            is_text_loop, loop_description = self.agent.safeguards.detect_text_loop(content)
            if is_text_loop:
                final_response = (
                    "I seem to be repeating myself. "
                    "Let me know if you'd like me to try a different approach."
                )
                summary.final_response = final_response
                summary.failures.append(loop_description)
                final_message = Message(role=Role.ASSISTANT, content=final_response)
                self.agent.session.append(final_message)
                summary.assistant_messages.append(final_message)
                await emit(
                    AgentEvent(
                        type="error",
                        content=f"Text loop detected: {loop_description}. Stopping.",
                    )
                )
                await emit(AgentEvent(type="response", content=final_response))
                return self.finalizer.finalize_summary(summary)

            self.agent.safeguards.record_response(content)
            effective_task = original_task or task
            if (
                cfg.completion_check
                and not dod.mutating_actions
                and continuation_count < cfg.max_continuation_prompts
            ):
                is_premature = (
                    detect_premature_completion(effective_task, content, actions_taken)
                    if cfg.use_quick_completion
                    else False
                )
                if is_premature:
                    continuation_count += 1
                    continuation_prompt = get_continuation_prompt(
                        effective_task,
                        actions_taken,
                        content,
                    )
                    await emit(
                        AgentEvent(
                            type="completion_check",
                            content=f"Task may be incomplete ({len(actions_taken)} actions taken)",
                            completion_check=TaskCompletionCheck(
                                original_task=effective_task,
                                is_complete=False,
                                accomplished=[action.split(":")[0] for action in actions_taken],
                                continuation_prompt=continuation_prompt,
                            ),
                        )
                    )
                    self.agent.session.append(
                        Message(role=Role.ASSISTANT, content=response_content)
                    )
                    self.agent.session.append(Message(role=Role.USER, content=continuation_prompt))
                    continue

            final_response = content
            if (
                actions_taken
                and final_response.strip()
                and not final_response.rstrip().endswith("?")
            ):
                final_response = (
                    final_response.rstrip()
                    + "\n\nWould you like me to make any changes or additions?"
                )

            final_message = Message(role=Role.ASSISTANT, content=response_content)
            self.agent.session.append(final_message)
            summary.assistant_messages.append(final_message)

            gate_result = await self.finalizer.run_definition_of_done_gate(
                dod=dod,
                candidate_response=final_response,
                emit=emit,
                summary=summary,
                executor=self.executor,
            )
            if gate_result.should_continue:
                continue
            final_response = gate_result.final_response

            if rollback_plan and rollback_plan.actions:
                await emit(
                    AgentEvent(
                        type="rollback_summary",
                        content=f"Rollback plan: {len(rollback_plan.actions)} action(s) tracked",
                        rollback_plan=rollback_plan,
                    )
                )

            summary.final_response = final_response
            await emit(AgentEvent(type="response", content=final_response))
            break

        return self.finalizer.finalize_summary(summary)

    async def _prepare_workflow(
        self,
        *,
        task: str,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        on_confirmation: ConfirmationHandler,
        on_user_question: UserQuestionHandler,
        requested_mode: str | None,
    ) -> str:
        requested = WorkflowMode.from_str(requested_mode)
        decision = self.router.route(
            task,
            requested_mode=requested,
            has_brief=self._artifact_exists(dod.clarify_brief),
            has_plan=self._artifact_exists(dod.implementation_plan)
            and self._artifact_exists(dod.verification_plan),
        )
        await self._set_workflow_mode(
            decision.mode,
            dod=dod,
            emit=emit,
            summary=summary,
            reason=decision.reason,
        )

        if decision.mode == WorkflowMode.CLARIFY:
            await self._run_clarify_mode(
                task=task,
                dod=dod,
                emit=emit,
                summary=summary,
                on_user_question=on_user_question,
            )
            decision = self.router.route(
                task,
                has_brief=self._artifact_exists(dod.clarify_brief),
                has_plan=self._artifact_exists(dod.implementation_plan)
                and self._artifact_exists(dod.verification_plan),
                allow_clarify=False,
            )
            await self._set_workflow_mode(
                decision.mode,
                dod=dod,
                emit=emit,
                summary=summary,
                reason=f"clarify handoff: {decision.reason}",
            )

        if decision.mode == WorkflowMode.PLAN:
            await self._run_plan_mode(
                task=task,
                dod=dod,
                emit=emit,
                summary=summary,
                on_confirmation=on_confirmation,
                on_user_question=on_user_question,
            )
            await self._set_workflow_mode(
                WorkflowMode.EXECUTE,
                dod=dod,
                emit=emit,
                summary=summary,
                reason="plan artifacts created; switching to execute",
            )

        bridge = build_execute_bridge(
            Path(dod.clarify_brief) if dod.clarify_brief else None,
            Path(dod.implementation_plan) if dod.implementation_plan else None,
            Path(dod.verification_plan) if dod.verification_plan else None,
        )
        if bridge and not any(
            message.role == Role.USER and "[WORKFLOW BRIDGE]" in message.content
            for message in self.agent.messages[-4:]
        ):
            self.agent.session.append(
                Message(
                    role=Role.USER,
                    content=(
                        "[WORKFLOW BRIDGE]\n"
                        f"{bridge}\n\n"
                        "Honor these artifacts while you execute the task. "
                        "Keep TodoWrite current when the work spans multiple steps."
                    ),
                )
            )
        return task

    async def _set_workflow_mode(
        self,
        mode: WorkflowMode,
        *,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        reason: str,
    ) -> None:
        self.agent.set_workflow_mode(mode.value)
        self.agent.session.update_runtime_state(workflow_mode=mode.value)
        dod.current_mode = mode.value
        if not dod.mode_history or dod.mode_history[-1] != mode.value:
            dod.mode_history.append(mode.value)
        summary.workflow_mode = mode.value
        summary.definition_of_done = dod
        self.dod_store.save(dod)
        await emit(
            AgentEvent(
                type="workflow_mode",
                content=f"Workflow: {mode.value} ({reason})",
                workflow_mode=mode.value,
                definition_of_done=dod,
            )
        )

    async def _emit_artifact(
        self,
        *,
        emit: EventSink,
        kind: str,
        path: Path,
        preview: str,
    ) -> None:
        await emit(
            AgentEvent(
                type="artifact",
                content=preview,
                artifact_kind=kind,
                artifact_path=str(path),
            )
        )

    async def _complete_in_mode(
        self,
        *,
        prompt: str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float = 0.2,
    ):
        return await self.agent.backend.complete(
            messages=self.agent.session.build_request_messages()
            + [Message(role=Role.USER, content=prompt)],
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _run_clarify_mode(
        self,
        *,
        task: str,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        on_user_question: UserQuestionHandler,
    ) -> None:
        ask_tool = self.agent.registry.get("AskUserQuestion")
        assert ask_tool is not None
        prompt = (
            "Clarify the task before planning or implementation.\n"
            "Ask exactly one focused question with AskUserQuestion.\n"
            "Target missing outcome, scope, or decision-boundary information.\n"
            "Do not propose solutions yet.\n\n"
            f"Task: {task}"
        )
        response = await self._complete_in_mode(
            prompt=prompt,
            tools=[ask_tool.to_schema()],
            max_tokens=300,
        )
        tool_call = next(
            (tool for tool in response.tool_calls if tool.name == "AskUserQuestion"),
            None,
        )
        if tool_call is None:
            tool_call = ToolCall(
                id="clarify-question-1",
                name="AskUserQuestion",
                arguments={
                    "question": self._fallback_clarify_question(task, response.content),
                },
            )

        assistant_message = Message(
            role=Role.ASSISTANT,
            content=response.content or tool_call.arguments.get("question", ""),
            tool_calls=[tool_call],
        )
        self.agent.session.append(assistant_message)
        summary.assistant_messages.append(assistant_message)

        await emit(
            AgentEvent(
                type="tool_call",
                tool_name=tool_call.name,
                tool_args=tool_call.arguments,
                phase="clarify",
            )
        )
        assert self.executor is not None
        outcome = await self.executor.execute_tool_call(
            tool_call,
            on_confirmation=None,
            on_user_question=on_user_question,
            emit_confirmation=None,
            source="clarify",
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
                phase="clarify",
            )
        )
        self.agent.session.append(outcome.message)
        summary.tool_result_messages.append(outcome.message)

        question = str(tool_call.arguments.get("question", "")).strip()
        answer = ""
        if outcome.registry_result is not None:
            answer = str(outcome.registry_result.metadata.get("answer", "")).strip()

        brief_prompt = (
            "Write a concise task brief in markdown using these exact sections:\n"
            "## Task Statement\n"
            "## Desired Outcome\n"
            "## In Scope\n"
            "## Non Goals\n"
            "## Decision Boundaries\n"
            "## Constraints\n"
            "## Likely Touchpoints\n"
            "## Assumptions\n"
            "## Acceptance Criteria\n\n"
            "Use short bullet lists when helpful. Do not start implementing.\n\n"
            f"Task: {task}\n"
            f"Question: {question}\n"
            f"Answer: {answer or 'No answer provided.'}"
        )
        brief_response = await self._complete_in_mode(
            prompt=brief_prompt,
            tools=None,
            max_tokens=900,
            temperature=0.1,
        )
        brief = (
            ClarifyBrief.from_markdown(
                brief_response.content,
                task_statement=task,
                question=question,
                answer=answer,
            )
            if brief_response.content.strip()
            else ClarifyBrief.fallback(
                task_statement=task,
                question=question,
                answer=answer,
            )
        )
        brief_path = self.artifact_store.write_brief(task, brief)
        dod.clarify_brief = str(brief_path)
        dod.acceptance_criteria = list(dict.fromkeys(brief.acceptance_criteria))
        self.dod_store.save(dod)
        await self._emit_artifact(
            emit=emit,
            kind="clarify_brief",
            path=brief_path,
            preview=(f"Clarify brief: {brief_path}\nOutcome: {brief.desired_outcome[0]}"),
        )

    async def _run_plan_mode(
        self,
        *,
        task: str,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        on_confirmation: ConfirmationHandler,
        on_user_question: UserQuestionHandler,
    ) -> None:
        prompt = (
            "Produce two markdown planning artifacts separated by the exact line "
            f"`{VERIFICATION_SEPARATOR}`.\n\n"
            "Before the separator, write an Implementation Plan with these sections:\n"
            "## File Changes\n"
            "## Execution Order\n"
            "## Risks\n\n"
            "After the separator, write a Verification Plan with these sections:\n"
            "## Acceptance Criteria\n"
            "## Verification Commands\n"
            "## Notes\n\n"
            "Do not start writing code.\n\n"
            f"Task: {task}"
        )
        response = await self._complete_in_mode(
            prompt=prompt,
            tools=None,
            max_tokens=1400,
            temperature=0.2,
        )
        artifacts = (
            PlanningArtifacts.from_model_output(
                response.content,
                task_statement=task,
            )
            if response.content.strip()
            else PlanningArtifacts.fallback(task_statement=task)
        )
        implementation_path, verification_path = self.artifact_store.write_plan(
            task,
            artifacts,
        )
        dod.implementation_plan = str(implementation_path)
        dod.verification_plan = str(verification_path)
        dod.acceptance_criteria = list(
            dict.fromkeys(dod.acceptance_criteria + artifacts.acceptance_criteria)
        )
        if artifacts.verification_commands:
            dod.verification_commands = artifacts.verification_commands
        self.dod_store.save(dod)
        await self._emit_artifact(
            emit=emit,
            kind="implementation_plan",
            path=implementation_path,
            preview=(
                f"Implementation plan: {implementation_path}\n"
                f"Steps: {len(artifacts.implementation_steps)}"
            ),
        )
        await self._emit_artifact(
            emit=emit,
            kind="verification_plan",
            path=verification_path,
            preview=(
                f"Verification plan: {verification_path}\n"
                f"Commands: {len(artifacts.verification_commands)}"
            ),
        )
        await self._seed_todos_from_plan(
            artifacts=artifacts,
            dod=dod,
            emit=emit,
        )

    async def _seed_todos_from_plan(
        self,
        *,
        artifacts: PlanningArtifacts,
        dod: DefinitionOfDone,
        emit: EventSink,
    ) -> None:
        if not artifacts.implementation_steps:
            return

        todos = [
            {
                "content": step,
                "active_form": f"Working on: {step}",
                "status": "pending",
            }
            for step in artifacts.implementation_steps[:8]
        ]
        tool_call = ToolCall(
            id="plan-todos-1",
            name="TodoWrite",
            arguments={"todos": todos},
        )
        await emit(
            AgentEvent(
                type="tool_call",
                tool_name=tool_call.name,
                tool_args=tool_call.arguments,
                phase="plan",
            )
        )
        assert self.executor is not None
        outcome = await self.executor.execute_tool_call(
            tool_call,
            on_confirmation=None,
            on_user_question=None,
            emit_confirmation=None,
            source="plan",
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
                phase="plan",
            )
        )
        if outcome.registry_result is not None:
            new_todos = outcome.registry_result.metadata.get("new_todos", [])
            if isinstance(new_todos, list):
                sync_todos_to_definition_of_done(dod, new_todos)
                self.dod_store.save(dod)

    @staticmethod
    def _artifact_exists(path_str: str | None) -> bool:
        return bool(path_str and Path(path_str).exists())

    @staticmethod
    def _fallback_clarify_question(task: str, response_content: str) -> str:
        match = re.search(r"([A-Z][^?]+\?)", response_content)
        if match:
            return match.group(1).strip()
        return (
            "What outcome matters most here, and what should stay out of scope?"
            if task.strip()
            else "What outcome matters most?"
        )

    async def _prepare_runtime_capabilities(self) -> None:
        describe_model = getattr(self.agent.backend, "describe_model", None)
        if callable(describe_model):
            await describe_model()

        previous_profile = self.agent.capability_profile
        self.agent.refresh_capability_profile()
        if self.agent.capability_profile != previous_profile:
            self.tracer.record(
                "runtime.capabilities_refreshed",
                model_name=self.agent.capability_profile.model_name,
                supports_native_tools=self.agent.capability_profile.supports_native_tools,
                preferred_tool_call_format=(
                    self.agent.capability_profile.preferred_tool_call_format
                ),
            )

    @staticmethod
    def _emit_confirmation(emit: EventSink):
        async def _emit(tool_name: str, message: str, details: str) -> None:
            await emit(
                AgentEvent(
                    type="confirmation",
                    tool_name=tool_name,
                    confirm_message=message,
                    confirm_details=details,
                )
            )

        return _emit
