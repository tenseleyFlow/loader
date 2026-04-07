"""Typed turn engine for Loader runtime execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..agent.reasoning import (
    RollbackPlan,
    estimate_complexity,
    get_token_budget,
)
from ..llm.base import Message, Role
from .artifact_invalidation import ArtifactInvalidationAssessor
from .assistant_turns import AssistantTurnRequester
from .completion_policy import CompletionPolicy
from .dod import DefinitionOfDone, DefinitionOfDoneStore
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .finalization import TurnFinalizer, merge_usage
from .hooks import build_default_tool_hooks
from .phases import TurnPhase, TurnPhaseTracker, TurnTransitionKind
from .repair import ResponseRepairer
from .tool_batches import ToolBatchRunner
from .tracing import RuntimeTracer
from .workflow import (
    ClarifyReview,
    ModeDecision,
    WorkflowArtifactStore,
    WorkflowDecisionKind,
    WorkflowMode,
    WorkflowPolicy,
    WorkflowSignalExtractor,
    WorkflowTimelineEntry,
    WorkflowTimelineEntryKind,
    build_execute_bridge,
)
from .workflow_lanes import WorkflowLaneRunner
from .workflow_recovery import WorkflowRecoveryController

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
        self.workflow_signals = WorkflowSignalExtractor()
        self.workflow_policy = WorkflowPolicy(self.workflow_signals)
        self.artifact_invalidation = ArtifactInvalidationAssessor()
        self.artifact_store = WorkflowArtifactStore(agent.project_root)
        self.turn_requester = AssistantTurnRequester(agent, self.tracer)
        self.tool_batches = ToolBatchRunner(agent, self.dod_store)
        self.workflow_lanes = WorkflowLaneRunner(
            agent,
            artifact_store=self.artifact_store,
            dod_store=self.dod_store,
            workflow_policy=self.workflow_policy,
        )
        self.workflow_recovery = WorkflowRecoveryController(
            agent,
            artifact_invalidation=self.artifact_invalidation,
            workflow_policy=self.workflow_policy,
            workflow_signals=self.workflow_signals,
            workflow_lanes=self.workflow_lanes,
            set_workflow_mode=self._set_workflow_mode,
            append_timeline=self._append_workflow_timeline_from_decision,
            append_execute_bridge=self._maybe_append_execute_bridge,
        )
        self.repairer = ResponseRepairer(agent)
        self.completion_policy = CompletionPolicy(agent)
        self.phase_tracker = TurnPhaseTracker(agent, self.tracer)
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

        await self.phase_tracker.enter(
            TurnPhase.PREPARE,
            emit,
            detail="Preparing runtime state",
            reason_code="prepare_runtime",
        )
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

            if await self.workflow_recovery.maybe_refresh_plan_for_drift(
                task=original_task or task,
                dod=dod,
                emit=emit,
                summary=summary,
                on_user_question=on_user_question,
                executor=self.executor,
            ):
                continue

            await self.phase_tracker.enter(
                TurnPhase.ASSISTANT,
                emit,
                detail="Requesting assistant response",
                reason_code="request_assistant_response",
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
                await self.phase_tracker.enter(
                    TurnPhase.REPAIR,
                    emit,
                    detail="Repairing empty assistant response",
                    reason_code="repair_empty_response",
                    kind=TurnTransitionKind.RETRY,
                )
                empty_retry_count += 1
                empty_decision = self.repairer.handle_empty_response(
                    task=task,
                    original_task=original_task,
                    empty_retry_count=empty_retry_count,
                    max_empty_retries=max_empty_retries,
                )
                if empty_decision.should_continue and empty_decision.retry_prompt:
                    self.agent.session.append(
                        Message(
                            role=Role.ASSISTANT,
                            content=empty_decision.retry_prompt,
                        )
                    )
                    continue

                final_response = empty_decision.final_response or ""
                summary.final_response = final_response
                if empty_decision.failure:
                    summary.failures.append(empty_decision.failure)
                await emit(AgentEvent(type="response", content=final_response))
                break

            analysis = self.repairer.analyze_response(
                content=content,
                response_content=response_content,
                tool_calls=tool_calls,
                extracted_iterations=extracted_iterations,
                max_extracted_iterations=max_extracted_iterations,
            )
            content = analysis.content
            tool_calls = list(analysis.tool_calls)
            tool_source = analysis.tool_source
            extracted_iterations = analysis.extracted_iterations
            if analysis.clear_stream:
                await self.phase_tracker.enter(
                    TurnPhase.REPAIR,
                    emit,
                    detail="Repairing raw-text tool fallback",
                    reason_code="repair_raw_text_tool_fallback",
                    kind=TurnTransitionKind.REROUTE,
                )
                await emit(AgentEvent(type="clear_stream"))

            if analysis.is_final_answer:
                assistant_message = Message(role=Role.ASSISTANT, content=response_content)
                self.agent.session.append(assistant_message)
                summary.assistant_messages.append(assistant_message)
                final_response = analysis.final_response or content
                summary.final_response = final_response
                self.tracer.record("turn.completed", reason="final_answer")
                await emit(AgentEvent(type="response", content=final_response))
                break

            if tool_calls:
                if analysis.should_stop:
                    assistant_message = Message(role=Role.ASSISTANT, content=response_content)
                    self.agent.session.append(assistant_message)
                    summary.assistant_messages.append(assistant_message)
                    final_response = analysis.final_response or content
                    summary.final_response = final_response
                    if analysis.failure:
                        summary.failures.append(analysis.failure)
                    await emit(AgentEvent(type="response", content=final_response))
                    break

                await self.phase_tracker.enter(
                    TurnPhase.TOOLS,
                    emit,
                    detail="Executing tool batch",
                    reason_code="execute_tool_batch",
                )
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
                    return await self._finalize_turn(
                        summary,
                        emit,
                        reason_code="tool_batch_halted",
                        reason_summary="Finalizing after halted tool batch",
                    )

                continue

            repair_message = self.repairer.fake_tool_narration_message(
                response_content=response_content,
                iterations=iterations,
                max_iterations=self.agent.config.max_iterations,
            )
            if repair_message is not None:
                await self.phase_tracker.enter(
                    TurnPhase.REPAIR,
                    emit,
                    detail="Repairing fake tool narration",
                    reason_code="repair_fake_tool_narration",
                    kind=TurnTransitionKind.REROUTE,
                )
                self.agent.session.append(Message(role=Role.ASSISTANT, content=response_content))
                self.agent.session.append(Message(role=Role.USER, content=repair_message))
                continue

            deflection_message = self.repairer.deflection_message(
                content=content,
                actions_taken=actions_taken,
                iterations=iterations,
                max_iterations=self.agent.config.max_iterations,
            )
            if deflection_message is not None:
                await self.phase_tracker.enter(
                    TurnPhase.REPAIR,
                    emit,
                    detail="Repairing execution deflection",
                    reason_code="repair_execution_deflection",
                    kind=TurnTransitionKind.REROUTE,
                )
                self.agent.session.append(Message(role=Role.ASSISTANT, content=response_content))
                self.agent.session.append(
                    Message(role=Role.USER, content=deflection_message)
                )
                continue

            cfg = self.agent.config.reasoning
            if cfg.self_critique and len(content) > 100:
                await self.phase_tracker.enter(
                    TurnPhase.CRITIQUE,
                    emit,
                    detail="Evaluating self-critique",
                    reason_code="evaluate_self_critique",
                )
                critique_decision = await self.completion_policy.maybe_self_critique(
                    content=content,
                    response_content=response_content,
                    task=task,
                    emit=emit,
                )
                if critique_decision.should_continue:
                    continue

            await self.phase_tracker.enter(
                TurnPhase.COMPLETION,
                emit,
                detail="Checking completion policy",
                reason_code="completion_gate",
            )
            text_loop_decision = await self.completion_policy.maybe_stop_for_text_loop(
                content=content,
                emit=emit,
                summary=summary,
            )
            if text_loop_decision.should_stop:
                return await self._finalize_turn(
                    summary,
                    emit,
                    reason_code="text_loop_bailout",
                    reason_summary="Finalizing after text-loop bailout",
                )

            self.agent.safeguards.record_response(content)
            effective_task = original_task or task
            if (
                cfg.completion_check
                and not dod.mutating_actions
                and continuation_count < cfg.max_continuation_prompts
            ):
                continuation_decision = (
                    await self.completion_policy.maybe_continue_for_completion(
                        content=content,
                        response_content=response_content,
                        task=effective_task,
                        actions_taken=actions_taken,
                        continuation_count=continuation_count,
                        emit=emit,
                    )
                )
                if continuation_decision.should_continue:
                    continuation_count += 1
                    continue

            final_response = self.completion_policy.finalize_response_text(
                content=content,
                actions_taken=actions_taken,
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

        return await self._finalize_turn(
            summary,
            emit,
            reason_code="turn_complete",
            reason_summary="Finalizing completed turn",
        )

    async def _finalize_turn(
        self,
        summary: TurnSummary,
        emit: EventSink,
        *,
        reason_code: str,
        reason_summary: str,
    ) -> TurnSummary:
        await self.phase_tracker.enter(
            TurnPhase.FINALIZE,
            emit,
            detail=reason_summary,
            reason_code=reason_code,
            kind=TurnTransitionKind.TERMINAL,
        )
        final_summary = self.finalizer.finalize_summary(summary)
        self.phase_tracker.clear()
        return final_summary

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
        decision = self.workflow_policy.route_from_signals(
            self.workflow_signals.extract_route_signals(
                task,
                requested_mode=requested.value if requested is not None else None,
                has_brief=self._artifact_exists(dod.clarify_brief),
                has_plan=self._artifact_exists(dod.implementation_plan)
                and self._artifact_exists(dod.verification_plan),
                timeline=self.agent.session.workflow_timeline,
            )
        )
        await self._set_workflow_mode(
            decision,
            dod=dod,
            emit=emit,
            summary=summary,
        )

        clarify_review = ClarifyReview(
            should_continue=False,
            reason_code="clarify_not_needed",
            reason_summary="clarify was not needed for this route",
        )

        if decision.mode == WorkflowMode.CLARIFY:
            clarify_review = await self.workflow_lanes.run_clarify_mode(
                task=task,
                dod=dod,
                emit=emit,
                summary=summary,
                on_user_question=on_user_question,
                append_timeline=self._append_workflow_timeline_from_decision,
            )
            decision = self.workflow_policy.route_from_signals(
                self.workflow_signals.extract_route_signals(
                    task,
                    has_brief=self._artifact_exists(dod.clarify_brief),
                    has_plan=self._artifact_exists(dod.implementation_plan)
                    and self._artifact_exists(dod.verification_plan),
                    allow_clarify=False,
                    unresolved_questions=clarify_review.unresolved_questions,
                    timeline=self.agent.session.workflow_timeline,
                )
            )
            await self._set_workflow_mode(
                decision.with_context(
                    reason_code=f"post_clarify_{decision.reason_code}",
                    reason_summary=f"clarify handoff: {decision.reason_summary}",
                    decision_kind=WorkflowDecisionKind.HANDOFF,
                    unresolved_questions=clarify_review.unresolved_questions,
                ),
                dod=dod,
                emit=emit,
                summary=summary,
            )

        if decision.mode == WorkflowMode.PLAN:
            await self.workflow_lanes.run_plan_mode(
                task=task,
                dod=dod,
                emit=emit,
                refresh_reasons=clarify_review.unresolved_questions or None,
                executor=self.executor,
            )
            await self._set_workflow_mode(
                ModeDecision.transition(
                    WorkflowMode.EXECUTE,
                    reason_code="plan_artifacts_created",
                    reason_summary="plan artifacts created; switching to execute",
                    decision_kind=WorkflowDecisionKind.HANDOFF,
                ),
                dod=dod,
                emit=emit,
                summary=summary,
            )

        self._maybe_append_execute_bridge(dod)
        return task

    async def _set_workflow_mode(
        self,
        decision: ModeDecision,
        *,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
    ) -> None:
        mode = decision.mode
        self.agent.set_workflow_mode(mode.value)
        self.agent.session.update_runtime_state(
            workflow_mode=mode.value,
            workflow_reason_code=decision.reason_code,
            workflow_reason_summary=decision.reason_summary,
            workflow_decision_kind=decision.decision_kind.value,
            workflow_ambiguity_score=decision.ambiguity_score,
            workflow_complexity_score=decision.complexity_score,
            workflow_scheduled_next_mode=(
                decision.scheduled_next_mode.value
                if decision.scheduled_next_mode is not None
                else None
            ),
        )
        dod.current_mode = mode.value
        if not dod.mode_history or dod.mode_history[-1] != mode.value:
            dod.mode_history.append(mode.value)
        summary.workflow_mode = mode.value
        summary.workflow_reason_code = decision.reason_code
        summary.workflow_reason_summary = decision.reason_summary
        summary.workflow_decision_kind = decision.decision_kind.value
        self._append_workflow_timeline_from_decision(
            decision,
            kind={
                WorkflowDecisionKind.HANDOFF: WorkflowTimelineEntryKind.HANDOFF,
                WorkflowDecisionKind.REENTRY: WorkflowTimelineEntryKind.REENTRY,
            }.get(decision.decision_kind, WorkflowTimelineEntryKind.ROUTE),
            summary=summary,
            artifact_paths=[
                path
                for path in (
                    dod.clarify_brief,
                    dod.implementation_plan,
                    dod.verification_plan,
                )
                if path
            ],
        )
        summary.definition_of_done = dod
        self.dod_store.save(dod)
        await emit(
            AgentEvent(
                type="workflow_mode",
                content=f"Workflow: {mode.value} ({decision.reason_summary})",
                workflow_mode=mode.value,
                definition_of_done=dod,
            )
        )

    @staticmethod
    def _artifact_exists(path_str: str | None) -> bool:
        return bool(path_str and Path(path_str).exists())

    def _append_workflow_timeline_from_decision(
        self,
        decision: ModeDecision,
        *,
        kind: WorkflowTimelineEntryKind,
        summary: TurnSummary | None = None,
        artifact_paths: list[str] | None = None,
    ) -> None:
        entry = WorkflowTimelineEntry.from_decision(
            decision,
            kind=kind,
            prompt_format=self.agent.prompt_format,
            prompt_sections=self.agent.prompt_sections,
            artifact_paths=artifact_paths,
        )
        self.agent.session.append_workflow_timeline_entry(entry)
        if summary is not None:
            summary.workflow_timeline = list(self.agent.session.workflow_timeline)

    def _maybe_append_execute_bridge(self, dod: DefinitionOfDone) -> None:
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
