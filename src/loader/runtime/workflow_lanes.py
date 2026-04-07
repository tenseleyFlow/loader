"""Clarify and plan lane execution for the workflow runtime."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..llm.base import Message, Role, ToolCall
from .clarify_strategy import ClarifySnapshot, build_clarify_question, describe_clarify_slot
from .dod import DefinitionOfDone, DefinitionOfDoneStore
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .workflow import (
    VERIFICATION_SEPARATOR,
    ClarifyBrief,
    ClarifyReview,
    ModeDecision,
    PlanningArtifacts,
    WorkflowArtifactStore,
    WorkflowDecisionKind,
    WorkflowMode,
    WorkflowPolicy,
    WorkflowTimelineEntryKind,
    sync_todos_to_definition_of_done,
)

EventSink = Callable[[AgentEvent], Awaitable[None]]
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None
TimelineAppender = Callable[..., None]


class WorkflowLaneRunner:
    """Run clarify and plan lanes while persisting workflow artifacts."""

    def __init__(
        self,
        agent: Any,
        *,
        artifact_store: WorkflowArtifactStore,
        dod_store: DefinitionOfDoneStore,
        workflow_policy: WorkflowPolicy,
    ) -> None:
        self.agent = agent
        self.artifact_store = artifact_store
        self.dod_store = dod_store
        self.workflow_policy = workflow_policy

    async def run_clarify_mode(
        self,
        *,
        task: str,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        on_user_question: UserQuestionHandler,
        append_timeline: TimelineAppender,
    ) -> ClarifyReview:
        max_rounds = max(1, self.agent.config.clarify_max_rounds)
        rounds: list[tuple[str, str]] = []
        latest_brief: ClarifyBrief | None = None
        review = ClarifyReview(
            should_continue=False,
            reason_code="clarify_complete",
            reason_summary="clarify gathered enough boundaries to proceed",
            unresolved_slots=[],
            focus_slot=None,
        )

        for round_index in range(1, max_rounds + 1):
            latest_brief, question, answer = await self._run_clarify_round(
                task=task,
                emit=emit,
                summary=summary,
                on_user_question=on_user_question,
                round_index=round_index,
                rounds=rounds,
                unresolved_questions=review.unresolved_questions,
                unresolved_slots=review.unresolved_slots,
            )
            rounds.append((question, answer))
            review = self.workflow_policy.review_clarify(
                task=task,
                answer=answer,
                snapshot=self._clarify_snapshot(task, latest_brief),
                round_index=round_index,
                max_rounds=max_rounds,
            )
            if review.should_continue:
                append_timeline(
                    ModeDecision.transition(
                        WorkflowMode.CLARIFY,
                        reason_code=review.reason_code,
                        reason_summary=review.reason_summary,
                        decision_kind=WorkflowDecisionKind.FORCED,
                        unresolved_questions=review.unresolved_questions,
                    ),
                    kind=WorkflowTimelineEntryKind.CLARIFY_CONTINUE,
                    summary=summary,
                )
                continue
            break

        assert latest_brief is not None
        brief_path = self.artifact_store.write_brief(task, latest_brief)
        dod.clarify_brief = str(brief_path)
        dod.acceptance_criteria = list(dict.fromkeys(latest_brief.acceptance_criteria))
        self.dod_store.save(dod)
        append_timeline(
            ModeDecision.transition(
                WorkflowMode.CLARIFY,
                reason_code=review.reason_code,
                reason_summary=review.reason_summary,
                decision_kind=WorkflowDecisionKind.FORCED,
                unresolved_questions=review.unresolved_questions,
            ),
            kind=WorkflowTimelineEntryKind.CLARIFY_EXIT,
            summary=summary,
            artifact_paths=[str(brief_path)],
        )
        await self._emit_artifact(
            emit=emit,
            kind="clarify_brief",
            path=brief_path,
            preview=(
                f"Clarify brief: {brief_path}\n"
                f"Outcome: {latest_brief.desired_outcome[0]}"
            ),
        )
        return review

    async def run_plan_mode(
        self,
        *,
        task: str,
        dod: DefinitionOfDone,
        emit: EventSink,
        refresh_reasons: list[str] | None,
        executor: ToolExecutor | None,
    ) -> None:
        prompt = self._plan_prompt(
            task=task,
            dod=dod,
            refresh_reasons=refresh_reasons,
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
        if refresh_reasons:
            dod.acceptance_criteria = list(dict.fromkeys(artifacts.acceptance_criteria))
        else:
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
            executor=executor,
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

    async def _seed_todos_from_plan(
        self,
        *,
        artifacts: PlanningArtifacts,
        dod: DefinitionOfDone,
        emit: EventSink,
        executor: ToolExecutor | None,
    ) -> None:
        if not artifacts.implementation_steps:
            return
        assert executor is not None

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
        outcome = await executor.execute_tool_call(
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

    async def _run_clarify_round(
        self,
        *,
        task: str,
        emit: EventSink,
        summary: TurnSummary,
        on_user_question: UserQuestionHandler,
        round_index: int,
        rounds: list[tuple[str, str]],
        unresolved_questions: list[str],
        unresolved_slots: list[str],
    ) -> tuple[ClarifyBrief, str, str]:
        ask_tool = self.agent.registry.get("AskUserQuestion")
        assert ask_tool is not None
        response = await self._complete_in_mode(
            prompt=self._clarify_prompt(
                task=task,
                round_index=round_index,
                rounds=rounds,
                unresolved_questions=unresolved_questions,
                unresolved_slots=unresolved_slots,
            ),
            tools=[ask_tool.to_schema()],
            max_tokens=500,
            temperature=0.2,
        )

        question = ""
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            question = str(tool_call.arguments.get("question", "")).strip()
            title = tool_call.arguments.get("title")
            options = tool_call.arguments.get("options")
        else:
            question = self._fallback_clarify_question(
                task,
                response.content,
                unresolved_slots,
            )
            title = None
            options = None

        await emit(
            AgentEvent(
                type="tool_call",
                tool_name="AskUserQuestion",
                tool_args={
                    "question": question,
                    "title": title,
                    "options": options,
                },
                phase="clarify",
            )
        )
        if on_user_question is not None:
            answer = await on_user_question(question, options)
        else:
            answer = ""
        payload = {
            "question": question,
            "title": title,
            "options": options,
            "allow_freeform": True,
            "context": None,
            "answer": answer,
            "status": "answered",
        }
        rendered_result = json.dumps(payload, indent=2, sort_keys=True)
        await emit(
            AgentEvent(
                type="tool_result",
                content=rendered_result,
                tool_name="AskUserQuestion",
                phase="clarify",
            )
        )
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=response.content,
            tool_calls=[tool_call],
        )
        self.agent.session.append(assistant_message)
        tool_result_message = Message.tool_result_message(
            tool_call_id=tool_call.id,
            display_content=rendered_result,
            result_content=rendered_result,
        )
        self.agent.session.append(
            tool_result_message
        )
        summary.assistant_messages.append(assistant_message)
        summary.tool_result_messages.append(tool_result_message)
        brief_response = await self._complete_in_mode(
            prompt=self._clarify_brief_prompt(task, question, answer, rounds),
            tools=None,
            max_tokens=700,
            temperature=0.2,
        )
        brief = (
            ClarifyBrief.from_markdown(
                brief_response.content,
                task_statement=task,
            )
            if brief_response.content.strip()
            else ClarifyBrief.fallback(
                task_statement=task,
                question=question,
                answer=answer,
            )
        )
        return brief, question, answer

    def _clarify_prompt(
        self,
        *,
        task: str,
        round_index: int,
        rounds: list[tuple[str, str]],
        unresolved_questions: list[str],
        unresolved_slots: list[str],
    ) -> str:
        history_lines = []
        for index, (question, answer) in enumerate(rounds, start=1):
            history_lines.extend(
                [
                    f"Round {index} question: {question}",
                    f"Round {index} answer: {answer}",
                ]
            )
        unresolved = "\n".join(f"- {item}" for item in unresolved_questions) or "- none"
        focus_slot = unresolved_slots[0] if unresolved_slots else None
        focus_label = describe_clarify_slot(focus_slot)
        return (
            "Clarify the task before planning or implementation.\n\n"
            f"Task: {task}\n"
            f"Round: {round_index}\n"
            f"Focus slot: {focus_label}\n"
            "Ask exactly one focused question via AskUserQuestion.\n"
            "Use the unresolved questions and prior answers to tighten scope.\n\n"
            "Unresolved questions:\n"
            f"{unresolved}\n\n"
            "Prior clarify history:\n"
            + ("\n".join(history_lines) if history_lines else "- none yet")
        )

    def _clarify_brief_prompt(
        self,
        task: str,
        question: str,
        answer: str,
        rounds: list[tuple[str, str]],
    ) -> str:
        history_lines = []
        for index, (previous_question, previous_answer) in enumerate(rounds, start=1):
            history_lines.extend(
                [
                    f"Round {index} question: {previous_question}",
                    f"Round {index} answer: {previous_answer}",
                ]
            )
        history_lines.extend(
            [
                f"Latest question: {question}",
                f"Latest answer: {answer}",
            ]
        )
        return (
            "Write a concise task brief in markdown using these exact sections:\n"
            "## Task Statement\n"
            "## Desired Outcome\n"
            "## Non Goals\n"
            "## Decision Boundaries\n"
            "## Constraints\n"
            "## Likely Touchpoints\n"
            "## Acceptance Criteria\n\n"
            f"Task: {task}\n\n"
            "Clarify history:\n"
            + "\n".join(history_lines)
        )

    def _plan_prompt(
        self,
        *,
        task: str,
        dod: DefinitionOfDone,
        refresh_reasons: list[str] | None,
    ) -> str:
        brief_text = ""
        if dod.clarify_brief and Path(dod.clarify_brief).exists():
            brief_text = Path(dod.clarify_brief).read_text().strip()

        refresh_block = ""
        if refresh_reasons:
            refresh_block = (
                "Refresh the existing planning artifacts instead of creating a fresh plan "
                "from scratch.\n"
                "Use the current task state and these recovery reasons:\n"
                + "\n".join(f"- {item}" for item in refresh_reasons)
                + "\n\n"
            )

        return (
            f"{refresh_block}"
            "Write planning artifacts in markdown.\n"
            "Respond with an implementation plan, then the marker "
            f"`{VERIFICATION_SEPARATOR}`.\n\n"
            "Implementation plan sections:\n"
            "# Implementation Plan\n"
            "## File Changes\n"
            "## Execution Order\n"
            "## Risks\n\n"
            "Verification plan sections:\n"
            "# Verification Plan\n"
            "## Acceptance Criteria\n"
            "## Verification Commands\n"
            "## Notes\n\n"
            f"Task: {task}\n\n"
            "Clarify brief:\n"
            f"{brief_text or 'No clarify brief recorded.'}"
        )

    @staticmethod
    def _fallback_clarify_question(
        task: str,
        response_content: str,
        unresolved_slots: list[str],
    ) -> str:
        match = re.search(r"([A-Z][^?]+\?)", response_content)
        if match:
            return match.group(1).strip()
        focus_slot = unresolved_slots[0] if unresolved_slots else None
        return build_clarify_question(task, focus_slot)

    @staticmethod
    def _clarify_snapshot(task: str, brief: ClarifyBrief) -> ClarifySnapshot:
        return ClarifySnapshot(
            task_statement=task,
            explicit_sections=list(brief.explicit_sections),
            desired_outcome=list(brief.desired_outcome),
            non_goals=list(brief.non_goals),
            acceptance_criteria=list(brief.acceptance_criteria),
            constraints=list(brief.constraints),
            decision_boundaries=list(brief.decision_boundaries),
            likely_touchpoints=list(brief.likely_touchpoints),
        )
