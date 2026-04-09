"""Public runtime launcher helpers for conversation and explore entrypoints."""

from __future__ import annotations

from ..llm.base import Message, Role
from .bootstrap import RuntimeBootstrapSource
from .chat_lane import ConversationalTurnRunner
from .conversation import ConfirmationHandler, ConversationRuntime, EventSink, UserQuestionHandler
from .decomposition_lane import DecompositionTurnRunner
from .deliberation import should_decompose
from .events import TurnSummary
from .explore import ExploreRuntime
from .task_classification import is_conversational
from .workflow import WorkflowMode


class RuntimeLauncher:
    """Thin launcher over the shared runtime bootstrap contract."""

    def __init__(self, source: RuntimeBootstrapSource) -> None:
        self.source = source

    async def run_conversational(
        self,
        user_message: str,
        emit: EventSink,
    ) -> str:
        """Run the runtime-owned conversational fast path."""

        runner = ConversationalTurnRunner(self.source)
        return await runner.run(user_message, emit)

    async def run_user_message(
        self,
        user_message: str,
        emit: EventSink,
        *,
        on_confirmation: ConfirmationHandler = None,
        on_user_question: UserQuestionHandler = None,
        use_plan: bool | None = None,
    ) -> str:
        """Run one user message through the public runtime entrypoint seam."""

        if is_conversational(user_message):
            return await self.run_conversational(user_message, emit)

        if self.source.current_task is None:
            self.source.current_task = user_message

        requested_mode = self._requested_workflow_mode(use_plan)

        if self.source.config.reasoning.decomposition and should_decompose(user_message):
            return await self.run_decomposed(
                user_message,
                emit,
                on_confirmation=on_confirmation,
                on_user_question=on_user_question,
                requested_mode=requested_mode,
                original_task=self.source.current_task,
            )

        self.source.session.append(Message(role=Role.USER, content=user_message))
        return await self._run_task_response(
            user_message,
            emit,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            requested_mode=requested_mode,
            original_task=self.source.current_task,
        )

    async def run_turn(
        self,
        task: str,
        emit: EventSink,
        *,
        on_confirmation: ConfirmationHandler = None,
        on_user_question: UserQuestionHandler = None,
        requested_mode: str | None = None,
        original_task: str | None = None,
    ) -> TurnSummary:
        """Run one conversation turn through the shared launcher seam."""

        runtime = ConversationRuntime(self.source)
        summary = await runtime.run_turn(
            task,
            emit,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            requested_mode=requested_mode,
            original_task=original_task,
        )
        self.source.last_turn_summary = summary
        return summary

    async def run_decomposed(
        self,
        task: str,
        emit: EventSink,
        *,
        on_confirmation: ConfirmationHandler = None,
        on_user_question: UserQuestionHandler = None,
        requested_mode: str | None = None,
        original_task: str | None = None,
    ) -> str:
        """Run a decomposition-guided task through the shared launcher seam."""

        runner = DecompositionTurnRunner(self.source, run_task=self._run_task_response)
        return await runner.run(
            task,
            emit,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            requested_mode=requested_mode,
            original_task=original_task,
        )

    async def run_explore(
        self,
        prompt: str,
        emit: EventSink,
    ) -> TurnSummary:
        """Run one read-only explore query through the shared launcher seam."""

        runtime = ExploreRuntime(self.source)
        summary = await runtime.run_query(prompt, emit)
        self.source.last_turn_summary = summary
        return summary

    async def _run_task_response(
        self,
        task: str,
        emit: EventSink,
        on_confirmation: ConfirmationHandler = None,
        on_user_question: UserQuestionHandler = None,
        requested_mode: str | None = None,
        original_task: str | None = None,
    ) -> str:
        """Run one runtime turn and return only the final response text."""

        summary = await self.run_turn(
            task,
            emit,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            requested_mode=requested_mode,
            original_task=original_task,
        )
        return summary.final_response

    def _requested_workflow_mode(self, use_plan: bool | None) -> str | None:
        """Resolve any explicit workflow-mode request for the current entrypoint."""

        if use_plan is True:
            return WorkflowMode.PLAN.value
        if use_plan is False:
            return WorkflowMode.EXECUTE.value
        if self.source.config.workflow_mode_override:
            return self.source.config.workflow_mode_override
        if self.source.config.auto_plan:
            return WorkflowMode.PLAN.value
        return None


def build_runtime_launcher(source: RuntimeBootstrapSource) -> RuntimeLauncher:
    """Build a public runtime launcher from the shared bootstrap source."""

    return RuntimeLauncher(source)
