"""Public runtime launcher helpers for conversation and explore entrypoints."""

from __future__ import annotations

from .bootstrap import RuntimeBootstrapSource
from .conversation import ConfirmationHandler, ConversationRuntime, EventSink, UserQuestionHandler
from .events import TurnSummary
from .explore import ExploreRuntime


class RuntimeLauncher:
    """Thin launcher over the shared runtime bootstrap contract."""

    def __init__(self, source: RuntimeBootstrapSource) -> None:
        self.source = source

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
        return await runtime.run_turn(
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
        return await runtime.run_query(prompt, emit)


def build_runtime_launcher(source: RuntimeBootstrapSource) -> RuntimeLauncher:
    """Build a public runtime launcher from the shared bootstrap source."""

    return RuntimeLauncher(source)
