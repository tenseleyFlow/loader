"""Definition-of-done gating and turn finalization for the runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from ..llm.base import Message, Role, ToolCall
from .dod import (
    DefinitionOfDone,
    DefinitionOfDoneStore,
    VerificationEvidence,
    build_verification_summary,
    derive_verification_commands,
)
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .memory import MemoryStore
from .session import normalize_usage
from .tracing import RuntimeTracer
from .workflow import WorkflowMode, extract_verification_commands_from_markdown

EventSink = Callable[[AgentEvent], Awaitable[None]]
WorkflowSetter = Callable[
    [WorkflowMode, DefinitionOfDone, EventSink, TurnSummary, str],
    Awaitable[None],
]


@dataclass
class CompletionGateResult:
    """Outcome of the definition-of-done completion gate."""

    should_continue: bool
    final_response: str


class TurnFinalizer:
    """Owns DoD verification, status emission, and turn finalization."""

    def __init__(
        self,
        agent,
        tracer: RuntimeTracer,
        dod_store: DefinitionOfDoneStore,
        set_workflow_mode: WorkflowSetter,
    ) -> None:
        self.agent = agent
        self.tracer = tracer
        self.dod_store = dod_store
        self.set_workflow_mode = set_workflow_mode

    async def run_definition_of_done_gate(
        self,
        *,
        dod: DefinitionOfDone,
        candidate_response: str,
        emit: EventSink,
        summary: TurnSummary,
        executor: ToolExecutor,
    ) -> CompletionGateResult:
        """Gate completion on DoD state and verification evidence."""

        implementation_item = "Complete the requested work"
        if implementation_item in dod.pending_items:
            dod.pending_items.remove(implementation_item)
            dod.completed_items.append(implementation_item)

        tracked_pending_items = [
            item for item in dod.pending_items if item != "Collect verification evidence"
        ]

        mutating_paths = [path for path in dod.touched_files if path]
        requires_verification = bool(mutating_paths or dod.mutating_actions)
        if tracked_pending_items and not requires_verification:
            pending_text = "\n".join(f"- {item}" for item in tracked_pending_items)
            self.dod_store.save(dod)
            await self.emit_dod_status(emit, dod)
            self.agent.session.append(
                Message(
                    role=Role.USER,
                    content=(
                        "[PENDING WORK REMAINS]\n"
                        "The tracked work items are not complete yet:\n"
                        f"{pending_text}\n\n"
                        "Continue the task, and update TodoWrite as you make progress."
                    ),
                )
            )
            return CompletionGateResult(should_continue=True, final_response="")

        if not requires_verification:
            dod.status = "done"
            dod.last_verification_result = "skipped"
            summary.verification_status = "skipped"
            summary.definition_of_done = dod
            self.dod_store.save(dod)
            await self.emit_dod_status(emit, dod)
            return CompletionGateResult(
                should_continue=False,
                final_response=candidate_response,
            )

        verify_item = "Collect verification evidence"
        if verify_item not in dod.pending_items and verify_item not in dod.completed_items:
            dod.pending_items.append(verify_item)

        if (
            not dod.verification_commands
            and dod.verification_plan
            and Path(dod.verification_plan).exists()
        ):
            dod.verification_commands = extract_verification_commands_from_markdown(
                Path(dod.verification_plan).read_text()
            )

        if not dod.verification_commands:
            dod.verification_commands = derive_verification_commands(
                dod,
                project_root=self.agent.project_root,
                task_statement=dod.task_statement,
            )

        await self.set_workflow_mode(
            WorkflowMode.VERIFY,
            dod=dod,
            emit=emit,
            summary=summary,
            reason="definition-of-done gate requires verification",
        )
        verification_passed = await self.verify_definition_of_done(
            dod=dod,
            emit=emit,
            summary=summary,
            executor=executor,
        )
        if verification_passed:
            if verify_item in dod.pending_items:
                dod.pending_items.remove(verify_item)
            if verify_item not in dod.completed_items:
                dod.completed_items.append(verify_item)
            for pending in list(dod.pending_items):
                if pending not in dod.completed_items:
                    dod.completed_items.append(pending)
            dod.pending_items = []
            dod.status = "done"
            dod.last_verification_result = "passed"
            dod.confidence = "high"
            summary.verification_status = "passed"
            summary.definition_of_done = dod
            self.dod_store.save(dod)
            await self.emit_dod_status(emit, dod)
            verified_response = candidate_response
            verification_summary = build_verification_summary(dod.evidence)
            if verification_summary not in verified_response:
                verified_response = f"{candidate_response.rstrip()}\n\n{verification_summary}"
            return CompletionGateResult(
                should_continue=False,
                final_response=verified_response,
            )

        dod.last_verification_result = "failed"
        summary.verification_status = "failed"
        summary.definition_of_done = dod
        if dod.retry_count >= dod.retry_budget:
            dod.status = "failed"
            dod.confidence = "low"
            self.dod_store.save(dod)
            await self.emit_dod_status(emit, dod)
            failure_summary = build_verification_summary(dod.evidence)
            exhausted_response = (
                "I couldn't verify that the task is complete within the retry budget.\n\n"
                f"{failure_summary}"
            )
            return CompletionGateResult(
                should_continue=False,
                final_response=exhausted_response,
            )

        dod.retry_count += 1
        dod.status = "fixing"
        dod.confidence = "medium"
        self.dod_store.save(dod)
        await self.emit_dod_status(emit, dod)
        await self.set_workflow_mode(
            WorkflowMode.EXECUTE,
            dod=dod,
            emit=emit,
            summary=summary,
            reason="verification failed; returning to execute for fixes",
        )
        failure_prompt = (
            "[DEFINITION OF DONE CHECK FAILED]\n"
            f"Task: {dod.task_statement}\n"
            f"Attempt: {dod.retry_count}/{dod.retry_budget}\n"
            f"Pending items: {', '.join(dod.pending_items)}\n\n"
            f"{build_verification_summary(dod.evidence)}\n\n"
            "Fix the failures above, then finish the task again."
        )
        self.agent.session.append(Message(role=Role.USER, content=failure_prompt))
        return CompletionGateResult(should_continue=True, final_response="")

    async def verify_definition_of_done(
        self,
        *,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        executor: ToolExecutor,
    ) -> bool:
        """Collect verification evidence for one DoD."""

        dod.status = "verifying"
        self.dod_store.save(dod)
        await self.emit_dod_status(emit, dod)

        if not dod.verification_commands:
            summary.verification_status = "failed"
            return False

        dod.evidence = []
        all_passed = True
        for index, command in enumerate(dod.verification_commands, start=1):
            verification_call = ToolCall(
                id=f"verify-{summary.iterations}-{index}",
                name="bash",
                arguments={"command": command, "cwd": str(self.agent.project_root)},
            )
            await emit(
                AgentEvent(
                    type="tool_call",
                    tool_name=verification_call.name,
                    tool_args=verification_call.arguments,
                    phase="verification",
                )
            )
            outcome = await executor.execute_tool_call(
                verification_call,
                on_confirmation=None,
                emit_confirmation=None,
                source="verification",
                skip_duplicate_check=True,
                record_action=False,
                skip_confirmation=True,
            )
            await emit(
                AgentEvent(
                    type="tool_result",
                    content=outcome.event_content,
                    tool_name=verification_call.name,
                    is_error=outcome.is_error,
                    phase="verification",
                )
            )

            metadata = outcome.registry_result.metadata if outcome.registry_result else {}
            evidence = VerificationEvidence(
                command=command,
                passed=not outcome.is_error,
                exit_code=metadata.get("exit_code"),
                stdout=str(metadata.get("stdout", "")),
                stderr=str(metadata.get("stderr", "")),
                output=outcome.result_output,
                kind=self.classify_verification_kind(command),
            )
            dod.evidence.append(evidence)
            all_passed = all_passed and evidence.passed
            summary.tool_result_messages.append(outcome.message)
            self.agent.session.append(outcome.message)

        self.dod_store.save(dod)
        summary.verification_status = "passed" if all_passed else "failed"
        return all_passed

    def finalize_summary(self, summary: TurnSummary) -> TurnSummary:
        """Finalize usage, memory capture, and trace data for one turn."""

        summary.usage["tool_calls"] = len(summary.tool_result_messages)
        summary.usage["iterations"] = summary.iterations
        summary.cumulative_usage = self.agent.session.record_turn_usage(
            summary.usage,
            tool_calls=len(summary.tool_result_messages),
            iterations=summary.iterations,
        )
        summary.session_id = self.agent.session.session_id
        if summary.definition_of_done and summary.definition_of_done.status == "done":
            MemoryStore(self.agent.project_root).capture_definition_of_done(
                build_verification_summary(summary.definition_of_done.evidence)
            )
        summary.trace = list(self.tracer.events)
        return summary

    async def emit_dod_status(self, emit: EventSink, dod: DefinitionOfDone) -> None:
        """Emit the latest definition-of-done status."""

        self.dod_store.save(dod)
        await emit(
            AgentEvent(
                type="dod_status",
                content=(
                    f"DoD: {dod.status} "
                    f"({len(dod.pending_items)} pending"
                    + (
                        f", last verification: {dod.last_verification_result}"
                        if dod.last_verification_result
                        else ""
                    )
                    + ")"
                ),
                dod_status=dod.status,
                pending_items_count=len(dod.pending_items),
                last_verification_result=dod.last_verification_result,
                definition_of_done=dod,
            )
        )

    @staticmethod
    def classify_verification_kind(command: str) -> str:
        """Classify the verification command into a summary kind."""

        command_lower = command.lower()
        if "lint" in command_lower or "ruff" in command_lower:
            return "lint"
        if "type" in command_lower or "mypy" in command_lower or "py_compile" in command_lower:
            return "typecheck"
        if "test" in command_lower or "pytest" in command_lower:
            return "test"
        if "build" in command_lower:
            return "build"
        return "runtime"


def merge_usage(target: dict[str, int], update: dict[str, int]) -> None:
    """Merge normalized usage into an existing usage accumulator."""

    for key, value in normalize_usage(update).items():
        target[key] = target.get(key, 0) + value
