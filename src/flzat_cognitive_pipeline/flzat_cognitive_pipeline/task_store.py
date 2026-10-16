"""Task/run state with strict transitions and clarification resume."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from flzat_cognitive_pipeline.domain import (
    PipelineState,
    TaskRecord,
    TERMINAL_STATES,
    TransitionEvent,
)


class TaskStateError(RuntimeError):
    pass


_ALLOWED: dict[PipelineState, set[PipelineState]] = {
    PipelineState.NEW: {PipelineState.CONTEXT_RETRIEVAL},
    PipelineState.CONTEXT_RETRIEVAL: {
        PipelineState.PROMPT_READY,
        PipelineState.REJECTED,
    },
    PipelineState.PROMPT_READY: {PipelineState.VLM_QUEUED},
    PipelineState.VLM_QUEUED: {PipelineState.VLM_ACTIVE, PipelineState.REJECTED},
    PipelineState.VLM_ACTIVE: {PipelineState.OUTPUT_VALIDATING},
    PipelineState.OUTPUT_VALIDATING: {
        PipelineState.VLM_QUEUED,
        PipelineState.WAITING_CLARIFICATION,
        PipelineState.RESPONSE_ONLY,
        PipelineState.REJECTED,
        PipelineState.COMMAND_READY,
    },
    PipelineState.WAITING_CLARIFICATION: {PipelineState.CONTEXT_RETRIEVAL},
    PipelineState.COMMAND_READY: {PipelineState.WRITING_ARTIFACT},
    PipelineState.WRITING_ARTIFACT: {PipelineState.PUBLISHING_COMMAND},
    PipelineState.PUBLISHING_COMMAND: {PipelineState.DISPATCHED},
    PipelineState.DISPATCHED: {
        PipelineState.LOW_RUNNING,
        PipelineState.SUCCEEDED,
        PipelineState.FAILED,
        PipelineState.CANCELLED,
    },
    PipelineState.LOW_RUNNING: {
        PipelineState.SUCCEEDED,
        PipelineState.FAILED,
        PipelineState.CANCELLED,
    },
}


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskRunStore:
    def __init__(
        self,
        *,
        run_id: str = "",
        run_id_factory: Callable[[], str] = _default_run_id,
        clock: Callable[[], str] = _utc_now,
        max_clarifications: int = 3,
    ) -> None:
        if max_clarifications < 1:
            raise ValueError("max_clarifications must be positive")
        self.run_id = run_id.strip() or run_id_factory().strip()
        if (
            not self.run_id.startswith("run-")
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in self.run_id
            )
        ):
            raise ValueError("run_id must be a safe identifier starting with 'run-'")
        self._clock = clock
        self._max_clarifications = max_clarifications
        self._tasks: dict[str, TaskRecord] = {}
        self._task_count = 0
        self._request_counts: dict[str, int] = {}
        self._waiting_task_id = ""

    def accept_transcript(
        self,
        transcript: str,
        mode: str = "task_command",
    ) -> TaskRecord:
        value = transcript.strip()
        if not value:
            raise ValueError("transcript cannot be empty")
        route_mode = mode.strip()
        if not route_mode:
            raise ValueError("task mode cannot be empty")
        if self._waiting_task_id:
            task = self._tasks[self._waiting_task_id]
            if len(task.clarification_history) >= self._max_clarifications:
                raise TaskStateError("clarification limit reached")
            question = task.pending_question
            task.clarification_history.append((question, value))
            task.pending_question = ""
            task.latest_transcript = value
            task.mode = route_mode
            request_count = self._request_counts[task.task_id] + 1
            self._request_counts[task.task_id] = request_count
            task.request_id = f"{task.task_id}-r{request_count:02d}"
            self._waiting_task_id = ""
            self.transition(task, PipelineState.CONTEXT_RETRIEVAL, "clarification answer")
            return task

        self._task_count += 1
        task_id = f"task-{self._task_count:04d}"
        self._request_counts[task_id] = 1
        task = TaskRecord(
            run_id=self.run_id,
            task_id=task_id,
            request_id=f"{task_id}-r01",
            original_transcript=value,
            latest_transcript=value,
            mode=route_mode,
        )
        self._tasks[task.task_id] = task
        self.transition(task, PipelineState.CONTEXT_RETRIEVAL, "transcript accepted")
        return task

    def transition(
        self,
        task: TaskRecord,
        target: PipelineState,
        reason: str,
    ) -> None:
        if task.state in TERMINAL_STATES:
            raise TaskStateError(f"task is terminal: {task.state.value}")
        if target not in _ALLOWED.get(task.state, set()):
            raise TaskStateError(
                f"invalid transition {task.state.value} -> {target.value}"
            )
        task.state = target
        task.transitions.append(TransitionEvent(target, self._clock(), reason.strip()))

    def wait_for_clarification(self, task: TaskRecord, question: str) -> None:
        value = question.strip()
        if not value:
            raise ValueError("clarification question cannot be empty")
        self.transition(task, PipelineState.WAITING_CLARIFICATION, "model clarification")
        task.pending_question = value
        self._waiting_task_id = task.task_id

    def mark_command(self, task: TaskRecord, command_id: str) -> None:
        if task.command_id:
            raise TaskStateError("task already owns a command")
        value = command_id.strip()
        if not value:
            raise ValueError("command_id cannot be empty")
        task.command_id = value

    def force_terminal(
        self,
        task: TaskRecord,
        target: PipelineState,
        reason: str,
    ) -> None:
        if target not in TERMINAL_STATES:
            raise ValueError("force_terminal requires a terminal state")
        if task.state in TERMINAL_STATES:
            return
        task.state = target
        task.transitions.append(TransitionEvent(target, self._clock(), reason.strip()))
        if self._waiting_task_id == task.task_id:
            self._waiting_task_id = ""

    def get(self, task_id: str) -> TaskRecord:
        return self._tasks[task_id]

    @property
    def waiting_task_id(self) -> str:
        return self._waiting_task_id
