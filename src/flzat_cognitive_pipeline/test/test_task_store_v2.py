import pytest

from flzat_cognitive_pipeline.domain import PipelineState
from flzat_cognitive_pipeline.task_store import TaskRunStore, TaskStateError


def store(max_clarifications=3):
    return TaskRunStore(
        run_id="run-20261014T000000Z",
        clock=lambda: "2026-10-14T00:00:00+00:00",
        max_clarifications=max_clarifications,
    )


def test_new_transcript_creates_correlated_task_and_transition():
    current = store()
    task = current.accept_transcript("go to the cup")

    assert task.run_id == "run-20261014T000000Z"
    assert task.task_id == "task-0001"
    assert task.request_id == "task-0001-r01"
    assert task.state == PipelineState.CONTEXT_RETRIEVAL
    assert task.transitions[-1].reason == "transcript accepted"


def test_clarification_answer_resumes_same_task_with_new_request():
    current = store()
    task = current.accept_transcript("bring the bottle")
    for state in (
        PipelineState.PROMPT_READY,
        PipelineState.VLM_QUEUED,
        PipelineState.VLM_ACTIVE,
        PipelineState.OUTPUT_VALIDATING,
    ):
        current.transition(task, state, "test")
    current.wait_for_clarification(task, "Which bottle?")

    resumed = current.accept_transcript("the blue one")

    assert resumed is task
    assert resumed.task_id == "task-0001"
    assert resumed.request_id == "task-0001-r02"
    assert resumed.clarification_history == [("Which bottle?", "the blue one")]
    assert "Answer: the blue one" in resumed.query_text
    assert resumed.state == PipelineState.CONTEXT_RETRIEVAL


def test_invalid_transition_and_duplicate_command_are_rejected():
    current = store()
    task = current.accept_transcript("find the chair")
    with pytest.raises(TaskStateError, match="invalid transition"):
        current.transition(task, PipelineState.VLM_ACTIVE, "skip")

    current.mark_command(task, "task-0001-c01")
    with pytest.raises(TaskStateError, match="already owns"):
        current.mark_command(task, "task-0001-c02")


def test_task_sequence_is_readable_and_scoped_to_the_run():
    current = store()

    first = current.accept_transcript("find the chair")
    current.force_terminal(first, PipelineState.CANCELLED, "test")
    second = current.accept_transcript("find the desk")

    assert first.task_id == "task-0001"
    assert second.task_id == "task-0002"
    assert second.request_id == "task-0002-r01"
    assert second.run_id == first.run_id


def test_clarification_limit_is_enforced():
    current = store(max_clarifications=1)
    task = current.accept_transcript("bring a cup")
    for state in (
        PipelineState.PROMPT_READY,
        PipelineState.VLM_QUEUED,
        PipelineState.VLM_ACTIVE,
        PipelineState.OUTPUT_VALIDATING,
    ):
        current.transition(task, state, "test")
    current.wait_for_clarification(task, "Which cup?")
    current.accept_transcript("the red cup")
    for state in (
        PipelineState.PROMPT_READY,
        PipelineState.VLM_QUEUED,
        PipelineState.VLM_ACTIVE,
        PipelineState.OUTPUT_VALIDATING,
    ):
        current.transition(task, state, "test")
    current.wait_for_clarification(task, "Where is it?")

    with pytest.raises(TaskStateError, match="limit"):
        current.accept_transcript("on the table")


def test_route_mode_is_stored_and_canonicalized_on_resume():
    current = store()
    task = current.accept_transcript("get that", "clarification")
    for state in (
        PipelineState.PROMPT_READY,
        PipelineState.VLM_QUEUED,
        PipelineState.VLM_ACTIVE,
        PipelineState.OUTPUT_VALIDATING,
    ):
        current.transition(task, state, "test")
    current.wait_for_clarification(task, "Which object?")

    resumed = current.accept_transcript("the book", "task_command")

    assert resumed.mode == "task_command"
    assert resumed.task_id == task.task_id


def test_output_validation_can_queue_bounded_format_repair():
    current = store()
    task = current.accept_transcript("get the book")
    for state in (
        PipelineState.PROMPT_READY,
        PipelineState.VLM_QUEUED,
        PipelineState.VLM_ACTIVE,
        PipelineState.OUTPUT_VALIDATING,
    ):
        current.transition(task, state, "test")

    current.transition(task, PipelineState.VLM_QUEUED, "repair")
    current.transition(task, PipelineState.VLM_ACTIVE, "repair accepted")

    assert task.state == PipelineState.VLM_ACTIVE
