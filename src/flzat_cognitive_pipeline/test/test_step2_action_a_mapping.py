"""Action A business outcomes remain distinct from ROS lifecycle status."""

from concurrent.futures import Future
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from action_msgs.msg import GoalStatus
from std_msgs.msg import String

from flzat_cognitive_pipeline.cognitive_manager import (
    CognitiveManagerNode,
    RuntimeRequest,
)
from flzat_cognitive_pipeline.domain import PipelineState, TaskRecord
from flzat_cognitive_pipeline.task_store import TaskRunStore


def manager(outcome: str):
    node = object.__new__(CognitiveManagerNode)
    node._lock = threading.RLock()
    node._tasks = TaskRunStore(run_id="run-test")
    task = TaskRecord(
        run_id="run-test",
        task_id="task-0001",
        request_id="task-0001-r01",
        original_transcript="Bring the bottle.",
        latest_transcript="Bring the bottle.",
        state=PipelineState.HANDOFF_PENDING,
        handoff_id="task-0001-c01",
    )
    runtime = RuntimeRequest(
        task=task,
        deadline_ns=10**30,
        input_record={},
        handoff_id="task-0001-c01",
    )
    node._active = runtime
    node._interaction = None
    node._interaction_reserved = False
    node._speech_busy = False
    node._publish_status = Mock()
    node._finish_metrics = Mock()
    node._publish_action_event = Mock()
    node._publish_ready_if_available = Mock()
    node.get_logger = Mock(return_value=Mock())
    status = {
        "STARTED_FAST": GoalStatus.STATUS_SUCCEEDED,
        "STARTED_PLAN": GoalStatus.STATUS_SUCCEEDED,
        "REFUSED": GoalStatus.STATUS_SUCCEEDED,
        "FAILED": GoalStatus.STATUS_ABORTED,
        "CANCELED": GoalStatus.STATUS_CANCELED,
    }[outcome]
    future = Future()
    future.set_result(
        SimpleNamespace(
            status=status,
            result=SimpleNamespace(outcome=outcome, detail="mock detail"),
        )
    )
    return node, runtime, future


@pytest.mark.parametrize("outcome", ["STARTED_FAST", "STARTED_PLAN"])
def test_started_means_low_running_not_task_succeeded(outcome):
    node, runtime, future = manager(outcome)

    node._on_intent_result("task-0001", "task-0001-r01", future)

    assert runtime.task.state == PipelineState.LOW_RUNNING
    assert node._active is None


@pytest.mark.parametrize(
    "outcome,state",
    [
        ("REFUSED", PipelineState.REFUSED),
        ("FAILED", PipelineState.FAILED),
        ("CANCELED", PipelineState.CANCELLED),
    ],
)
def test_terminal_business_outcomes_keep_their_meaning(outcome, state):
    node, runtime, future = manager(outcome)

    node._on_intent_result("task-0001", "task-0001-r01", future)

    assert runtime.task.state == state


def test_stop_while_a_active_requests_cancel_and_waits_for_low_result():
    node, runtime, _future = manager("CANCELED")
    handle = Mock()
    runtime.handoff_goal_handle = handle
    node._dispatched = {runtime.handoff_id: runtime.task}
    node._waiting_deadlines = {}

    node._on_cancel(String(data=runtime.task.task_id))

    handle.cancel_goal_async.assert_called_once()
    assert runtime.handoff_cancel_requested
    assert runtime.task.state == PipelineState.HANDOFF_PENDING
    assert node._active is runtime
    node._publish_action_event.assert_called_once_with(
        event="ACTION_A_CANCEL_REQUESTED",
        task=runtime.task,
        handoff_id=runtime.handoff_id,
        detail="waiting for Low Action A cancellation result",
    )
