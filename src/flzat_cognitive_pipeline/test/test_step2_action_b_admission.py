"""Single-session Action B admission and observable BUSY behavior."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from rclpy.action import GoalResponse

from flzat_cognitive_pipeline.cognitive_manager import CognitiveManagerNode
from flzat_cognitive_pipeline.domain import PipelineState, TaskRecord


def request():
    return SimpleNamespace(
        run_id="run-test",
        task_id="task-0001",
        handoff_id="task-0001-c01",
        interaction_id="task-0001-c01-i01",
        kind="ASK",
        content_hint="Which bottle do you mean?",
        options=[],
        timeout_s=10.0,
    )


def manager(*, speech_busy=False, active=None):
    node = object.__new__(CognitiveManagerNode)
    task = TaskRecord(
        run_id="run-test",
        task_id="task-0001",
        request_id="task-0001-r01",
        original_transcript="Bring the bottle.",
        latest_transcript="Bring the bottle.",
        state=PipelineState.LOW_RUNNING,
        handoff_id="task-0001-c01",
    )
    node._lock = threading.RLock()
    node._dispatched = {"task-0001-c01": task}
    node._interaction = None
    node._interaction_reserved = False
    node._interaction_reservation_deadline_ns = 0
    node._speech_busy = speech_busy
    node._active = active
    node._publish_busy = Mock()
    node.get_logger = Mock(return_value=Mock())
    return node


def test_b_during_accompanying_tts_is_rejected_with_observable_busy():
    node = manager(speech_busy=True)

    response = node._on_ask_user_goal(request())

    assert response == GoalResponse.REJECT
    node._publish_busy.assert_called_once_with(
        event="action_b_rejected",
        reason="tts_active",
        interaction_id="task-0001-c01-i01",
    )
    assert not node._interaction_reserved


def test_one_valid_b_is_reserved_and_second_b_is_rejected_without_queue():
    node = manager()
    assert node._on_ask_user_goal(request()) == GoalResponse.ACCEPT
    assert node._interaction_reserved

    second = request()
    second.interaction_id = "task-0001-c01-i02"
    response = node._on_ask_user_goal(second)

    assert response == GoalResponse.REJECT
    node._publish_busy.assert_called_once_with(
        event="action_b_rejected",
        reason="user_interaction_active",
        interaction_id="task-0001-c01-i02",
    )
