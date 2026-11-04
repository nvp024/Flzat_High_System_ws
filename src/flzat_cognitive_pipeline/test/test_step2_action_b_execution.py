"""Action B terminal semantics without requiring a live ROS graph."""

from types import SimpleNamespace
import threading
from unittest.mock import Mock

import pytest

from flzat_cognitive_pipeline.cognitive_manager import CognitiveManagerNode


class GoalHandle:
    def __init__(self, kind: str, timeout_s: float = 5.0):
        self.request = SimpleNamespace(
            run_id="run-test",
            task_id="task-0001",
            handoff_id="task-0001-c01",
            interaction_id=f"task-0001-c01-{kind.casefold()}",
            kind=kind,
            content_hint="Please tell the user.",
            options=[],
            timeout_s=timeout_s,
        )
        self.is_cancel_requested = False
        self.feedback = []
        self.terminal = ""

    def publish_feedback(self, value):
        self.feedback.append((value.stage, value.detail))

    def succeed(self):
        self.terminal = "SUCCEEDED"

    def abort(self):
        self.terminal = "ABORTED"

    def canceled(self):
        self.terminal = "CANCELED"


def manager():
    node = object.__new__(CognitiveManagerNode)
    node._lock = threading.RLock()
    node._interaction = None
    node._interaction_reserved = True
    node._interaction_reservation_deadline_ns = 0
    node._interaction_max_reprompts = 2
    node._dispatched = {}
    node._speak_interaction = Mock(return_value="COMPLETED")
    node._publish_action_event = Mock()
    node._publish_ready_if_available = Mock()
    node.get_logger = Mock(return_value=Mock())
    return node


@pytest.mark.parametrize("kind", ["INFORM", "REFUSE"])
def test_inform_and_refuse_speak_then_finish_without_listening(kind):
    node = manager()
    handle = GoalHandle(kind)

    result = node._execute_ask_user(handle)

    assert handle.terminal == "SUCCEEDED"
    assert result.outcome == "ANSWERED"
    assert result.answer == ""
    assert not any(stage == "listening" for stage, _ in handle.feedback)
    node._speak_interaction.assert_called_once()


def test_ask_deadline_is_timeout_not_declined():
    node = manager()
    handle = GoalHandle("ASK", timeout_s=0.000001)

    result = node._execute_ask_user(handle)

    assert handle.terminal == "SUCCEEDED"
    assert result.outcome == "TIMEOUT"
    assert result.answer == ""
    assert result.error_message == ""
