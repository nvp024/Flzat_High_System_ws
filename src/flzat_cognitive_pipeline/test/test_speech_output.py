"""Production TTS callbacks, with deterministic transport and no speaker."""

from concurrent.futures import Future
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from action_msgs.msg import GoalStatus

from flzat_cognitive_pipeline.cognitive_manager import CognitiveManagerNode


def manager():
    node = object.__new__(CognitiveManagerNode)
    node._lock = threading.RLock()
    node._speech_serial = 0
    node._speech_busy = False
    node._speech_handle = None
    node._speech_pending = None
    node._speech_deadline = 0.0
    node._speech_timeout_s = 65.0
    node._tts_client = Mock()
    node._tts_client.send_goal_async.side_effect = lambda goal: Future()
    node.get_logger = Mock(return_value=Mock())
    return node


def complete(node, status=GoalStatus.STATUS_SUCCEEDED, success=True):
    future = Future()
    future.set_result(SimpleNamespace(status=status, result=SimpleNamespace(success=success)))
    serial = node._speech_serial
    node._on_speech_result(serial, future)
    return serial, future


def test_latest_pending_reply_is_sent_only_after_current_finishes():
    node = manager()
    node._speak("first")
    node._speak("second")
    node._speak("third")
    assert node._tts_client.send_goal_async.call_count == 1
    old_serial, old_result = complete(node)
    assert node._tts_client.send_goal_async.call_count == 2
    assert node._tts_client.send_goal_async.call_args.args[0].text == "third"
    node._on_speech_result(old_serial, old_result)
    assert node._speech_busy  # Late callbacks cannot finish the next reply.
    complete(node)
    assert not node._speech_busy


@pytest.mark.parametrize("status,success", [
    (GoalStatus.STATUS_ABORTED, False), (GoalStatus.STATUS_SUCCEEDED, False),
    (GoalStatus.STATUS_CANCELED, False),
])
def test_failed_reply_is_reported_and_slot_released(status, success):
    node = manager()
    node._speak("hello")
    complete(node, status, success)
    assert not node._speech_busy
    node.get_logger().warn.assert_called()


@pytest.mark.parametrize("failure", ["unavailable", "send", "reject", "goal", "result"])
def test_tts_transport_errors_never_leave_busy_slot(failure):
    node = manager()
    if failure == "unavailable":
        node._tts_client.server_is_ready.return_value = False
    if failure == "send":
        node._tts_client.send_goal_async.side_effect = RuntimeError("transport")
    node._speak("hello")
    future = Future()
    if failure == "reject":
        future.set_result(SimpleNamespace(accepted=False))
        node._on_speech_goal(node._speech_serial, future)
    elif failure in {"goal", "result"}:
        future.set_exception(RuntimeError("transport"))
        callback = node._on_speech_goal if failure == "goal" else node._on_speech_result
        callback(node._speech_serial, future)
    assert not node._speech_busy
    node.get_logger().warn.assert_called()


def test_expired_goal_is_canceled_even_before_watchdog_runs():
    node = manager()
    node._speak("hello")
    node._speak("queued")
    node._speech_deadline = time.monotonic() - 1
    handle = Mock(accepted=True)
    future = Future()
    future.set_result(handle)
    node._on_speech_goal(node._speech_serial, future)
    handle.cancel_goal_async.assert_called_once()
    handle.get_result_async.assert_not_called()
    assert not node._speech_busy and node._speech_pending is None


def test_stop_cancels_active_tts_and_discards_pending():
    node = manager()
    node._speak("hello")
    node._speak("queued")
    handle = Mock()
    node._speech_handle = handle
    node._stop_speech()
    handle.cancel_goal_async.assert_called_once()
    assert not node._speech_busy and node._speech_pending is None
