from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from builtin_interfaces.msg import Time
from flzat_cognitive_pipeline.cognitive_manager import CognitiveManagerNode


def test_frame_lookup_uses_original_input_stamp_after_slow_stt_and_routing():
    node = object.__new__(CognitiveManagerNode)
    node._frame_client = Mock()
    node._frame_client.call_async.return_value = Future()
    node.get_clock = Mock()
    runtime = SimpleNamespace(
        input_stamp=Time(sec=12),
        task=SimpleNamespace(task_id="task", request_id="request"),
    )
    node._obtain_frame(runtime)
    request = node._frame_client.call_async.call_args.args[0]
    assert request.target_stamp.sec == 12
    assert request.max_age_s == 2.0
    node.get_clock.assert_not_called()


def test_expired_frame_degrades_to_text_only_without_retrying_current_time():
    node = object.__new__(CognitiveManagerNode)
    runtime = SimpleNamespace(
        context_availability={},
        task=SimpleNamespace(request_id="request"),
    )
    node._current = Mock(return_value=runtime)
    node._telemetry = Mock()
    node._compose_prompt_and_send = Mock()
    future = Future()
    future.set_result(
        SimpleNamespace(
            success=False,
            message="frame expired",
            frame=SimpleNamespace(data=b""),
        )
    )
    node._on_frame_result("task", "request", future)
    assert runtime.context_availability["vision"]["status"] == "unavailable"
    node._compose_prompt_and_send.assert_called_once()
    assert not node._compose_prompt_and_send.call_args.args[1].data
