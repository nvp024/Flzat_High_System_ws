from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from builtin_interfaces.msg import Time
from std_msgs.msg import Header, String

from flzat_high_interfaces.msg import SpeechTranscript
from flzat_cognitive_pipeline.intent_router import IntentRouterNode
from flzat_cognitive_pipeline.cognitive_manager import CognitiveManagerNode


def test_speech_input_keeps_audio_timestamp_and_text_uses_receipt_time():
    node = object.__new__(IntentRouterNode)
    node._route_utterance = Mock()
    header = Header(stamp=Time(sec=12, nanosec=345), frame_id="vad_node")
    node._on_speech(SpeechTranscript(header=header, transcript="get the book"))
    assert node._route_utterance.call_args.args == ("get the book", header)
    node.get_clock = Mock()
    node.get_clock().now().to_msg.return_value = Time(sec=99)
    node._on_utterance(String(data="get the book"))
    assert node._route_utterance.call_args.args[1].stamp.sec == 99


def test_router_preserves_header_on_routed_command():
    node = object.__new__(IntentRouterNode)
    node._router = Mock()
    node._router.route.return_value = SimpleNamespace(
        router_version="test", utterance="get the book", mode="task_command",
        confidence=0.9, language="en", urgency="normal", needs_vision=True,
        needs_task_memory=True, needs_conversation_memory=False,
        needs_user_profile=False, needs_external_knowledge=False,
        route_names=["task_command"], route_scores=[0.9], latency_ms=1.0,
        as_log_value=lambda: {},
    )
    node._routed_pub = Mock()
    node.get_logger = Mock(return_value=Mock())
    header = Header(stamp=Time(sec=12))
    node._route_utterance("get the book", header)
    assert node._routed_pub.publish.call_args.args[0].header == header


def test_frame_lookup_uses_original_input_stamp_after_slow_stt_and_routing():
    node = object.__new__(CognitiveManagerNode)
    node._frame_client = Mock()
    node._frame_client.call_async.return_value = Future()
    node.get_clock = Mock()
    runtime = SimpleNamespace(
        input_stamp=Time(sec=12), policy=SimpleNamespace(needs_vision=True),
        task=SimpleNamespace(task_id="task", request_id="request"),
    )
    node._obtain_frame(runtime)
    request = node._frame_client.call_async.call_args.args[0]
    assert request.target_stamp.sec == 12
    assert request.max_age_s == 2.0
    node.get_clock.assert_not_called()


def test_expired_frame_is_rejected_without_retrying_at_current_time():
    node = object.__new__(CognitiveManagerNode)
    node._current = Mock(return_value=object())
    node._reject = Mock()
    node._send_vlm = Mock()
    future = Future()
    future.set_result(SimpleNamespace(success=False, message="frame expired"))
    node._on_frame_result("task", "request", future)
    node._reject.assert_called_once()
    node._send_vlm.assert_not_called()
