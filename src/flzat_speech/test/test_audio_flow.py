"""Exercise speech boundaries without microphone, Whisper weights or ROS graph."""

from concurrent.futures import Future
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from rclpy.action import GoalResponse
from rclpy.node import Node

from flzat_high_interfaces.msg import SpeechAudio
from flzat_speech.speech_ingress import SpeechIngressNode
from flzat_speech.vad_node import _MicReader, _VADStream, VadNode, CHUNK_SAMPLES
from flzat_speech.stt_node import SttNode
from flzat_speech.tts_node import TtsNode


def audio(value=1):
    return SpeechAudio(header=Header(stamp=Time(sec=12)), audio_data=[value], sample_rate=16000)


def dispatcher(echo=False):
    node = object.__new__(SpeechIngressNode)
    node._lock = threading.RLock()
    node._busy = False
    node._pending = None
    node._serial = 0
    node._handle = None
    node._deadline = 0.0
    node._kind = "STT"
    node._max_age = 2.0
    node._timeouts = {"STT": 30.0, "TTS": 60.0}
    node.get_logger = Mock(return_value=Mock())
    node._stt_client = Mock()
    node._stt_client.send_goal_async.side_effect = lambda goal: Future()
    node._tts_client = Mock() if echo else None
    if echo:
        node._tts_client.send_goal_async.side_effect = lambda goal: Future()
    node._transcript_pub = None if echo else Mock()
    return node


def response(node, *, status=GoalStatus.STATUS_SUCCEEDED, text="go to the kitchen", success=True):
    future = Future()
    future.set_result(SimpleNamespace(
        status=status, result=SimpleNamespace(transcript=text, success=success)
    ))
    node._on_result(node._serial, future)
    return future


@pytest.mark.parametrize("status", [GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED])
def test_failed_action_never_publishes_nonempty_text(status):
    node = dispatcher()
    node._on_audio(audio())
    response(node, status=status)
    node._transcript_pub.publish.assert_not_called()
    assert not node._busy


def test_success_publishes_once_and_ignores_duplicate_result():
    node = dispatcher()
    node._on_audio(audio())
    serial = node._serial
    future = response(node)
    node._on_result(serial, future)
    node._transcript_pub.publish.assert_called_once()
    result = node._transcript_pub.publish.call_args.args[0]
    assert result.transcript == "go to the kitchen"
    assert result.header.stamp.sec == 12


@pytest.mark.parametrize("failure", ["unavailable", "exception"])
def test_dispatch_failure_releases_busy(failure):
    node = dispatcher()
    if failure == "unavailable":
        node._stt_client.server_is_ready.return_value = False
    else:
        node._stt_client.send_goal_async.side_effect = RuntimeError("transport")
    node._on_audio(audio())
    assert not node._busy
    node._transcript_pub.publish.assert_not_called()


def test_only_latest_pending_audio_runs():
    node = dispatcher()
    node._on_audio(audio(1))
    node._on_audio(audio(2))
    node._on_audio(audio(3))
    assert node._stt_client.send_goal_async.call_count == 1
    response(node)
    assert node._stt_client.send_goal_async.call_count == 2
    assert list(node._stt_client.send_goal_async.call_args.args[0].audio_packet.audio_data) == [3]


@pytest.mark.parametrize("echo", [False, True])
def test_stale_audio_dropped_in_main_and_loopback(echo):
    node = dispatcher(echo)
    node._on_audio(audio())
    node._on_audio(audio(2))
    if echo:
        response(node)  # STT → TTS; still busy and keeps pending segment.
        assert node._kind == "TTS" and node._busy
    node._pending = (audio(2), time.monotonic() - 3.0)
    response(node)
    assert node._stt_client.send_goal_async.call_count == 1
    assert not node._busy
    if echo:
        assert node._transcript_pub is None


def test_timeout_cancels_and_discards_pending_and_late_result():
    node = dispatcher()
    node._on_audio(audio())
    serial = node._serial
    handle = Mock()
    node._handle = handle
    node._on_audio(audio(2))
    node._deadline = time.monotonic() - 1.0
    node._check_timeout()
    assert not node._busy and node._pending is None
    handle.cancel_goal_async.assert_called_once()
    node._on_audio(audio(3))
    future = Future()
    future.set_result(SimpleNamespace(
        status=GoalStatus.STATUS_SUCCEEDED, result=SimpleNamespace(transcript="old command")
    ))
    node._on_result(serial, future)
    assert node._busy  # Old callback cannot finish the new request.
    node._transcript_pub.publish.assert_not_called()


def test_goal_accepted_after_deadline_is_canceled_without_timer_tick():
    node = dispatcher()
    node._on_audio(audio())
    serial = node._serial
    node._deadline = time.monotonic() - 1.0
    future = Future()
    handle = Mock(accepted=True)
    future.set_result(handle)
    node._on_goal(serial, future)
    handle.cancel_goal_async.assert_called_once()
    handle.get_result_async.assert_not_called()
    assert not node._busy


@pytest.mark.parametrize("failure", ["rejected", "goal_exception", "result_exception"])
def test_action_callback_failures_release_busy(failure):
    node = dispatcher()
    node._on_audio(audio())
    future = Future()
    if failure == "rejected":
        future.set_result(SimpleNamespace(accepted=False))
    else:
        future.set_exception(RuntimeError("transport"))
    callback = node._on_result if failure == "result_exception" else node._on_goal
    callback(node._serial, future)
    assert not node._busy


def mic():
    return _MicReader(16000, CHUNK_SAMPLES, "unused", 2)


def fill(reader, count=1, status=False):
    for _ in range(count):
        reader._callback(np.zeros((CHUNK_SAMPLES, 1)), CHUNK_SAMPLES, None, status)


@pytest.mark.parametrize("failure", ["overflow", "status", "old", "short"])
def test_mic_discontinuity_discards_entire_backlog(failure):
    reader = mic()
    fill(reader)
    if failure == "overflow":
        fill(reader, 2)
    elif failure == "status":
        fill(reader, status=True)
    elif failure == "short":
        reader._callback(np.zeros((2, 1)), 2, None, False)
    else:
        reader._queue[0] = (time.monotonic() - 1.0, np.zeros(CHUNK_SAMPLES))
    chunk, dropped = reader.pop()
    assert chunk is None and dropped > 0
    fill(reader)
    chunk, dropped = reader.pop()
    assert chunk is not None and dropped == 0


def vad():
    node = object.__new__(VadNode)
    node._mic = mic()
    node._vad_stream = Mock()
    node._vad_stream.process.return_value = None
    node._tts_active = False
    node._resume_vad_at = 0.0
    node._tts_resume_delay_s = 0.8
    node.get_logger = Mock(return_value=Mock())
    return node


def test_mute_cooldown_and_resume_clear_audio_backlog():
    node = vad()
    fill(node._mic)
    node._on_tts_active(SimpleNamespace(data=True))
    assert not node._mic._queue
    fill(node._mic)
    node._timer_callback()
    assert not node._mic._queue
    node._on_tts_active(SimpleNamespace(data=False))
    fill(node._mic)
    node._timer_callback()  # During cooldown.
    assert not node._mic._queue
    fill(node._mic)
    node._resume_vad_at = time.monotonic() - 1.0
    node._timer_callback()  # Stalled timer at end of cooldown.
    node._vad_stream.process.assert_not_called()
    fill(node._mic)
    node._timer_callback()
    node._vad_stream.process.assert_called_once()


def test_gap_resets_vad_before_any_further_audio():
    node = vad()
    fill(node._mic, 3)
    node._timer_callback()
    node._vad_stream.discard_until_silence.assert_called_once()
    node._vad_stream.process.assert_not_called()


class VoiceModel:
    confidence = 1.0

    def reset_states(self):
        pass

    def __call__(self, tensor, rate):
        return SimpleNamespace(item=lambda: self.confidence)


def test_long_utterance_and_its_tail_are_discarded_until_silence():
    model = VoiceModel()
    stream = _VADStream(model, 2.0, 500)
    chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    events = [stream.process(chunk) for _ in range(100)]
    assert [e[0] for e in events if e] == ["start", "max_duration"]
    assert next(e for e in events if e and e[0] == "max_duration")[1] == []
    assert stream._discard_until_silence
    model.confidence = 0.0
    for _ in range(stream._silence_chunks):
        assert stream.process(chunk) is None
    assert not stream._discard_until_silence
    model.confidence = 1.0
    for _ in range(10):
        stream.process(chunk)
    model.confidence = 0.0
    ends = [stream.process(chunk) for _ in range(stream._silence_chunks)]
    assert ends[-1][0] == "end" and ends[-1][1]


def test_vad_never_publishes_max_duration_event_even_with_frames():
    node = vad()
    fill(node._mic)
    node._vad_stream.process.return_value = ("max_duration", [np.ones(CHUNK_SAMPLES)])
    node._publish_segment = Mock()
    node._timer_callback()
    node._publish_segment.assert_not_called()


def test_discontinuity_drops_tail_instead_of_restarting_mid_command():
    model = VoiceModel()
    stream = _VADStream(model, 30.0, 500)
    chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    stream.process(chunk)
    stream.discard_until_silence()
    assert all(stream.process(chunk) is None for _ in range(50))
    assert not stream._speech_frames


def test_cancel_keeps_stt_model_locked_until_inference_returns(monkeypatch):
    import flzat_speech.stt_node as module

    entered, release = threading.Event(), threading.Event()
    node = object.__new__(SttNode)
    node._goal_lock = threading.Lock()
    node._goal_active = True
    node._inference_done = threading.Event()
    node._inference_done.set()
    node._shutting_down = False
    node._expected_sample_rate = 16000
    node._max_audio_duration_s = 30.0
    node._language, node._beam_size, node._initial_prompt = "en", 5, ""
    node._max_decode_tokens = 64
    node._max_no_speech_probability, node._min_avg_logprob = 0.65, -1.0
    node._stt_model = object()
    node.get_logger = Mock(return_value=Mock())
    handle = Mock(is_cancel_requested=False, request=SimpleNamespace(audio_packet=audio()))

    def inference(*args):
        entered.set()
        assert release.wait(2.0)
        return {"segments": []}

    monkeypatch.setattr(module, "_transcribe_bounded", inference)
    worker = threading.Thread(target=node._execute_callback, args=(handle,))
    worker.start()
    try:
        assert entered.wait(2.0)
        handle.is_cancel_requested = True
        assert node._goal_callback(SimpleNamespace(audio_packet=audio())) == GoalResponse.REJECT
    finally:
        release.set()
        worker.join(2.0)
    assert not worker.is_alive() and not node._goal_active
    handle.canceled.assert_called_once()
    handle.succeed.assert_not_called()


@pytest.mark.parametrize("node_class", [SttNode, TtsNode])
def test_shutdown_waits_before_destroying_action_handles(monkeypatch, node_class):
    calls = []
    node = object.__new__(node_class)
    monkeypatch.setattr(Node, "destroy_node", lambda self: calls.append("node_destroy"))
    node._action_server = Mock()
    node._action_server.destroy.side_effect = lambda: calls.append("action_destroy")
    if node_class is SttNode:
        node._inference_done = Mock()
        node._inference_done.wait.side_effect = lambda: calls.append("wait_model")
    else:
        node._stop_event = threading.Event()
        node._state_lock = threading.Lock()
        node._active_engine = Mock()
        node._active_engine.request_stop.side_effect = lambda: calls.append("stop_engine")
        node._execution_done = Mock()
        node._execution_done.wait.side_effect = lambda: calls.append("wait_callback")
        node._speech_worker = Mock()
        node._speech_worker.join.side_effect = lambda: calls.append("wait_worker")
    node.destroy_node()
    assert node._shutting_down
    assert calls[-2:] == ["action_destroy", "node_destroy"]
    assert calls[:-2] == (
        ["wait_model"] if node_class is SttNode
        else ["stop_engine", "wait_callback", "wait_worker"]
    )
