"""One bounded audio dispatcher shared by the pipeline and speech loopback."""

from __future__ import annotations

import math
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from flzat_high_interfaces.action import SpeechToText, TextToSpeech
from flzat_high_interfaces.msg import SpeechAudio, SpeechTranscript


class SpeechIngressNode(Node):
    def __init__(self, *, echo: bool = False) -> None:
        super().__init__("audio_loopback_node" if echo else "speech_ingress")
        self._lock = threading.RLock()
        self._busy = False
        self._pending = None
        self._serial = 0
        self._handle = None
        self._deadline = 0.0
        self._kind = "STT"
        self._audio_header = None
        for name, default in (("pending_audio_max_age_s", 2.0),
                              ("stt_timeout_s", 30.0), ("tts_timeout_s", 65.0)):
            self.declare_parameter(name, default)
            value = self.get_parameter(name).value
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self._max_age = self.get_parameter("pending_audio_max_age_s").value
        self._timeouts = {
            "STT": self.get_parameter("stt_timeout_s").value,
            "TTS": self.get_parameter("tts_timeout_s").value,
        }
        group = ReentrantCallbackGroup()
        self._stt_client = ActionClient(self, SpeechToText, "/stt_action", callback_group=group)
        self._tts_client = (
            ActionClient(self, TextToSpeech, "/tts_action", callback_group=group)
            if echo else None
        )
        # Loopback must not accidentally send test speech to the robot manager.
        self._transcript_pub = None if echo else self.create_publisher(
            SpeechTranscript, "/flzat/high_system/stt_result", 10
        )
        self._audio_sub = self.create_subscription(
            SpeechAudio, "/audio_events", self._on_audio, 1, callback_group=group
        )
        self._timer = self.create_timer(
            0.1, self._check_timeout, callback_group=group,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.get_logger().info(
            "Speech ready: VAD → STT → " + ("TTS echo." if echo else "transcript topic.")
        )

    def _on_audio(self, message: SpeechAudio) -> None:
        if not message.audio_data or message.sample_rate != 16_000:
            self.get_logger().warn("Ignored empty or non-16 kHz speech audio.")
            return
        with self._lock:
            if not (message.header.stamp.sec or message.header.stamp.nanosec):
                message.header.stamp = self.get_clock().now().to_msg()
            # Age is measured since local receipt, independent of ROS/sim clocks.
            if self._busy:
                self._pending = (message, time.monotonic())
                self.get_logger().debug("Speech busy; kept only the latest segment.")
            else:
                self._dispatch(message)

    def _dispatch(self, message: SpeechAudio) -> None:
        self._audio_header = message.header
        goal = SpeechToText.Goal()
        goal.audio_packet = message
        self._request(self._stt_client, goal, "STT")

    def _request(self, client, goal, kind: str) -> None:
        # Called under _lock, including continuations from callbacks.
        self._serial += 1
        serial = self._serial
        self._busy = True
        self._kind = kind
        self._handle = None
        self._deadline = time.monotonic() + self._timeouts[kind]
        try:
            if not client.server_is_ready():
                raise RuntimeError(f"{kind} action server unavailable")
            client.send_goal_async(goal).add_done_callback(
                lambda future: self._on_goal(serial, future)
            )
        except Exception as exc:
            self.get_logger().error(f"{kind} dispatch failed: {exc}")
            self._complete()

    def _on_goal(self, serial: int, future) -> None:
        with self._lock:
            try:
                handle = future.result()
                if not self._is_current(serial):
                    if handle.accepted:
                        self._cancel(handle)
                    return
                if not handle.accepted:
                    self.get_logger().warn(f"{self._kind} goal rejected.")
                    self._complete()
                    return
                self._handle = handle
                handle.get_result_async().add_done_callback(
                    lambda result: self._on_result(serial, result)
                )
            except Exception as exc:
                if self._is_current(serial):
                    self.get_logger().error(f"{self._kind} goal failed: {exc}")
                    self._complete()

    def _on_result(self, serial: int, future) -> None:
        with self._lock:
            if not self._is_current(serial):
                return
            try:
                response = future.result()
                if response.status != GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().warn(f"{self._kind} failed: status={response.status}.")
                elif self._kind == "STT":
                    transcript = response.result.transcript.strip()
                    if transcript and any(char.isalnum() for char in transcript):
                        if self._tts_client is not None:
                            self.get_logger().info(f"Audio transcript: '{transcript}'")
                            goal = TextToSpeech.Goal()
                            goal.text = transcript
                            self._request(self._tts_client, goal, "TTS")
                            return
                        self._transcript_pub.publish(SpeechTranscript(
                            header=self._audio_header, transcript=transcript
                        ))
                    else:
                        self.get_logger().warn("STT returned no usable transcript.")
                elif not response.result.success:
                    self.get_logger().warn("TTS did not complete successfully.")
            except Exception as exc:
                self.get_logger().error(f"{self._kind} result failed: {exc}")
            self._complete()

    def _cancel(self, handle) -> None:
        try:
            handle.cancel_goal_async()
        except Exception as exc:
            self.get_logger().warn(f"Could not cancel speech goal: {exc}")

    def _is_current(self, serial: int) -> bool:
        self._check_timeout()
        return self._busy and serial == self._serial

    def _check_timeout(self) -> None:
        with self._lock:
            if not self._busy or time.monotonic() < self._deadline:
                return
            self.get_logger().warn(f"{self._kind} timed out; discarded result and queued audio.")
            handle = self._handle
            self._pending = None
            self._complete()
            if handle is not None:
                self._cancel(handle)
            # Cancellation is cooperative: the STT server remains locked until
            # Whisper returns. It rejects new goals while still decoding.

    def _complete(self) -> None:
        self._busy = False
        self._handle = None
        self._serial += 1  # Invalidate duplicate/late callbacks.
        pending, self._pending = self._pending, None
        if pending is not None:
            message, received_at = pending
            if time.monotonic() - received_at <= self._max_age:
                self._dispatch(message)
            else:
                self.get_logger().warn("Discarded stale pending speech audio.")

    def destroy_node(self) -> None:
        self._stt_client.destroy()
        if self._tts_client is not None:
            self._tts_client.destroy()
        super().destroy_node()


def main(args=None, *, echo: bool = False) -> None:
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = SpeechIngressNode(echo=echo)
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
