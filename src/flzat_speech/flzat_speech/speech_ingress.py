"""Bridge VAD audio segments through STT to the canonical transcript topic."""

from __future__ import annotations

import threading
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from flzat_high_interfaces.action import SpeechToText
from flzat_high_interfaces.msg import SpeechAudio


class SpeechIngressNode(Node):
    """Own STT dispatch so downstream consumers receive normalized text only."""

    def __init__(self) -> None:
        super().__init__("speech_ingress")
        self._cb_group = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self._busy = False
        self._pending: SpeechAudio | None = None
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._transcript_pub = self.create_publisher(
            String, "/flzat/high_system/stt_result", qos
        )
        self._audio_sub = self.create_subscription(
            SpeechAudio,
            "/audio_events",
            self._on_audio,
            qos,
            callback_group=self._cb_group,
        )
        self._stt_client = ActionClient(
            self,
            SpeechToText,
            "/stt_action",
            callback_group=self._cb_group,
        )
        self.get_logger().info(
            "Speech ingress ready: /audio_events → /stt_action → "
            "/flzat/high_system/stt_result."
        )

    def _on_audio(self, message: SpeechAudio) -> None:
        if not message.audio_data or message.sample_rate != 16_000:
            self.get_logger().warn("Ignored empty or non-16 kHz speech audio.")
            return
        with self._lock:
            if self._busy:
                self._pending = message
                self.get_logger().warn(
                    "STT busy; retained only the newest speech segment."
                )
                return
            self._busy = True
        self._dispatch(message)

    def _dispatch(self, message: SpeechAudio) -> None:
        if not self._stt_client.server_is_ready():
            self.get_logger().error("STT action server is unavailable.")
            self._complete_and_continue()
            return
        goal = SpeechToText.Goal()
        goal.audio_packet = message
        future = self._stt_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal)

    def _on_goal(self, future) -> None:
        try:
            handle = future.result()
            if not handle.accepted:
                self.get_logger().warn("STT goal was rejected.")
                self._complete_and_continue()
                return
            result_future = handle.get_result_async()
            result_future.add_done_callback(self._on_result)
        except Exception as exc:
            self.get_logger().error(f"STT dispatch failed: {exc}")
            self._complete_and_continue()

    def _on_result(self, future) -> None:
        try:
            transcript = future.result().result.transcript.strip()
            if transcript and any(character.isalnum() for character in transcript):
                self._transcript_pub.publish(String(data=transcript))
                self.get_logger().debug(f"Published STT result: '{transcript}'")
            else:
                self.get_logger().warn("STT returned no usable transcript.")
        except Exception as exc:
            self.get_logger().error(f"STT result failed: {exc}")
        finally:
            self._complete_and_continue()

    def _complete_and_continue(self) -> None:
        next_message = None
        with self._lock:
            next_message = self._pending
            self._pending = None
            if next_message is None:
                self._busy = False
        if next_message is not None:
            self._dispatch(next_message)

    def destroy_node(self) -> None:
        self._stt_client.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[SpeechIngressNode] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = SpeechIngressNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=2.0)
            except (KeyboardInterrupt, ExternalShutdownException):
                pass
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
