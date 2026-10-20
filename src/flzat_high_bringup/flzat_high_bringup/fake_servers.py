"""Deterministic action servers matching the production ROS contracts."""

from __future__ import annotations

import time
import threading
from typing import Optional, Type

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from flzat_high_interfaces.action import RunVlm, SpeechToText, TextToSpeech


class FakeSttServer(Node):
    def __init__(self) -> None:
        super().__init__("fake_stt_server")
        self.declare_parameter("transcript", "go to the water cup")
        self._server = ActionServer(
            self,
            SpeechToText,
            "/stt_action",
            execute_callback=self._execute,
            goal_callback=lambda _goal: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

    def _execute(self, goal_handle):
        result = SpeechToText.Result()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.transcript = ""
            return result
        feedback = SpeechToText.Feedback()
        feedback.status = "fake_inference"
        goal_handle.publish_feedback(feedback)
        result.transcript = (
            self.get_parameter("transcript").get_parameter_value().string_value
        )
        goal_handle.succeed()
        return result

    def destroy_node(self) -> None:
        self._server.destroy()
        super().destroy_node()


class FakeVlmServer(Node):
    def __init__(self) -> None:
        super().__init__("fake_vlm_server")
        self.declare_parameter(
            "response_text",
            '{"intent":"fetch_object","target":{"type":"object",'
            '"name":"water_cup"},"selected_context_refs":[],"execution_strategy":'
            '"search","missing_information":[],"clarification_question":null,'
            '"speech":"I will search for the water cup."}',
        )
        self.declare_parameter("delay_s", 0.0)
        self.declare_parameter("fail", False)
        self.declare_parameter("ignore_cancel", False)
        self.declare_parameter("response_sequence", [""])
        self.declare_parameter("second_response_text", "")
        # Routing has its own fixtures so it cannot consume a main/repair response.
        self.declare_parameter(
            "routing_response_text",
            '{"mode":null,"relation_to_pending":"new_request",'
            '"needs_clarification":true,'
            '"clarification_question":"Could you clarify your request?"}',
        )
        self.declare_parameter("routing_response_sequence", [""])
        self._response_lock = threading.Lock()
        self._response_index = 0
        self._routing_response_index = 0
        self._server = ActionServer(
            self,
            RunVlm,
            "/vlm/run",
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

    @staticmethod
    def _goal(request) -> GoalResponse:
        if (
            not request.transcript.strip()
            or not request.system_prompt.strip()
            or not request.user_prompt.strip()
        ):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        result = RunVlm.Result()
        result.request_id = goal_handle.request.request_id
        ignore_cancel = (
            self.get_parameter("ignore_cancel").get_parameter_value().bool_value
        )
        delay = self.get_parameter("delay_s").get_parameter_value().double_value
        deadline = time.monotonic() + max(0.0, delay)
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested and not ignore_cancel:
                goal_handle.canceled()
                result.success = False
                result.response_text = ""
                result.error_message = "cancelled"
                return result
            time.sleep(min(0.02, deadline - time.monotonic()))
        feedback = RunVlm.Feedback()
        feedback.status = "fake_inference"
        goal_handle.publish_feedback(feedback)
        if self.get_parameter("fail").get_parameter_value().bool_value:
            goal_handle.abort()
            result.success = False
            result.response_text = ""
            result.error_message = "configured fake failure"
            return result
        result.success = True
        result.response_text = (
            self._next_routing_response()
            if result.request_id.endswith("-routing")
            else self._next_response()
        )
        result.error_message = ""
        if goal_handle.is_cancel_requested:
            goal_handle.abort()
        else:
            goal_handle.succeed()
        return result

    def _next_routing_response(self) -> str:
        """Return configured classifier fixtures, without pretending to classify."""
        sequence = tuple(
            str(value) for value in self.get_parameter("routing_response_sequence").value
            if str(value).strip()
        )
        with self._response_lock:
            if not sequence:
                return str(self.get_parameter("routing_response_text").value)
            index = min(self._routing_response_index, len(sequence) - 1)
            self._routing_response_index += 1
            return sequence[index]

    def _next_response(self) -> str:
        sequence = tuple(
            str(value)
            for value in self.get_parameter("response_sequence").value
            if str(value).strip()
        )
        with self._response_lock:
            if not sequence:
                first = (
                    self.get_parameter("response_text")
                    .get_parameter_value()
                    .string_value
                )
                second = (
                    self.get_parameter("second_response_text")
                    .get_parameter_value()
                    .string_value
                )
                response = second if self._response_index > 0 and second else first
                self._response_index += 1
                return response
            index = min(self._response_index, len(sequence) - 1)
            self._response_index += 1
            return sequence[index]

    def destroy_node(self) -> None:
        self._server.destroy()
        super().destroy_node()


class FakeTtsServer(Node):
    def __init__(self) -> None:
        super().__init__("fake_tts_server")
        self._server = ActionServer(
            self,
            TextToSpeech,
            "/tts_action",
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

    @staticmethod
    def _goal(request) -> GoalResponse:
        return (
            GoalResponse.ACCEPT
            if request.text.strip()
            else GoalResponse.REJECT
        )

    @staticmethod
    def _execute(goal_handle):
        result = TextToSpeech.Result()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            return result
        goal_handle.succeed()
        result.success = True
        return result

    def destroy_node(self) -> None:
        self._server.destroy()
        super().destroy_node()


def _spin(node_type: Type[Node], args=None) -> None:
    rclpy.init(args=args)
    node: Optional[Node] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = node_type()
        executor = MultiThreadedExecutor(num_threads=2)
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
            # Jazzy's MultiThreadedExecutor.shutdown() stops ROS work but does
            # not close its Python ThreadPoolExecutor. Without this, these
            # standalone mock processes remain alive after ros2 launch exits.
            worker_pool = getattr(executor, "_executor", None)
            if worker_pool is not None:
                try:
                    worker_pool.shutdown(wait=True, cancel_futures=True)
                except KeyboardInterrupt:
                    pass
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        try:
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


def stt_main(args=None) -> None:
    _spin(FakeSttServer, args)


def vlm_main(args=None) -> None:
    _spin(FakeVlmServer, args)


def tts_main(args=None) -> None:
    _spin(FakeTtsServer, args)
