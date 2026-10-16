"""Serve the canonical Qwen3-VL cognitive action."""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Optional

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

from flzat_high_interfaces.action import RunVlm

from std_msgs.msg import String

from flzat_vlm.backends import available_backends, create_backend
from flzat_vlm.backends.base import (
    BackendConfig,
    GenerationCancelled,
    GenerationRequest,
)
from flzat_vlm.cancellation import GoalCancellationRegistry
from flzat_vlm.image_io import decode_compressed_images
from flzat_vlm.priority_broker import (
    InferenceTicket,
    PriorityInferenceBroker,
    TicketState,
)


VOICE_PRIORITY = 30


class VlmNode(Node):
    """One Qwen3-VL backend serving precomposed cognitive requests."""

    def __init__(self) -> None:
        super().__init__("vlm_node")
        self._callback_group = ReentrantCallbackGroup()
        self._broker = PriorityInferenceBroker()
        self._cancel_tokens = GoalCancellationRegistry()
        self._declare_parameters()

        backend_name = self._string_parameter("backend")
        model_id = self._string_parameter("model_id")
        self._max_new_tokens = self._integer_parameter("max_new_tokens")
        self._max_input_bytes = self._integer_parameter("max_input_bytes")
        self._max_image_pixels = self._integer_parameter("max_image_pixels")
        if self._max_new_tokens < 1:
            raise ValueError("VLM token limit must be at least one")
        if self._max_input_bytes < 1 or self._max_image_pixels < 1:
            raise ValueError("image input limits must be positive")

        raw_response_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._raw_response_pub = self.create_publisher(
            String,
            "/vlm/raw_response",
            raw_response_qos,
        )
        self.get_logger().info("Cognitive requests require precomposed prompts.")

        config = BackendConfig(
            model_id=model_id,
            model_revision=self._string_parameter("model_revision"),
            device=self._string_parameter("device"),
            dtype=self._string_parameter("dtype"),
            quantization=self._string_parameter("quantization"),
            trust_remote_code=self._boolean_parameter("trust_remote_code"),
            local_files_only=self._boolean_parameter("local_files_only"),
            min_image_pixels=self._integer_parameter("qwen_min_image_pixels"),
            max_image_pixels=self._integer_parameter("qwen_max_image_pixels"),
        )
        self.get_logger().info(
            f"Loading VLM: backend={backend_name}, model={model_id}, "
            f"device={config.device}, dtype={config.dtype}, "
            f"quantization={config.quantization}."
        )
        started = time.perf_counter()
        try:
            self._backend = create_backend(backend_name, config)
            self._backend.load()
        except Exception as exc:
            available = ", ".join(available_backends())
            self.get_logger().fatal(
                f"VLM initialization failed: {exc}. Available backends: {available}."
            )
            raise RuntimeError(f"VLM initialization failed: {exc}") from exc
        self.get_logger().info(
            f"VLM ready: backend={self._backend.name}, "
            f"device={self._backend.device_description}, "
            f"initialization={time.perf_counter() - started:.2f} s."
        )

        self._run_action_server = ActionServer(
            self,
            RunVlm,
            "/vlm/run",
            execute_callback=self._execute_run,
            goal_callback=self._on_run_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._callback_group,
        )
        self.get_logger().info("VLM action ready on /vlm/run.")

    def _declare_parameters(self) -> None:
        self.declare_parameter("backend", "qwen3_vl")
        self.declare_parameter(
            "model_id", "Qwen/Qwen3-VL-2B-Instruct"
        )
        self.declare_parameter("model_revision", "main")
        self.declare_parameter("device", "auto")
        self.declare_parameter("dtype", "auto")
        self.declare_parameter("quantization", "none")
        self.declare_parameter("max_new_tokens", 256)
        self.declare_parameter("max_input_bytes", 10_000_000)
        self.declare_parameter("max_image_pixels", 16_000_000)
        self.declare_parameter("trust_remote_code", False)
        self.declare_parameter("local_files_only", False)
        self.declare_parameter("qwen_min_image_pixels", 256 * 256)
        self.declare_parameter("qwen_max_image_pixels", 768 * 768)

    def _on_run_goal(self, goal_request: RunVlm.Goal) -> GoalResponse:
        if not goal_request.transcript.strip():
            self.get_logger().warn("Rejected VLM goal without a transcript.")
            return GoalResponse.REJECT
        prompt_values = (
            goal_request.system_prompt.strip(),
            goal_request.user_prompt.strip(),
            goal_request.prompt_version.strip(),
            goal_request.prompt_hash.strip(),
            goal_request.request_id.strip(),
        )
        if not all(prompt_values):
            self.get_logger().warn("Rejected request without complete composed prompt metadata.")
            return GoalResponse.REJECT
        expected_hash = hashlib.sha256(
            (prompt_values[0] + "\0" + prompt_values[1]).encode("utf-8")
        ).hexdigest()
        if prompt_values[3] != expected_hash:
            self.get_logger().warn("Rejected precomposed prompt hash mismatch.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        if self._cancel_tokens.request_cancel(goal_handle):
            self.get_logger().info("Cooperative VLM cancellation was signalled.")
        return CancelResponse.ACCEPT

    def _execute_run(self, goal_handle):
        result = RunVlm.Result()
        result.request_id = goal_handle.request.request_id
        images = ()
        token = threading.Event()
        ticket = None
        token_registered = False
        inference_started = 0.0
        try:
            self._cancel_tokens.register(goal_handle, token)
            token_registered = True
            ticket = self._broker.submit(
                self._goal_key(goal_handle),
                VOICE_PRIORITY,
                token,
            )
            self._publish_run_feedback(goal_handle, "queued")
            if not self._wait_for_turn(goal_handle, ticket):
                return self._finish_run_without_inference(goal_handle, result, ticket)

            request = goal_handle.request
            system_prompt = request.system_prompt
            user_prompt = request.user_prompt
            if request.image.data:
                self._publish_run_feedback(goal_handle, "decoding_image")
                images = decode_compressed_images(
                    [request.image], self._max_input_bytes, self._max_image_pixels
                )
            else:
                self._publish_run_feedback(goal_handle, "text_only")
            self._raise_if_cancelled(goal_handle, ticket)
            self._publish_run_feedback(goal_handle, "inference")
            inference_started = time.perf_counter()
            raw_response = self._generate(
                images,
                system_prompt,
                user_prompt,
                self._max_new_tokens,
                token,
            )
            self._raise_if_cancelled(goal_handle, ticket)
            self._publish_raw_response(raw_response)
            result.success = True
            result.response_text = raw_response
            result.error_message = ""
            goal_handle.succeed()
            self._publish_run_feedback(goal_handle, "complete")
            self.get_logger().info(
                f"VLM response ({time.perf_counter() - inference_started:.2f} s): "
                f"'{raw_response[:500]}'"
            )
            return result
        except GenerationCancelled:
            return self._cancel_or_preempt_run(goal_handle, result, ticket)
        except Exception as exc:
            if ticket is not None and token.is_set():
                return self._cancel_or_preempt_run(goal_handle, result, ticket)
            self.get_logger().error(f"VLM request failed: {exc}")
            goal_handle.abort()
            result.success = False
            result.response_text = ""
            result.error_message = str(exc)
            return result
        finally:
            for image in images:
                image.close()
            if ticket is not None:
                self._broker.complete(ticket)
            if token_registered:
                self._cancel_tokens.unregister(goal_handle, token)

    def _wait_for_turn(self, goal_handle, ticket: InferenceTicket) -> bool:
        while not ticket.ready_event.wait(0.1):
            if goal_handle.is_cancel_requested:
                self._broker.cancel(ticket, "cancelled by action client")
        return (
            ticket.state == TicketState.ACTIVE
            and not goal_handle.is_cancel_requested
        )

    def _raise_if_cancelled(self, goal_handle, ticket: InferenceTicket) -> None:
        if goal_handle.is_cancel_requested:
            self._broker.cancel(ticket, "cancelled by action client")
        if ticket.cancel_event.is_set():
            raise GenerationCancelled(ticket.reason or "generation cancelled")

    def _generate(self, images, system_prompt, user_prompt, token_limit, token):
        response = self._backend.generate(
            GenerationRequest(
                images=images,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=token_limit,
                cancel_event=token,
            )
        ).strip()
        if not response:
            raise RuntimeError("VLM backend returned an empty response")
        return response

    def _finish_run_without_inference(self, goal_handle, result, ticket):
        if goal_handle.is_cancel_requested or ticket.state == TicketState.CANCELLED:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result.success = False
        result.response_text = ""
        result.error_message = ticket.reason or "VLM request was not scheduled"
        return result

    def _cancel_or_preempt_run(self, goal_handle, result, ticket):
        client_cancel = goal_handle.is_cancel_requested
        if client_cancel:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result.success = False
        result.response_text = ""
        result.error_message = (
            "VLM request cancelled by client"
            if client_cancel
            else (ticket.reason or "VLM request preempted")
        )
        return result

    def _publish_raw_response(self, response: str) -> None:
        self._raw_response_pub.publish(String(data=response))

    @staticmethod
    def _goal_key(goal_handle) -> str:
        goal_id = getattr(goal_handle, "goal_id", None)
        uuid = getattr(goal_id, "uuid", None)
        return str(id(goal_handle)) if uuid is None else bytes(uuid).hex()

    @staticmethod
    def _publish_run_feedback(goal_handle, status: str) -> None:
        feedback = RunVlm.Feedback()
        feedback.status = status
        goal_handle.publish_feedback(feedback)

    def _integer_parameter(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _string_parameter(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _boolean_parameter(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def destroy_node(self) -> None:
        if hasattr(self, "_run_action_server"):
            self._run_action_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: Optional[VlmNode] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = VlmNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("VlmNode shutting down …")
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
