"""Low-side contract mock for High Action A and reverse Action B."""

from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time
from typing import Optional

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from flzat_high_interfaces.action import AskUser, HandleIntent


_A_OUTCOMES = {"STARTED_FAST", "STARTED_PLAN", "REFUSED", "FAILED", "CANCELED"}
_A_STAGES = {"deciding", "validating", "planning", "waiting_for_user"}
_B_KINDS = {"ASK", "CONFIRM", "REFUSE", "INFORM"}


def _safe_id(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in normalized
    ):
        raise ValueError(f"{name} must be a nonempty safe identifier")
    return normalized


class LowSystemMock(Node):
    """Validate immutable handoffs and emulate only the agreed A/B contract."""

    def __init__(self, *, parameter_overrides=None) -> None:
        super().__init__(
            "low_system_mock", parameter_overrides=parameter_overrides or []
        )
        self._cb_group = ReentrantCallbackGroup()
        self.declare_parameter("result_outcome", "STARTED_PLAN")
        self.declare_parameter(
            "feedback_stages", ["deciding", "validating", "planning"]
        )
        self.declare_parameter("transition_delay_s", 0.1)
        self.declare_parameter("failure_error", "configured Low System failure")
        self.declare_parameter("ask_user_kind", "")
        self.declare_parameter("ask_user_content_hint", "Which object do you mean?")
        self.declare_parameter("ask_user_options", [])
        self.declare_parameter("ask_user_timeout_s", 30.0)
        self.declare_parameter("ask_user_retry_interval_s", 1.0)
        self.declare_parameter("ask_user_max_attempts", 10)

        self._result_outcome = str(
            self.get_parameter("result_outcome").value
        ).strip()
        if self._result_outcome not in _A_OUTCOMES:
            raise ValueError("result_outcome is not an Action A outcome")
        self._feedback_stages = tuple(
            str(value).strip()
            for value in self.get_parameter("feedback_stages").value
        )
        if any(stage not in _A_STAGES for stage in self._feedback_stages):
            raise ValueError("feedback_stages contains an unsupported stage")
        self._transition_delay_s = float(
            self.get_parameter("transition_delay_s").value
        )
        if (
            not math.isfinite(self._transition_delay_s)
            or self._transition_delay_s < 0.0
        ):
            raise ValueError("transition_delay_s must be finite and nonnegative")
        self._failure_error = str(
            self.get_parameter("failure_error").value
        ).strip()
        self._ask_kind = str(
            self.get_parameter("ask_user_kind").value
        ).strip().upper()
        if self._ask_kind and self._ask_kind not in _B_KINDS:
            raise ValueError("ask_user_kind is unsupported")
        self._ask_content = str(
            self.get_parameter("ask_user_content_hint").value
        ).strip()
        self._ask_options = tuple(
            str(value).strip()
            for value in self.get_parameter("ask_user_options").value
        )
        self._ask_timeout_s = float(
            self.get_parameter("ask_user_timeout_s").value
        )
        self._ask_retry_s = float(
            self.get_parameter("ask_user_retry_interval_s").value
        )
        self._ask_max_attempts = int(
            self.get_parameter("ask_user_max_attempts").value
        )
        if (
            not math.isfinite(self._ask_timeout_s)
            or self._ask_timeout_s <= 0.0
            or not math.isfinite(self._ask_retry_s)
            or self._ask_retry_s <= 0.0
            or self._ask_max_attempts < 1
        ):
            raise ValueError("Action B timeout/retry parameters are invalid")

        self._lock = threading.RLock()
        self._handoffs: set[tuple[str, str]] = set()
        self._busy_rejections: dict[str, str] = {}
        status_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_sub = self.create_subscription(
            String,
            "/flzat/high_system/status",
            self._on_high_status,
            status_qos,
            callback_group=self._cb_group,
        )
        self._intent_server = ActionServer(
            self,
            HandleIntent,
            "/llm/handle_intent",
            execute_callback=self._execute_intent,
            goal_callback=self._on_intent_goal,
            cancel_callback=lambda _handle: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )
        self._ask_client = ActionClient(
            self,
            AskUser,
            "/brain/ask_user",
            callback_group=self._cb_group,
        )
        self.get_logger().info("[ACTION_A] Server ready; backend=low_mock.")

    def _on_high_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if (
                payload.get("schema_version") == "flzat.high_availability.v1"
                and payload.get("state") == "BUSY"
                and payload.get("event") == "action_b_rejected"
                and payload.get("interaction_id")
            ):
                with self._lock:
                    self._busy_rejections[str(payload["interaction_id"])] = str(
                        payload.get("reason", "")
                    )
        except (json.JSONDecodeError, AttributeError, TypeError):
            return

    def _on_intent_goal(self, request: HandleIntent.Goal) -> GoalResponse:
        try:
            run_id = _safe_id(request.run_id, "run_id")
            _safe_id(request.task_id, "task_id")
            handoff_id = _safe_id(request.handoff_id, "handoff_id")
            path = Path(str(request.json_path))
            if not path.is_absolute():
                raise ValueError("json_path must be absolute")
            key = (run_id, handoff_id)
            with self._lock:
                if key in self._handoffs:
                    raise ValueError("duplicate handoff_id")
                self._handoffs.add(key)
            return GoalResponse.ACCEPT
        except Exception as exc:
            self.get_logger().warn(f"[ACTION_A] Rejected goal: {exc}")
            return GoalResponse.REJECT

    def _execute_intent(self, goal_handle) -> HandleIntent.Result:
        request = goal_handle.request
        try:
            payload = self._read_handoff(request)
            self.get_logger().debug(
                f"[ACTION_A] accepted {request.handoff_id} from {request.json_path}"
            )
            for stage in self._feedback_stages:
                if self._wait_or_cancel(goal_handle, self._transition_delay_s):
                    return self._cancel_a(goal_handle)
                self._feedback_a(goal_handle, stage, f"mock stage: {stage}")

            if self._ask_kind:
                self._feedback_a(
                    goal_handle,
                    "waiting_for_user",
                    "Low mock requested user interaction through Action B",
                )
                interaction = self._call_ask_user(goal_handle, request)
                if interaction is None:
                    return self._failed_a(
                        goal_handle,
                        "Action B could not be established or completed",
                    )
                self.get_logger().debug(
                    "[ACTION_B] "
                    f"{interaction.outcome or 'ERROR'} answer={interaction.answer!r} "
                    f"error={interaction.error_message!r}"
                )
                if interaction.error_message:
                    return self._failed_a(
                        goal_handle,
                        "Action B failed: " + interaction.error_message,
                    )

            del payload
            if self._result_outcome in {"STARTED_FAST", "STARTED_PLAN", "REFUSED"}:
                goal_handle.succeed()
            elif self._result_outcome == "CANCELED":
                goal_handle.canceled()
            else:
                goal_handle.abort()
            result = HandleIntent.Result()
            result.outcome = self._result_outcome
            result.detail = (
                self._failure_error
                if self._result_outcome == "FAILED"
                else "configured Low mock result"
            )
            return result
        except Exception as exc:
            return self._failed_a(goal_handle, f"handoff validation failed: {exc}")

    @staticmethod
    def _read_handoff(request: HandleIntent.Goal) -> dict:
        path = Path(str(request.json_path)).resolve(strict=True)
        # v2 keeps run/task IDs in Goal A and the immutable directory layout,
        # rather than repeating them in the cognitive payload.
        if (
            path.parent.parent.name != request.run_id
            or path.parent.name != request.task_id
            or path.name != f"{request.handoff_id}.json"
        ):
            raise ValueError("artifact path does not match Action A run/task/handoff IDs")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("handoff artifact must contain a JSON object")
        expected = {
            "schema_version": "flzat.high_handoff.v2",
            "handoff_id": request.handoff_id,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise ValueError(f"artifact {field} does not match Action A goal")
        user_request = payload.get("user_request")
        goal = payload.get("goal")
        context = payload.get("context")
        if not isinstance(user_request, dict) or not isinstance(goal, dict):
            raise ValueError("handoff requires user_request and goal objects")
        if not isinstance(user_request.get("original"), str) or not user_request["original"].strip():
            raise ValueError("handoff requires the original user request")
        if not isinstance(user_request.get("clarifications"), list):
            raise ValueError("handoff requires clarification history")
        if not isinstance(goal.get("description"), str) or not goal["description"].strip():
            raise ValueError("handoff requires a goal description")
        for field in ("constraints", "missing_information"):
            if not isinstance(goal.get(field), list) or not all(
                isinstance(value, str) for value in goal[field]
            ):
                raise ValueError(f"goal.{field} must be an array of strings")
        if not isinstance(context, dict) or context.get("source") != "environment_memory":
            raise ValueError("handoff requires environment-memory context")
        if context.get("current_positions_confirmed") is not False:
            raise ValueError("stored memory must not claim confirmed current positions")
        if not isinstance(payload.get("entities"), list):
            raise ValueError("handoff requires an entities array")
        return payload

    def _call_ask_user(self, intent_handle, request: HandleIntent.Goal):
        deadline = time.monotonic() + self._ask_timeout_s
        interaction_id = f"{request.handoff_id}-i01"
        for attempt in range(1, self._ask_max_attempts + 1):
            if intent_handle.is_cancel_requested:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            if not self._ask_client.wait_for_server(
                timeout_sec=min(remaining, 0.5)
            ):
                if not self._retry_wait(intent_handle, deadline, attempt):
                    return None
                continue
            goal = AskUser.Goal()
            goal.run_id = request.run_id
            goal.task_id = request.task_id
            goal.handoff_id = request.handoff_id
            goal.interaction_id = interaction_id
            goal.kind = self._ask_kind
            goal.content_hint = self._ask_content
            goal.options = list(self._ask_options)
            goal.timeout_s = float(max(0.001, deadline - time.monotonic()))
            send_done = threading.Event()
            send_future = self._ask_client.send_goal_async(goal)
            send_future.add_done_callback(lambda _future: send_done.set())
            if not self._wait_event(send_done, intent_handle, deadline):
                return None
            ask_handle = send_future.result()
            if not ask_handle.accepted:
                reason = self._wait_for_busy_rejection(
                    interaction_id,
                    min(deadline, time.monotonic() + 0.25),
                )
                if not reason:
                    self.get_logger().warn(
                        "[ACTION_B] rejected without a matching High BUSY event; "
                        "not retrying"
                    )
                    return None
                self.get_logger().debug(
                    f"[ACTION_B] attempt {attempt} rejected as {reason}; "
                    "retrying without queue"
                )
                if not self._retry_wait(intent_handle, deadline, attempt):
                    return None
                continue
            result_done = threading.Event()
            result_future = ask_handle.get_result_async()
            result_future.add_done_callback(lambda _future: result_done.set())
            if not self._wait_event(result_done, intent_handle, deadline):
                ask_handle.cancel_goal_async()
                return None
            return result_future.result().result
        return None

    def _wait_for_busy_rejection(self, interaction_id: str, deadline: float) -> str:
        while time.monotonic() < deadline:
            with self._lock:
                reason = self._busy_rejections.pop(interaction_id, "")
            if reason:
                return reason
            time.sleep(0.01)
        return ""

    def _retry_wait(self, intent_handle, deadline: float, attempt: int) -> bool:
        if attempt >= self._ask_max_attempts:
            return False
        wait_s = min(
            self._ask_retry_s,
            max(0.0, deadline - time.monotonic()),
        )
        return not self._wait_or_cancel(intent_handle, wait_s)

    @staticmethod
    def _wait_event(event: threading.Event, intent_handle, deadline: float) -> bool:
        while not event.wait(timeout=0.05):
            if intent_handle.is_cancel_requested or time.monotonic() >= deadline:
                return False
        return True

    @staticmethod
    def _wait_or_cancel(goal_handle, duration: float) -> bool:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return True
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        return goal_handle.is_cancel_requested

    @staticmethod
    def _feedback_a(goal_handle, stage: str, detail: str) -> None:
        feedback = HandleIntent.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _cancel_a(goal_handle) -> HandleIntent.Result:
        goal_handle.canceled()
        result = HandleIntent.Result()
        result.outcome = "CANCELED"
        result.detail = "Action A canceled by High"
        return result

    @staticmethod
    def _failed_a(goal_handle, detail: str) -> HandleIntent.Result:
        goal_handle.abort()
        result = HandleIntent.Result()
        result.outcome = "FAILED"
        result.detail = detail
        return result

    def destroy_node(self) -> None:
        self._ask_client.destroy()
        self._intent_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[LowSystemMock] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = LowSystemMock()
        executor = MultiThreadedExecutor(num_threads=4)
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
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        try:
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
