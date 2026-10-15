"""Small ROS client for repeatable mock/replay acceptance scenarios."""

from __future__ import annotations

import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class AcceptanceClient(Node):
    def __init__(self) -> None:
        super().__init__("high_system_acceptance_client")
        self.declare_parameter("command", "go to the water cup")
        self.declare_parameter("expected_state", "SUCCEEDED")
        self.declare_parameter("expect_command", True)
        self.declare_parameter("timeout_s", 15.0)
        self.declare_parameter("cancel_after_command", False)
        self.declare_parameter("cancel_delay_s", 0.1)
        self.declare_parameter("clarification_answer", "")
        self.command = str(self.get_parameter("command").value).strip()
        self.expected_state = str(self.get_parameter("expected_state").value).strip()
        self.expect_command = bool(self.get_parameter("expect_command").value)
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.cancel_after_command = bool(
            self.get_parameter("cancel_after_command").value
        )
        self.cancel_delay_s = float(self.get_parameter("cancel_delay_s").value)
        self.clarification_answer = str(
            self.get_parameter("clarification_answer").value
        ).strip()
        if (
            not self.command
            or not self.expected_state
            or self.timeout_s <= 0.0
            or self.cancel_delay_s < 0.0
        ):
            raise ValueError("command, expected_state and positive timeout_s are required")
        volatile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            String, "/flzat/high_system/text_command", volatile
        )
        self._cancel_publisher = self.create_publisher(
            String, "/flzat/high_system/cancel", volatile
        )
        self._status_sub = self.create_subscription(
            String, "/flzat/high_system/status", self._on_status, latched
        )
        self._command_sub = self.create_subscription(
            String, "/flzat/high_system/command", self._on_command, volatile
        )
        self._metrics_sub = self.create_subscription(
            String, "/flzat/high_system/metrics", self._on_metrics, volatile
        )
        self._health_sub = self.create_subscription(
            String, "/flzat/high_system/health", self._on_health, latched
        )
        self.run_id = ""
        self.task_id = ""
        self.request_id = ""
        self.state = ""
        self.command_payload: Optional[dict] = None
        self.metrics_payload: Optional[dict] = None
        self.health_payload: Optional[dict] = None
        self.command_received_at = 0.0
        self.cancel_sent = False
        self.clarification_sent = False
        self.published = False

    def publish(self) -> None:
        self.published = True
        self._publisher.publish(String(data=self.command))

    def _on_status(self, message: String) -> None:
        if not self.published:
            return
        try:
            value = json.loads(message.data)
            if value.get("schema_version") != "flzat.high_status.v2":
                return
            if not self.task_id and value.get("state") != "CONTEXT_RETRIEVAL":
                return
            if self.task_id and value.get("task_id") != self.task_id:
                return
            self.run_id = value.get("run_id", "")
            self.task_id = value.get("task_id", "")
            self.request_id = value.get("request_id", "")
            self.state = value.get("state", "")
        except (json.JSONDecodeError, AttributeError):
            return

    def _on_command(self, message: String) -> None:
        try:
            value = json.loads(message.data)
            if not self.task_id or value.get("task_id") == self.task_id:
                self.command_payload = value
                self.command_received_at = time.monotonic()
        except json.JSONDecodeError:
            return

    def _on_metrics(self, message: String) -> None:
        try:
            value = json.loads(message.data)
            if not self.request_id or value.get("request_id") == self.request_id:
                self.metrics_payload = value
        except json.JSONDecodeError:
            return

    def _on_health(self, message: String) -> None:
        try:
            self.health_payload = json.loads(message.data)
        except json.JSONDecodeError:
            return

    def result(self) -> dict:
        command_seen = self.command_payload is not None
        passed = (
            self.state == self.expected_state
            and command_seen == self.expect_command
            and self.health_payload is not None
            and self.metrics_payload is not None
        )
        return {
            "schema_version": "flzat.acceptance_result.v1",
            "passed": passed,
            "expected_state": self.expected_state,
            "actual_state": self.state,
            "expected_command": self.expect_command,
            "command_seen": command_seen,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "request_id": self.request_id,
            "health_ready": (
                self.health_payload.get("ready")
                if self.health_payload is not None
                else None
            ),
            "metrics_seen": self.metrics_payload is not None,
            "cancel_sent": self.cancel_sent,
            "clarification_sent": self.clarification_sent,
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AcceptanceClient()
    exit_code = 1
    try:
        wait_deadline = time.monotonic() + min(5.0, node.timeout_s / 2.0)
        while (
            node._publisher.get_subscription_count() < 1
            or node._command_sub.get_publisher_count() < 1
            or node._metrics_sub.get_publisher_count() < 1
            or node._health_sub.get_publisher_count() < 1
        ):
            if time.monotonic() >= wait_deadline:
                raise RuntimeError("text command subscriber is unavailable")
            rclpy.spin_once(node, timeout_sec=0.1)
        discovery_settle_deadline = time.monotonic() + 0.5
        while time.monotonic() < discovery_settle_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.publish()
        deadline = time.monotonic() + node.timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if (
                node.state == "WAITING_CLARIFICATION"
                and node.clarification_answer
                and not node.clarification_sent
            ):
                node._publisher.publish(String(data=node.clarification_answer))
                node.clarification_sent = True
            if (
                node.cancel_after_command
                and node.command_payload is not None
                and not node.cancel_sent
                and time.monotonic() - node.command_received_at >= node.cancel_delay_s
            ):
                node._cancel_publisher.publish(String(data=node.task_id))
                node.cancel_sent = True
            if (
                node.state == node.expected_state
                and node.health_payload is not None
                and node.metrics_payload is not None
            ):
                if (node.command_payload is not None) == node.expect_command:
                    break
        result = node.result()
        print("ACCEPTANCE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        exit_code = 0 if result["passed"] else 2
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
