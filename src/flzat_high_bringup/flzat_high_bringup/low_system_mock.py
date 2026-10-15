"""Validate High-Level Command transport and return correlated fake status."""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class LowSystemMock(Node):
    def __init__(self) -> None:
        super().__init__("low_system_mock")
        self.declare_parameter("status_sequence", ["accepted"])
        self.declare_parameter("terminal_state", "")
        self.declare_parameter("transition_delay_s", 0.1)
        self.declare_parameter("failure_error", "configured Low System failure")
        self._status_sequence = tuple(
            str(value).strip()
            for value in self.get_parameter("status_sequence").value
        )
        terminal_state = str(self.get_parameter("terminal_state").value).strip()
        if terminal_state:
            if terminal_state not in {"succeeded", "failed"}:
                raise ValueError("terminal_state must be succeeded or failed")
            self._status_sequence = ("accepted", "running", terminal_state)
        allowed = {"accepted", "running", "succeeded", "failed"}
        if (
            not self._status_sequence
            or self._status_sequence[0] != "accepted"
            or any(value not in allowed for value in self._status_sequence)
        ):
            raise ValueError(
                "status_sequence must start with accepted and contain only "
                "accepted/running/succeeded/failed"
            )
        self._transition_delay_s = float(
            self.get_parameter("transition_delay_s").value
        )
        if self._transition_delay_s <= 0.0:
            raise ValueError("transition_delay_s must be positive")
        self._failure_error = str(self.get_parameter("failure_error").value)
        self._lock = threading.RLock()
        self._commands: dict[tuple[str, str], dict] = {}
        self._terminal: set[tuple[str, str]] = set()
        self._stop_events: dict[tuple[str, str], threading.Event] = {}
        self._workers: list[threading.Thread] = []
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._status_pub = self.create_publisher(
            String, "/flzat/low_system/status", qos
        )
        self._command_sub = self.create_subscription(
            String,
            "/flzat/high_system/command",
            self._on_command,
            qos,
        )
        self._cancel_sub = self.create_subscription(
            String,
            "/flzat/low_system/cancel",
            self._on_cancel,
            qos,
        )

    def _on_command(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            run_id = payload["run_id"]
            command_id = payload["command_id"]
            schema_version = payload["schema_version"]
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("run_id must be a nonempty string")
            if not isinstance(command_id, str) or not command_id:
                raise ValueError("command_id must be a nonempty string")
            if schema_version != "flzat.high_command.v2":
                raise ValueError("unsupported command schema")
            if payload.get("status") != "validated":
                raise ValueError("command status must be validated")
        except Exception as exc:
            self.get_logger().error(f"Rejected command payload: {exc}")
            return
        command_key = (run_id, command_id)
        with self._lock:
            if command_key in self._commands:
                self.get_logger().warn(
                    f"Ignored duplicate command {run_id}/{command_id}."
                )
                return
            self._commands[command_key] = payload
            stop_event = threading.Event()
            self._stop_events[command_key] = stop_event
        self._publish_status(run_id, command_id, "accepted")
        worker = threading.Thread(
            target=self._run_sequence,
            args=(run_id, command_id, stop_event),
            name=f"low-mock-{run_id}-{command_id}",
            daemon=True,
        )
        self._workers.append(worker)
        worker.start()

    def _run_sequence(
        self,
        run_id: str,
        command_id: str,
        stop_event: threading.Event,
    ) -> None:
        command_key = (run_id, command_id)
        for state in self._status_sequence[1:]:
            if stop_event.wait(self._transition_delay_s):
                return
            with self._lock:
                if command_key in self._terminal:
                    return
                if state in {"succeeded", "failed"}:
                    self._terminal.add(command_key)
            error = self._failure_error if state == "failed" else ""
            self._publish_status(run_id, command_id, state, error)

    def _on_cancel(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            expected = {
                "schema_version",
                "run_id",
                "command_id",
                "task_id",
                "request_id",
                "reason",
            }
            if not isinstance(payload, dict) or set(payload) != expected:
                raise ValueError("cancel fields do not match v2 schema")
            if payload["schema_version"] != "flzat.low_cancel.v2":
                raise ValueError("unsupported cancel schema")
            run_id = payload["run_id"]
            command_id = payload["command_id"]
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("cancel run_id must be nonempty")
            if not isinstance(command_id, str) or not command_id:
                raise ValueError("cancel command_id must be nonempty")
        except Exception as exc:
            self.get_logger().warn(f"Ignored invalid cancel payload: {exc}")
            return
        command_key = (run_id, command_id)
        with self._lock:
            if command_key not in self._commands or command_key in self._terminal:
                return
            self._terminal.add(command_key)
            self._stop_events[command_key].set()
        self._publish_status(run_id, command_id, "cancelled")

    def _publish_status(
        self,
        run_id: str,
        command_id: str,
        state: str,
        error: str = "",
    ) -> None:
        status = {
            "schema_version": "flzat.low_status.v2",
            "run_id": run_id,
            "command_id": command_id,
            "state": state,
            "error": error,
        }
        serialized = json.dumps(status, separators=(",", ":"), sort_keys=True)
        self._status_pub.publish(String(data=serialized))
        self.get_logger().info(f"LOW_STATUS {serialized}")

    def destroy_node(self) -> None:
        for stop_event in self._stop_events.values():
            stop_event.set()
        deadline = time.monotonic() + 2.0
        for worker in self._workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[LowSystemMock] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = LowSystemMock()
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
            worker_pool = getattr(executor, "_executor", None)
            if worker_pool is not None:
                worker_pool.shutdown(wait=True, cancel_futures=True)
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
