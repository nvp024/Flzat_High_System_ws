#!/usr/bin/env python3
"""Pretty-print validated High System commands published as JSON strings."""

import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class CommandViewer(Node):
    """Display each final Low System command as readable JSON."""

    def __init__(self) -> None:
        super().__init__("high_command_viewer")
        self.create_subscription(
            String,
            "/flzat/high_system/command",
            self._on_command,
            10,
        )
        self.get_logger().info(
            "Waiting for JSON on /flzat/high_system/command"
        )

    def _on_command(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(f"Invalid JSON: {error}")
            print(message.data, flush=True)
            return

        print("\n========== HIGH SYSTEM COMMAND ==========", flush=True)
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        print("=========================================\n", flush=True)


def main() -> None:
    rclpy.init()
    node = CommandViewer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
