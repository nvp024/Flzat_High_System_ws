"""Opt-in live ROS check for the Step 1 unified cognitive ingress."""

import json
import os
from pathlib import Path
import threading
import time

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("FLZAT_RUN_ROS_TESTS") != "1",
    reason="set FLZAT_RUN_ROS_TESTS=1 in an isolated ROS_DOMAIN_ID",
)


def test_direct_text_reaches_one_cognitive_inference_without_router(tmp_path):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from std_msgs.msg import String
    from flzat_cognitive_pipeline.cognitive_manager import CognitiveManagerNode
    from flzat_cognitive_pipeline.domain import PipelineState
    from flzat_high_bringup.fake_servers import FakeVlmServer

    package = Path(__file__).resolve().parents[2] / "flzat_cognitive_pipeline"
    params = {
        "environment_id": "demo",
        "map_id": "map",
        "output_root": str(tmp_path),
        "prompt_directory": str(package / "prompts"),
        "safety_policy_path": str(package / "config/safety_policy.yaml"),
        "user_profile_path": str(package / "config/user_profile.yaml"),
        "enable_text_command": True,
    }
    args = ["--ros-args"]
    for name, value in params.items():
        args.extend(["-p", f"{name}:={value}"])
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=4)
    nodes = []
    thread = None
    try:
        manager = CognitiveManagerNode()
        fake = FakeVlmServer()
        fake.set_parameters(
            [
                Parameter(
                    "response_text",
                    value=json.dumps(
                        {
                            "speech": "Tokyo is the capital of Japan.",
                            "goal": None,
                            "interaction": None,
                        }
                    ),
                )
            ]
        )
        probe = rclpy.create_node("cognitive_ingress_test_probe")
        nodes.extend([manager, fake, probe])
        publisher = probe.create_publisher(
            String, "/flzat/high_system/text_command", 10
        )
        seen = []
        execute = fake._server._execute_callback

        def capture(handle):
            seen.append(handle.request)
            return execute(handle)

        fake._server._execute_callback = capture
        for node in nodes:
            executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()

        def wait_for(predicate):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if predicate():
                    return
                time.sleep(0.02)
            raise AssertionError("ROS cognitive integration timed out")

        wait_for(
            lambda: publisher.get_subscription_count() > 0
            and manager._vlm_client.server_is_ready()
        )
        publisher.publish(String(data="What is the capital of Japan?"))
        wait_for(
            lambda: manager._tasks._task_count >= 1
            and manager._tasks.get("task-0001").state
            == PipelineState.RESPONSE_ONLY
        )

        assert len(seen) == 1
        runtime = json.loads(seen[0].user_prompt.split("RUNTIME_INPUT_JSON\n", 1)[1])
        assert runtime["english_transcript"] == "What is the capital of Japan?"
        assert "router_mode" not in runtime
        assert runtime["context_availability"]["memory"]["status"] == "unavailable"
        assert runtime["context_availability"]["vision"]["status"] == "unavailable"
        assert not seen[0].image.data
        assert not list(tmp_path.rglob("resolved_routing.json"))
    finally:
        executor.shutdown(timeout_sec=5)
        if thread:
            thread.join(timeout=5)
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.try_shutdown()
