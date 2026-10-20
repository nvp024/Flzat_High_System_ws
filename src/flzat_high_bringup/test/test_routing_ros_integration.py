"""Opt-in live ROS action/topic check with the production manager and fake VLM."""

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


def test_routing_continuation_and_topic_switch_over_ros(tmp_path):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from std_msgs.msg import String
    from flzat_high_interfaces.msg import RoutedCommand
    from flzat_cognitive_pipeline.cognitive_manager import CognitiveManagerNode
    from flzat_cognitive_pipeline.domain import PipelineState
    from flzat_high_bringup.fake_servers import FakeVlmServer

    package = Path(__file__).resolve().parents[2] / "flzat_cognitive_pipeline"
    params = {
        "environment_id": "demo", "map_id": "map", "output_root": str(tmp_path),
        "prompt_directory": str(package / "prompts"),
        "context_policy_path": str(package / "config/intent_routes.yaml"),
        "skill_registry_path": str(package / "config/skill_registry.yaml"),
        "safety_policy_path": str(package / "config/safety_policy.yaml"),
        "user_profile_path": str(package / "config/user_profile.yaml"),
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
        probe = rclpy.create_node("routing_test_probe")
        nodes.extend([manager, fake, probe])
        publisher = probe.create_publisher(
            RoutedCommand, "/flzat/high_system/routed_command", 10,
        )
        commands = []
        probe.create_subscription(
            String, "/flzat/high_system/command", lambda msg: commands.append(msg.data), 10,
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
            raise AssertionError("ROS routing integration timed out")

        def routing(mode, relation, question=None):
            value = json.dumps({
                "mode": mode, "relation_to_pending": relation,
                "needs_clarification": question is not None,
                "clarification_question": question,
            })
            fake.set_parameters([Parameter("routing_response_text", value=value)])

        def send(text, mode="clarification"):
            message = RoutedCommand()
            message.utterance, message.mode = text, mode
            message.confidence = 0.3
            publisher.publish(message)

        fake.set_parameters([Parameter("response_text", value=json.dumps({
            "reply": "Scripted answer.", "speak": True, "requires_robot_action": False,
        }))])
        wait_for(lambda: publisher.get_subscription_count() > 0
                 and manager._vlm_client.server_is_ready())
        routing("general_qa", "new_request", "Which country do you mean?")
        send("Who is the president?")
        wait_for(lambda: manager._tasks.waiting_task_id == "task-0001")
        routing("general_qa", "continue")
        send("America")
        wait_for(lambda: manager._tasks.get("task-0001").state == PipelineState.RESPONSE_ONLY)
        assert len(seen) == 3  # routing, continuation routing, main inference
        runtime = json.loads(seen[-1].user_prompt.split("RUNTIME_INPUT_JSON\n")[1])
        assert runtime["english_transcript"] == "Who is the president?"
        assert runtime["dialogue_history"][0]["answer"] == "America"
        assert not any(goal.image.data for goal in seen)

        routing("general_qa", "new_request", "Which country do you mean?")
        send("Who is the president?")
        wait_for(lambda: manager._tasks.waiting_task_id == "task-0002")
        routing("general_qa", "new_request")
        send("What is the capital of Japan?", "general_qa")
        wait_for(lambda: manager._tasks._task_count == 3
                 and manager._tasks.get("task-0003").state == PipelineState.RESPONSE_ONLY)
        runtime = json.loads(seen[-1].user_prompt.split("RUNTIME_INPUT_JSON\n")[1])
        assert runtime["english_transcript"] == "What is the capital of Japan?"
        assert "dialogue_history" not in runtime
        assert not commands
        # No camera, RAG, or TTS server was started: QA still completes.
        assert list(tmp_path.rglob("resolved_routing.json"))
    finally:
        executor.shutdown(timeout_sec=5)
        if thread:
            thread.join(timeout=5)
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.try_shutdown()
