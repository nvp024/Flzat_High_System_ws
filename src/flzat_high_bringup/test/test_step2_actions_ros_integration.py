"""Opt-in live ROS A -> B -> natural answer integration test."""

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


def test_action_a_calls_b_and_b_answer_does_not_create_a_new_task(tmp_path):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from std_msgs.msg import String

    from flzat_cognitive_pipeline.cognitive_manager import CognitiveManagerNode
    from flzat_cognitive_pipeline.domain import PipelineState
    from flzat_high_bringup.fake_servers import FakeTtsServer, FakeVlmServer
    from flzat_high_bringup.low_system_mock import LowSystemMock

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
    executor = MultiThreadedExecutor(num_threads=8)
    nodes = []
    thread = None
    try:
        manager = CognitiveManagerNode()
        assert manager.get_name() == "vlm_brain"
        fake_vlm = FakeVlmServer()
        fake_vlm.set_parameters(
            [
                Parameter(
                    "response_text",
                    value=json.dumps(
                        {
                            "speech": "I will pass this goal to the robot.",
                            "goal": {
                                "description": "Bring the bottle to the user",
                                "constraints": [],
                                "selected_context_refs": [],
                                "missing_information": ["which bottle"],
                            },
                            "interaction": None,
                        }
                    ),
                )
            ]
        )
        fake_tts = FakeTtsServer()
        fake_tts.set_parameters([Parameter("delay_s", value=0.2)])
        low = LowSystemMock(
            parameter_overrides=[
                Parameter("result_outcome", value="STARTED_PLAN"),
                Parameter("ask_user_kind", value="ASK"),
                Parameter(
                    "ask_user_content_hint", value="Which bottle do you mean?"
                ),
                Parameter("ask_user_timeout_s", value=8.0),
                Parameter("ask_user_retry_interval_s", value=0.05),
                Parameter("ask_user_max_attempts", value=20),
                Parameter("transition_delay_s", value=0.01),
            ]
        )
        probe = rclpy.create_node("step2_action_probe")
        publisher = probe.create_publisher(
            String, "/flzat/high_system/text_command", 10
        )
        status_events = []
        raw_responses = []
        probe.create_subscription(
            String,
            "/flzat/high_system/status",
            lambda message: status_events.append(json.loads(message.data)),
            10,
        )
        probe.create_subscription(
            String, "/vlm/raw_response",
            lambda message: raw_responses.append(message.data), 10,
        )
        nodes.extend([manager, fake_vlm, fake_tts, low, probe])
        for node in nodes:
            executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()

        def wait_for(predicate, timeout=10.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return
                time.sleep(0.01)
            raise AssertionError("Step-2 ROS integration timed out")

        wait_for(
            lambda: publisher.get_subscription_count() > 0
            and manager._vlm_client.server_is_ready()
            and manager._tts_client.server_is_ready()
            and manager._intent_client.server_is_ready()
        )
        publisher.publish(String(data="Bring me the bottle."))
        wait_for(lambda: manager._interaction is not None and manager._interaction.listening)
        publisher.publish(String(data="The blue bottle on the table."))
        wait_for(
            lambda: manager._tasks.get("task-0001").state
            == PipelineState.LOW_RUNNING
        )

        assert manager._tasks._task_count == 1
        assert manager._interaction is None
        wait_for(lambda: bool(raw_responses))
        assert json.loads(raw_responses[0])["goal"]["description"] == "Bring the bottle to the user"
        assert any(event.get("event") == "ACTION_A_FEEDBACK" for event in status_events)
        assert any(
            event.get("state") == "BUSY"
            and event.get("event") == "action_b_rejected"
            and event.get("reason") == "tts_active"
            for event in status_events
        )
        artifacts = list(tmp_path.rglob("task-0001-c01.json"))
        assert len(artifacts) == 1
        handoff = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert handoff["schema_version"] == "flzat.high_handoff.v2"
        assert handoff["handoff_id"] == "task-0001-c01"
        assert "command_id" not in handoff
        assert "run_id" not in handoff
        assert handoff["goal"]["missing_information"] == ["which bottle"]
        assert handoff["entities"] == []
        audits = list(tmp_path.rglob("handoff_audit.json"))
        assert len(audits) == 1
        audit = json.loads(audits[0].read_text(encoding="utf-8"))
        assert audit["handoff_id"] == handoff["handoff_id"]
        assert audit["task_id"] == "task-0001"
    finally:
        executor.shutdown(timeout_sec=5)
        if thread:
            thread.join(timeout=5)
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.try_shutdown()
