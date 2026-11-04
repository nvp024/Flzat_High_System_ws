"""Step-1 manager ingress tests after removing semantic routing from the flow."""

from pathlib import Path
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

from builtin_interfaces.msg import Time
from std_msgs.msg import String

from flzat_cognitive_pipeline.artifacts import RequestArtifactWriter
from flzat_cognitive_pipeline.cognitive_manager import CognitiveManagerNode
from flzat_cognitive_pipeline.context_providers import (
    AdaptiveContextBuilder,
    TrustedYamlProvider,
)
from flzat_cognitive_pipeline.conversation_store import ConversationStore
from flzat_cognitive_pipeline.domain import PipelineState
from flzat_cognitive_pipeline.task_store import TaskRunStore


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def manager(tmp_path):
    node = SimpleNamespace()
    for name, method in CognitiveManagerNode.__dict__.items():
        if callable(method) and name != "__init__":
            setattr(node, name, MethodType(method, node))
    node._lock = threading.RLock()
    node._tasks = TaskRunStore(run_id="run-cognitive-test")
    node._conversation = ConversationStore()
    node._adaptive_context = AdaptiveContextBuilder(
        safety_policy=TrustedYamlProvider(
            CONFIG / "safety_policy.yaml", "safety_policy"
        ),
        user_profile=TrustedYamlProvider(CONFIG / "user_profile.yaml", "user_profile"),
        conversation_store=node._conversation,
    )
    node._environment_id, node._map_id = "demo", "map"
    node._request_timeout_ns = 60_000_000_000
    node._high_clarification_timeout_ns = 60_000_000_000
    node._active = None
    node._interaction = None
    node._interaction_reserved = False
    node._speech_busy = False
    node._waiting_deadlines = {}
    node._request_writer = RequestArtifactWriter(tmp_path)
    node._telemetry = Mock()
    node._publish_status = Mock()
    node._publish_busy = Mock()
    node._query_memory = Mock()
    node._on_cancel = Mock()
    node.get_logger = Mock(return_value=Mock())
    return node


def test_direct_input_starts_one_cognitive_request_without_router(tmp_path):
    node = manager(tmp_path)

    node._ingest_cognitive_input(
        "Who is the president of America?",
        source="text",
    )

    task = node._tasks.get("task-0001")
    assert task.mode == "cognitive"
    assert task.state == PipelineState.CONTEXT_RETRIEVAL
    assert node._active.task.mode == "cognitive"
    node._query_memory.assert_called_once_with(node._active)
    assert (tmp_path / node._tasks.run_id / "task-0001").is_dir()


def test_stt_timestamp_is_preserved_for_nearest_frame_lookup(tmp_path):
    node = manager(tmp_path)
    stamp = Time(sec=12, nanosec=345)

    node._ingest_cognitive_input(
        "Describe what is in front of you.",
        source="stt",
        input_stamp=stamp,
    )

    assert node._active.input_stamp == stamp


def test_pending_high_clarification_resumes_same_task(tmp_path):
    node = manager(tmp_path)
    node._ingest_cognitive_input("Bring me the bottle.", source="text")
    first = node._active
    node._tasks.wait_for_clarification(first.task, "Which bottle do you mean?")
    node._active = None
    node._query_memory.reset_mock()

    node._ingest_cognitive_input("The blue bottle on the table.", source="text")

    assert node._active.task.task_id == "task-0001"
    assert node._active.task.request_id == "task-0001-r02"
    assert node._active.task.clarification_history == [
        ("Which bottle do you mean?", "The blue bottle on the table.")
    ]
    node._query_memory.assert_called_once_with(node._active)


def test_ordinary_input_does_not_replace_active_inference(tmp_path):
    node = manager(tmp_path)
    node._ingest_cognitive_input("Bring me the book.", source="text")
    active = node._active

    node._ingest_cognitive_input("Go to the kitchen.", source="text")

    assert node._active is active
    assert node._tasks._task_count == 1
    node._publish_busy.assert_called_once_with(
        event="input_rejected",
        reason="cognitive_request_active",
        input_id="text",
    )


def test_explicit_stop_bypasses_busy_request(tmp_path):
    node = manager(tmp_path)
    node._active = object()

    node._on_text_command(String(data="stop"))

    node._on_cancel.assert_called_once()
