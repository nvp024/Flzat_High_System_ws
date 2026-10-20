"""Exercise production manager callbacks with scripted action futures, no GPU/DDS."""

from concurrent.futures import Future
import json
from pathlib import Path
import threading
import time
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from flzat_cognitive_pipeline.artifacts import RequestArtifactWriter
from flzat_cognitive_pipeline.cognitive_manager import CognitiveManagerNode
from flzat_cognitive_pipeline.context_policy import ContextPolicy, CONTEXT_FLAGS
from flzat_cognitive_pipeline.context_providers import AdaptiveContextBuilder, TrustedYamlProvider
from flzat_cognitive_pipeline.conversation_store import ConversationStore
from flzat_cognitive_pipeline.domain import PipelineState
from flzat_cognitive_pipeline.output_validator import ModeOutputValidator, RepairPolicy
from flzat_cognitive_pipeline.prompt_composer import PromptComposer
from flzat_cognitive_pipeline.semantic_routing import parse_semantic_route
from flzat_cognitive_pipeline.task_store import TaskRunStore, TaskStateError
from flzat_high_interfaces.msg import RoutedCommand


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def route(mode="general_qa", relation="new_request", question=None):
    return json.dumps({
        "mode": mode, "relation_to_pending": relation,
        "needs_clarification": question is not None,
        "clarification_question": question,
    })


ANSWER = json.dumps({
    "reply": "A general answer.", "speak": True, "requires_robot_action": False,
})


class ScriptedClient:
    """Keep each goal/result future controllable to test late delivery."""

    def __init__(self):
        self.calls = []

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal):
        accepted = Future()
        result = Future()
        handle = SimpleNamespace(
            accepted=True, cancel_goal_async=Mock(),
            get_result_async=lambda: result,
        )
        self.calls.append(SimpleNamespace(
            goal=goal, accepted=accepted, result=result, handle=handle,
        ))
        return accepted

    def finish(self, text, index=-1, request_id=None):
        call = self.calls[index]
        if not call.accepted.done():
            call.accepted.set_result(call.handle)
        call.result.set_result(SimpleNamespace(result=SimpleNamespace(
            success=True, request_id=request_id or call.goal.request_id,
            response_text=text, error_message="",
        )))


@pytest.fixture
def manager(tmp_path):
    # Real callback implementations with fake transport and no ROS node startup.
    node = SimpleNamespace()
    for name, method in CognitiveManagerNode.__dict__.items():
        if callable(method) and name != "__init__":
            setattr(node, name, MethodType(method, node))
    node._lock = threading.RLock()
    node._tasks = TaskRunStore(run_id="run-routing-test")
    node._context_policy = ContextPolicy.from_yaml(CONFIG / "intent_routes.yaml")
    node._prompts = PromptComposer(str(ROOT / "prompts"))
    node._conversation = ConversationStore()
    node._adaptive_context = AdaptiveContextBuilder(
        skill_registry=TrustedYamlProvider(CONFIG / "skill_registry.yaml", "skills"),
        safety_policy=TrustedYamlProvider(CONFIG / "safety_policy.yaml", "safety_policy"),
        user_profile=TrustedYamlProvider(CONFIG / "user_profile.yaml", "user_profile"),
        conversation_store=node._conversation,
    )
    node._environment_id, node._map_id = "demo", "map"
    node._request_timeout_ns = 60_000_000_000
    node._active = None
    node._dispatched = {}
    node._success_speech = {}
    node._request_writer = RequestArtifactWriter(tmp_path)
    node._telemetry = Mock()
    node._publish_status = Mock()
    node._finish_metrics = Mock()
    node._speak = Mock()
    node.get_logger = Mock(return_value=Mock())
    node._vlm_client = ScriptedClient()
    node._validator = ModeOutputValidator(["find_object", "fetch_object", "navigate"])
    node._repair_policy = RepairPolicy(0)
    node._text_parameter = lambda name: "Request rejected"
    node._publish_command = Mock()
    node._query_memory = Mock()
    node._obtain_frame = Mock()
    node._low_cancel_pub = Mock()
    return node


def send(node, text, mode="clarification"):
    policy = node._context_policy.requirements_for(mode)
    message = RoutedCommand()
    message.utterance = text
    message.mode = mode
    message.confidence = 0.3
    message.language = "en"
    for name in CONTEXT_FLAGS:
        setattr(message, name, getattr(policy, name))
    node._on_routed_command(message)


def runtime_input(goal):
    return json.loads(goal.user_prompt.split("RUNTIME_INPUT_JSON\n")[1])


def test_uncertain_question_routes_then_answers_without_rag_or_camera(manager):
    text = "Who is the president of America?"
    send(manager, text)
    first = manager._vlm_client.calls[0].goal
    assert first.request_id == "task-0001-r01-routing"
    assert not first.image.data
    assert runtime_input(first)["latest_utterance"] == text
    manager._vlm_client.finish(route())
    second = manager._vlm_client.calls[1].goal
    assert second.request_id == "task-0001-r01"
    assert runtime_input(second)["english_transcript"] == text
    assert runtime_input(second)["router_mode"] == "general_qa"
    manager._query_memory.assert_not_called()
    manager._obtain_frame.assert_not_called()
    manager._vlm_client.finish(ANSWER)
    assert manager._tasks.get("task-0001").state == PipelineState.RESPONSE_ONLY
    manager._publish_command.assert_not_called()


def test_confident_general_question_calls_model_once(manager):
    send(manager, "What is the capital of Japan?", "general_qa")
    assert len(manager._vlm_client.calls) == 1
    assert not manager._vlm_client.calls[0].goal.request_id.endswith("-routing")
    manager._vlm_client.finish(ANSWER)
    manager._query_memory.assert_not_called()


def start_country_clarification(manager):
    send(manager, "Who is the president?")
    manager._vlm_client.finish(route(question="Which country do you mean?"))
    assert manager._tasks.waiting_task_id == "task-0001"
    assert manager._tasks.get("task-0001").mode == "general_qa"


def test_country_answer_preserves_original_and_history(manager):
    start_country_clarification(manager)
    send(manager, "American")
    pending = runtime_input(manager._vlm_client.calls[-1].goal)["pending_request"]
    assert pending["mode"] == "general_qa"
    assert pending["original_transcript"] == "Who is the president?"
    manager._vlm_client.finish(route(relation="continue"))
    main = runtime_input(manager._vlm_client.calls[-1].goal)
    assert main["english_transcript"] == "Who is the president?"
    assert main["dialogue_history"] == [
        {"question": "Which country do you mean?", "answer": "American"},
    ]
    assert main["router_mode"] == "general_qa"
    assert manager._active.task.request_id == "task-0001-r02"
    manager._query_memory.assert_not_called()
    manager._vlm_client.finish(ANSWER)


def test_new_question_while_waiting_gets_fresh_task_even_with_confident_route(manager):
    start_country_clarification(manager)
    send(manager, "What is the capital of Viet Nam?", "general_qa")
    assert manager._vlm_client.calls[-1].goal.request_id.endswith("-routing")
    manager._vlm_client.finish(route(relation="new_request"))
    main = runtime_input(manager._vlm_client.calls[-1].goal)
    assert main["english_transcript"] == "What is the capital of Viet Nam?"
    assert "dialogue_history" not in main
    assert manager._active.task.task_id == "task-0002"
    assert manager._tasks.get("task-0001").state == PipelineState.CANCELLED
    manager._vlm_client.finish(ANSWER)
    send(manager, "What is the capital of Japan?", "general_qa")
    assert manager._active.task.task_id == "task-0003"
    manager._vlm_client.finish(ANSWER)


def test_task_answer_only_enables_rag_after_explicit_routing(manager):
    send(manager, "get that")
    manager._vlm_client.finish(route("task_command", question="Which object?"))
    send(manager, "the book")
    manager._query_memory.assert_not_called()
    manager._vlm_client.finish(route("task_command", relation="continue"))
    manager._query_memory.assert_called_once()
    task = manager._active.task
    assert task.mode == "task_command"
    assert task.query_text == "get that\nQuestion: Which object?\nAnswer: the book"
    # Verify the original task and answer reach the normal task prompt.
    manager._prepare_prompt(manager._active, manager._empty_snapshot(task.request_id))
    manager._obtain_frame.assert_called_once()
    main = manager._active.prompt.user_prompt
    assert '"english_transcript":"get that"' in main
    assert '"answer":"the book"' in main


@pytest.mark.parametrize("raw", [
    "not JSON",
    '{"mode":"task_command"}',
    route(mode="clarification"),
    route(mode=None),
    route(question="Which country"),  # missing question mark
])
def test_bad_route_rejected_without_main_inference_or_dispatch(manager, raw):
    send(manager, "uncertain request")
    manager._vlm_client.finish(raw)
    assert manager._active is None
    assert manager._tasks.get("task-0001").state == PipelineState.REJECTED
    assert len(manager._vlm_client.calls) == 1
    manager._query_memory.assert_not_called()
    manager._publish_command.assert_not_called()


def test_cancel_before_goal_acceptance_cancels_late_handle(manager):
    send(manager, "uncertain request")
    manager._on_cancel(String(data=""))
    call = manager._vlm_client.calls[0]
    call.accepted.set_result(call.handle)
    call.handle.cancel_goal_async.assert_called_once()
    assert manager._active is None
    assert manager._tasks.get("task-0001").state == PipelineState.CANCELLED


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_late_routing_result_does_not_resume_stopped_task(manager, stop):
    send(manager, "uncertain request")
    call = manager._vlm_client.calls[0]
    call.accepted.set_result(call.handle)
    if stop == "cancel":
        manager._on_cancel(String(data=""))
    else:
        manager._active.deadline_ns = time.monotonic_ns() - 1
        manager._on_timer()
    manager._vlm_client.finish(route())
    assert len(manager._vlm_client.calls) == 1
    assert manager._active is None
    call.handle.cancel_goal_async.assert_called_once()


def test_stale_result_id_rejected(manager):
    send(manager, "uncertain request")
    manager._vlm_client.finish(route(), request_id="task-old-routing")
    assert manager._tasks.get("task-0001").state == PipelineState.REJECTED
    manager._query_memory.assert_not_called()


def test_pending_task_cannot_implicitly_consume_any_transcript(manager):
    start_country_clarification(manager)
    with pytest.raises(TaskStateError, match="explicit routing"):
        manager._tasks.accept_transcript("What is the capital of Japan?", "general_qa")


def test_new_ambiguous_request_can_ask_without_old_history(manager):
    start_country_clarification(manager)
    send(manager, "get that")
    manager._vlm_client.finish(route("task_command", question="Which item?"))
    assert manager._tasks.waiting_task_id == "task-0002"
    assert manager._tasks.get("task-0002").clarification_history == []


def test_null_mode_supported_only_for_genuine_clarification():
    decision = parse_semantic_route(
        route(None, question="What would you like me to do?"), has_pending=False,
    )
    assert decision.mode is None
    assert decision.needs_clarification


def test_no_pending_task_normalizes_continue_and_preserves_general_qa(manager):
    send(manager, "who is the presdent of American")
    manager._vlm_client.finish(route(relation="continue"))
    main = runtime_input(manager._vlm_client.calls[-1].goal)
    assert main["router_mode"] == "general_qa"
    assert main["english_transcript"] == "who is the presdent of American"
    assert "dialogue_history" not in main
    assert manager._active.task.task_id == "task-0001"
    manager._query_memory.assert_not_called()
    manager._obtain_frame.assert_not_called()
    manager._vlm_client.finish(ANSWER)
    manager._publish_command.assert_not_called()


def test_normalization_does_not_allow_invalid_mode_or_relation():
    with pytest.raises(ValueError):
        parse_semantic_route(route(mode="invalid", relation="continue"), has_pending=False)
    with pytest.raises(ValueError):
        parse_semantic_route(route(relation="invalid"), has_pending=False)
    assert parse_semantic_route(
        route(relation="continue"), has_pending=True,
    ).relation_to_pending == "continue"


def test_main_output_cannot_turn_general_question_into_action(manager):
    send(manager, "Who is the president of America?")
    manager._vlm_client.finish(route())
    manager._vlm_client.finish(json.dumps({
        "intent": "find_object",
        "target": {"type": "environment_object", "name": "ball"},
        "selected_context_refs": [],
        "execution_strategy": "search",
        "missing_information": [],
        "clarification_question": None,
        "speech": "I found the ball.",
    }))
    assert manager._tasks.get("task-0001").state == PipelineState.REJECTED
    manager._publish_command.assert_not_called()


def test_clarification_limit_does_not_block_independent_request(manager):
    start_country_clarification(manager)
    task = manager._tasks.get("task-0001")
    task.clarification_history = [("Question?", "Answer")] * 3
    send(manager, "What is the capital of Japan?", "general_qa")
    manager._vlm_client.finish(route())
    assert manager._active.task.task_id == "task-0002"
    manager._vlm_client.finish(ANSWER)


def test_continuation_limit_still_rejects_further_answers(manager):
    start_country_clarification(manager)
    task = manager._tasks.get("task-0001")
    task.clarification_history = [("Question?", "Answer")] * 3
    send(manager, "America")
    manager._vlm_client.finish(route(relation="continue"))
    assert task.state == PipelineState.REJECTED
    assert len(manager._vlm_client.calls) == 2
