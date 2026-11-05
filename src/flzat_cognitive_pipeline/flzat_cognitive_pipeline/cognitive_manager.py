"""Canonical ROS orchestration for the FLZAT cognitive pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
from typing import Callable, Optional
from builtin_interfaces.msg import Time
from action_msgs.msg import GoalStatus

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from flzat_high_interfaces.action import AskUser, HandleIntent, RunVlm, TextToSpeech
from flzat_high_interfaces.msg import SpeechTranscript
from flzat_high_interfaces.srv import GetFrameNear, QueryMemory
from flzat_cognitive_pipeline.artifacts import (
    HandoffArtifactWriter,
    RequestArtifactWriter,
)
from flzat_cognitive_pipeline.command import (
    HighLevelHandoffAssembler,
    canonical_json,
)
from flzat_cognitive_pipeline.context_providers import (
    AdaptiveContextBuilder,
    TrustedYamlProvider,
)
from flzat_cognitive_pipeline.context_rag import ContextRagManager
from flzat_cognitive_pipeline.conversation_store import ConversationStore
from flzat_cognitive_pipeline.domain import (
    ContextSnapshot,
    MemoryCandidate,
    OutcomeKind,
    PipelineState,
    TaskRecord,
    TERMINAL_STATES,
    TrustedPosition,
)
from flzat_cognitive_pipeline.output_validator import (
    CognitiveOutputValidator,
    RepairPolicy,
    ValidatedCognitiveOutput,
)
from flzat_cognitive_pipeline.interaction import (
    ASK_USER_KINDS,
    ASK_USER_LISTENING_KINDS,
    HANDLE_INTENT_OUTCOMES,
    HANDLE_INTENT_STAGES,
    assess_user_answer,
    is_stop_command,
    validate_safe_identifier,
)
from flzat_cognitive_pipeline.prompt_composer import ComposedPrompt, PromptComposer
from flzat_cognitive_pipeline.task_store import TaskRunStore, TaskStateError
from flzat_cognitive_pipeline.telemetry import PipelineTelemetry
from flzat_cognitive_pipeline.transcript import is_usable_transcript


@dataclass
class RuntimeRequest:
    task: TaskRecord
    deadline_ns: int
    input_record: dict[str, object]
    snapshot: ContextSnapshot | None = None
    prompt: ComposedPrompt | None = None
    base_prompt: ComposedPrompt | None = None
    frame: CompressedImage | None = None
    attempts_sent: int = 0
    vlm_goal_handle: object | None = None
    input_stamp: Time | None = None
    context_availability: dict[str, dict[str, object]] | None = None
    handoff_id: str = ""
    handoff_path: str = ""
    handoff_goal_handle: object | None = None
    handoff_cancel_requested: bool = False
    vlm_sent_ns: int = 0


@dataclass
class UserInteractionSession:
    goal_handle: object
    run_id: str
    task_id: str
    handoff_id: str
    interaction_id: str
    kind: str
    content_hint: str
    options: tuple[str, ...]
    deadline_ns: int
    answer_event: threading.Event
    answer: str = ""
    interrupted_error: str = ""
    listening: bool = False
    repeat_count: int = 0


class CognitiveManagerNode(Node):
    """Connect speech, RAG, prompt, VLM, validation, artifacts and Low System."""

    def __init__(self) -> None:
        super().__init__("vlm_brain")
        self._cb_group = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self._declare_parameters()

        self._environment_id = self._text_parameter("environment_id")
        self._map_id = self._text_parameter("map_id")
        if not self._environment_id or not self._map_id:
            raise ValueError("environment_id and map_id are required")
        self._top_k = int(self.get_parameter("rag_top_k").value)
        if not 1 <= self._top_k <= 5:
            raise ValueError("rag_top_k must be between 1 and 5")
        self._request_timeout_ns = int(
            float(self.get_parameter("request_timeout_s").value) * 1_000_000_000
        )
        if self._request_timeout_ns <= 0:
            raise ValueError("request_timeout_s must be positive")
        self._high_clarification_timeout_ns = int(
            float(self.get_parameter("high_clarification_timeout_s").value)
            * 1_000_000_000
        )
        if self._high_clarification_timeout_ns <= 0:
            raise ValueError("high_clarification_timeout_s must be positive")
        self._interaction_max_reprompts = int(
            self.get_parameter("interaction_max_reprompts").value
        )
        if self._interaction_max_reprompts < 0:
            raise ValueError("interaction_max_reprompts cannot be negative")
        if float(self.get_parameter("health_period_s").value) <= 0.0:
            raise ValueError("health_period_s must be positive")
        self._tasks = TaskRunStore(
            max_clarifications=int(self.get_parameter("max_clarifications").value)
        )
        self._context_manager = ContextRagManager(
            max_items=int(self.get_parameter("context_max_items").value),
            max_context_characters=int(
                self.get_parameter("context_max_characters").value
            ),
            minimum_score=float(self.get_parameter("rag_minimum_score").value),
        )
        self._prompts = PromptComposer(self._text_parameter("prompt_directory"))
        self._conversation = ConversationStore(
            max_turns=int(self.get_parameter("conversation_max_turns").value),
            max_text_characters=int(
                self.get_parameter("conversation_max_characters").value
            ),
        )
        safety_policy = TrustedYamlProvider(
            self._trusted_config_path("safety_policy_path", "safety_policy.yaml"),
            "safety_policy",
        )
        user_profile = TrustedYamlProvider(
            self._trusted_config_path("user_profile_path", "user_profile.yaml"),
            "user_profile",
        )
        self._adaptive_context = AdaptiveContextBuilder(
            safety_policy=safety_policy,
            user_profile=user_profile,
            conversation_store=self._conversation,
        )
        self._validator = CognitiveOutputValidator()
        self._repair_policy = RepairPolicy(
            int(self.get_parameter("max_repair_attempts").value)
        )
        self._assembler = HighLevelHandoffAssembler()
        output_root = Path(self._text_parameter("output_root"))
        self._writer = HandoffArtifactWriter(output_root)
        self._request_writer = RequestArtifactWriter(output_root)
        self._telemetry = PipelineTelemetry(self._tasks.run_id)
        self._started_ns = time.monotonic_ns()
        self._active: RuntimeRequest | None = None
        self._dispatched: dict[str, TaskRecord] = {}
        self._handoff_paths: dict[str, str] = {}
        self._success_speech: dict[str, str] = {}
        self._interaction: UserInteractionSession | None = None
        self._interaction_reserved = False
        self._interaction_reservation_deadline_ns = 0
        self._waiting_deadlines: dict[str, int] = {}

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            String, "/flzat/high_system/status", status_qos
        )
        self._health_pub = self.create_publisher(
            String, "/flzat/high_system/health", status_qos
        )
        self._metrics_pub = self.create_publisher(
            String, "/flzat/high_system/metrics", qos
        )
        self._stt_sub = self.create_subscription(
            SpeechTranscript,
            "/flzat/high_system/stt_result",
            self._on_stt_result,
            qos,
            callback_group=self._cb_group,
        )
        self._text_sub = None
        if bool(self.get_parameter("enable_text_command").value):
            self._text_sub = self.create_subscription(
                String,
                "/flzat/high_system/text_command",
                self._on_text_command,
                qos,
                callback_group=self._cb_group,
            )
        self._cancel_sub = self.create_subscription(
            String,
            "/flzat/high_system/cancel",
            self._on_cancel,
            qos,
            callback_group=self._cb_group,
        )
        self._vlm_client = ActionClient(
            self, RunVlm, "/vlm/run", callback_group=self._cb_group
        )
        self._tts_client = ActionClient(
            self, TextToSpeech, "/tts_action", callback_group=self._cb_group
        )
        self._intent_client = ActionClient(
            self,
            HandleIntent,
            "/llm/handle_intent",
            callback_group=self._cb_group,
        )
        self._ask_user_server = ActionServer(
            self,
            AskUser,
            "/brain/ask_user",
            execute_callback=self._execute_ask_user,
            goal_callback=self._on_ask_user_goal,
            cancel_callback=self._on_ask_user_cancel,
            callback_group=self._cb_group,
        )
        self._speech_serial = 0
        self._speech_busy = False
        self._speech_handle = None
        self._speech_pending = None
        self._speech_completion: Callable[[bool, str], None] | None = None
        self._speech_deadline = 0.0
        self._speech_timeout_s = float(self.get_parameter("tts_timeout_s").value)
        if not math.isfinite(self._speech_timeout_s) or self._speech_timeout_s <= 0:
            raise ValueError("tts_timeout_s must be finite and positive")
        self._memory_client = self.create_client(
            QueryMemory, "/environment_memory/query", callback_group=self._cb_group
        )
        self._frame_client = self.create_client(
            GetFrameNear, "/vision/get_frame_near", callback_group=self._cb_group
        )
        self._timer = self.create_timer(
            0.2, self._on_timer, callback_group=self._cb_group
        )
        self._health_timer = self.create_timer(
            float(self.get_parameter("health_period_s").value),
            self._publish_health,
            callback_group=self._cb_group,
        )
        self.get_logger().info(
            "[BRAIN] Started; "
            f"run={self._tasks.run_id}, environment={self._environment_id}, "
            f"map={self._map_id}, "
            "frame_source=/vision/get_frame_near."
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("enable_text_command", False)
        self.declare_parameter("environment_id", "")
        self.declare_parameter("map_id", "")
        self.declare_parameter("rag_top_k", 5)
        self.declare_parameter("rag_minimum_score", -1.0)
        self.declare_parameter("context_max_items", 5)
        self.declare_parameter("context_max_characters", 6000)
        self.declare_parameter("conversation_max_turns", 5)
        self.declare_parameter("conversation_max_characters", 1000)
        self.declare_parameter("max_clarifications", 3)
        self.declare_parameter("interaction_max_reprompts", 2)
        self.declare_parameter("high_clarification_timeout_s", 60.0)
        self.declare_parameter("max_repair_attempts", 2)
        self.declare_parameter("request_timeout_s", 60.0)
        self.declare_parameter("health_period_s", 2.0)
        self.declare_parameter("tts_timeout_s", 65.0)
        self.declare_parameter("prompt_directory", "")
        self.declare_parameter("safety_policy_path", "")
        self.declare_parameter("user_profile_path", "")
        self.declare_parameter(
            "output_root", "~/.local/share/flzat/high_system/handoffs"
        )
        self.declare_parameter(
            "rejection_speech", "I could not safely process that request."
        )

    def _on_stt_result(self, message: SpeechTranscript) -> None:
        self._ingest_cognitive_input(
            message.transcript,
            source="stt",
            input_stamp=message.header.stamp,
        )

    def _on_text_command(self, message: String) -> None:
        self._ingest_cognitive_input(message.data, source="text")

    def _ingest_cognitive_input(
        self,
        transcript: str,
        *,
        source: str,
        input_stamp: Time | None = None,
    ) -> None:
        value = transcript.strip()
        if not is_usable_transcript(value):
            self.get_logger().warn("[BRAIN] Ignored input without a usable transcript.")
            return
        if is_stop_command(value):
            self._on_cancel(String(data=""))
            return
        with self._lock:
            interaction = self._interaction
            if interaction is not None and interaction.listening:
                interaction.answer = value
                interaction.listening = False
                self.get_logger().info(
                    f"[BRAIN] {interaction.interaction_id}: user={value!r}"
                )
                interaction.answer_event.set()
                return
            busy_reason = ""
            if interaction is not None or self._interaction_reserved:
                busy_reason = "user_interaction_active"
            elif self._active is not None:
                busy_reason = "cognitive_request_active"
            elif self._speech_busy:
                busy_reason = "tts_active"
            if busy_reason:
                self._publish_busy(
                    event="input_rejected",
                    reason=busy_reason,
                    input_id=source,
                )
                return
        self._accept_transcript(
            value,
            source,
            {
                "schema_version": "flzat.cognitive_input.v1",
                "source": source,
            },
            input_stamp=input_stamp,
        )

    def _accept_transcript(
        self,
        transcript: str,
        source: str,
        input_record: dict[str, object],
        *,
        input_stamp: Time | None = None,
    ) -> None:
        with self._lock:
            if self._active is not None:
                self.get_logger().warn("[BRAIN] Ignored transcript while a request is active.")
                return
            try:
                waiting_id = self._tasks.waiting_task_id
                task = self._tasks.accept_transcript(
                    transcript,
                    resume=bool(waiting_id),
                )
            except (ValueError, TaskStateError) as exc:
                if waiting_id and "clarification limit reached" in str(exc):
                    waiting = self._tasks.get(waiting_id)
                    self._tasks.force_terminal(
                        waiting,
                        PipelineState.REJECTED,
                        "UNRESOLVED_ANSWER",
                    )
                    self._waiting_deadlines.pop(waiting.task_id, None)
                    self._publish_status(waiting, "UNRESOLVED_ANSWER")
                self.get_logger().warn(f"[BRAIN] Transcript rejected: {exc}")
                return
            self._waiting_deadlines.pop(task.task_id, None)
            runtime = RuntimeRequest(
                task=task,
                deadline_ns=time.monotonic_ns() + self._request_timeout_ns,
                input_record=input_record,
                input_stamp=input_stamp,
                context_availability={
                    "memory": {"status": "pending", "reason": ""},
                    "vision": {"status": "pending", "reason": ""},
                },
            )
            self._telemetry.begin(task.request_id, task.task_id, source)
            self._active = runtime
            self.get_logger().info(f"[BRAIN] {task.request_id}: user={transcript!r}")
        try:
            self._write_initial_artifacts(runtime)
            self._telemetry.mark(task.request_id, "request_artifacts")
        except Exception as exc:
            self._reject(runtime, f"request artifact write failed: {exc}")
            return
        with self._lock:
            if not self._is_current(runtime):
                return
            self._publish_status(task, source)
        self._query_memory(runtime)

    def _query_memory(self, runtime: RuntimeRequest) -> None:
        if not self._is_current(runtime):
            return
        if not self._memory_client.service_is_ready():
            self._set_context_availability(
                runtime, "memory", "unavailable", "memory service is unavailable"
            )
            self._prepare_prompt(
                runtime, self._empty_snapshot(runtime.task.request_id)
            )
            return
        request = QueryMemory.Request()
        request.query = runtime.task.query_text
        request.top_k = self._top_k
        request.environment_id = self._environment_id
        request.map_id = self._map_id
        future = self._memory_client.call_async(request)
        future.add_done_callback(
            lambda completed: self._on_memory_result(
                runtime.task.task_id, runtime.task.request_id, completed
            )
        )

    def _on_memory_result(self, task_id: str, request_id: str, future) -> None:
        runtime = self._current(task_id, request_id)
        if runtime is None:
            return
        try:
            response = future.result()
            if not response.success:
                self._set_context_availability(
                    runtime,
                    "memory",
                    "unavailable",
                    response.error_message or "memory query failed",
                )
                self._prepare_prompt(runtime, self._empty_snapshot(request_id))
                return
            candidates = [
                self._memory_candidate(value, float(score))
                for value, score in zip(response.objects, response.cosine_scores)
            ]
            self.get_logger().debug(
                f"[RAG] {runtime.task.task_id}: {len(candidates)} result(s)"
            )
            snapshot = self._context_manager.build_snapshot(
                request_id,
                self._environment_id,
                self._map_id,
                candidates,
            )
            self._set_context_availability(
                runtime,
                "memory",
                "available",
                f"{len(snapshot.items)} grounded candidate(s)",
            )
            self._prepare_prompt(runtime, snapshot)
        except Exception as exc:
            self._set_context_availability(
                runtime, "memory", "unavailable", str(exc)
            )
            self._prepare_prompt(runtime, self._empty_snapshot(request_id))

    def _prepare_prompt(
        self,
        runtime: RuntimeRequest,
        snapshot: ContextSnapshot,
    ) -> None:
        if not self._is_current(runtime):
            return
        runtime.snapshot = snapshot
        self._telemetry.mark(runtime.task.request_id, "rag_context")
        self._obtain_frame(runtime)

    def _compose_prompt_and_send(
        self,
        runtime: RuntimeRequest,
        frame: CompressedImage,
    ) -> None:
        try:
            if not self._is_current(runtime):
                return
            snapshot = runtime.snapshot
            if snapshot is None:
                raise ValueError("trusted context snapshot is missing")
            adaptive = self._adaptive_context.build_cognitive(runtime.task, snapshot)
            interaction_context = None
            if runtime.task.clarification_history:
                interaction_context = {
                    "owner": "HIGH",
                    "original_request": runtime.task.original_transcript,
                    "turns": [
                        {"question": question, "answer": answer}
                        for question, answer in runtime.task.clarification_history
                    ],
                }
            runtime.prompt = self._prompts.compose(
                transcript=runtime.task.original_transcript,
                context_items=adaptive.context_items,
                context_refs=adaptive.context_refs,
                dialogue_history=adaptive.dialogue_history,
                mode="cognitive",
                trusted_context=adaptive.trusted_context,
                include_rag_context=True,
                context_availability=runtime.context_availability or {},
                interaction_context=interaction_context,
            )
            runtime.base_prompt = runtime.prompt
            self._telemetry.mark(runtime.task.request_id, "prompt_composition")
            self._request_writer.write_json(
                runtime.task.run_id,
                runtime.task.task_id,
                runtime.task.request_id,
                "rag_context.json",
                {
                    "schema_version": "flzat.rag_context.v1",
                    "enabled": True,
                    "environment_id": snapshot.environment_id,
                    "map_id": snapshot.map_id,
                    "context_refs": list(adaptive.context_refs),
                    "items": adaptive.context_items,
                    "truncated": snapshot.truncated,
                    "availability": runtime.context_availability or {},
                },
            )
            self._write_prompt_metadata(runtime, runtime.prompt, 1)
            self._telemetry.mark(runtime.task.request_id, "prompt_artifacts")
            with self._lock:
                if not self._is_current(runtime):
                    return
                self._tasks.transition(
                    runtime.task,
                    PipelineState.PROMPT_READY,
                    "cognitive context frozen",
                )
                self._tasks.transition(
                    runtime.task, PipelineState.VLM_QUEUED, "prompt composed"
                )
                self._publish_status(runtime.task, "prompt_ready")
            self._send_vlm(runtime, frame)
        except Exception as exc:
            self._reject(runtime, f"context/prompt processing failed: {exc}")

    def _obtain_frame(self, runtime: RuntimeRequest) -> None:
        stamp = runtime.input_stamp
        if stamp is None or not (stamp.sec or stamp.nanosec):
            stamp = self.get_clock().now().to_msg()
        if not self._frame_client.service_is_ready():
            self._set_context_availability(
                runtime, "vision", "unavailable", "frame service is unavailable"
            )
            self._telemetry.mark(runtime.task.request_id, "frame_skipped")
            self._compose_prompt_and_send(runtime, CompressedImage())
            return
        request = GetFrameNear.Request()
        request.target_stamp = stamp
        request.max_age_s = 2.0
        future = self._frame_client.call_async(request)
        future.add_done_callback(
            lambda completed: self._on_frame_result(
                runtime.task.task_id, runtime.task.request_id, completed
            )
        )

    def _on_frame_result(self, task_id: str, request_id: str, future) -> None:
        runtime = self._current(task_id, request_id)
        if runtime is None:
            return
        try:
            response = future.result()
            if not response.success or not response.frame.data:
                self._set_context_availability(
                    runtime,
                    "vision",
                    "unavailable",
                    response.message or "frame lookup failed",
                )
                self._telemetry.mark(request_id, "frame_skipped")
                self._compose_prompt_and_send(runtime, CompressedImage())
                return
            self._set_context_availability(
                runtime, "vision", "available", "frame matched input timestamp"
            )
            self._telemetry.mark(request_id, "frame_lookup")
            self._compose_prompt_and_send(runtime, response.frame)
        except Exception as exc:
            self._set_context_availability(runtime, "vision", "unavailable", str(exc))
            self._telemetry.mark(request_id, "frame_skipped")
            self._compose_prompt_and_send(runtime, CompressedImage())

    def _send_vlm(self, runtime: RuntimeRequest, frame: CompressedImage) -> None:
        if not self._is_current(runtime):
            return
        if time.monotonic_ns() >= runtime.deadline_ns:
            self._time_out(runtime, "deadline before main inference")
            return
        if not self._vlm_client.server_is_ready():
            self._reject(runtime, "VLM action server is unavailable")
            return
        prompt = runtime.prompt
        if prompt is None:
            self._reject(runtime, "composed prompt is missing")
            return
        if runtime.attempts_sent == 0:
            runtime.frame = frame
        if runtime.frame is None:
            self._reject(runtime, "frozen VLM frame is missing")
            return
        runtime.attempts_sent += 1
        attempt = runtime.attempts_sent
        goal = RunVlm.Goal()
        goal.image = runtime.frame
        goal.transcript = runtime.task.latest_transcript
        goal.request_id = runtime.task.request_id
        goal.system_prompt = prompt.system_prompt
        goal.user_prompt = prompt.user_prompt
        goal.prompt_version = prompt.version
        goal.prompt_hash = prompt.prompt_hash
        task_id, request_id = runtime.task.task_id, runtime.task.request_id
        runtime.vlm_sent_ns = time.monotonic_ns()
        self.get_logger().info(
            f"[VLM] {request_id}: processing; "
            f"image={'yes' if runtime.frame.data else 'no'}; attempt={attempt}"
        )
        future = self._vlm_client.send_goal_async(goal)
        future.add_done_callback(
            lambda completed: self._on_vlm_goal(
                task_id,
                request_id,
                attempt,
                completed,
            )
        )

    def _on_vlm_goal(
        self,
        task_id: str,
        request_id: str,
        attempt: int,
        future,
    ) -> None:
        runtime = self._current(task_id, request_id)
        try:
            handle = future.result()
            if runtime is None or runtime.attempts_sent != attempt:
                if handle.accepted:
                    handle.cancel_goal_async()
                return
            if not handle.accepted:
                self._reject(runtime, "VLM goal was rejected")
                return
            with self._lock:
                if not self._is_current(runtime):
                    handle.cancel_goal_async()
                    return
                runtime.vlm_goal_handle = handle
                self._telemetry.mark(request_id, f"vlm_queue_{attempt:02d}")
                self._tasks.transition(
                    runtime.task, PipelineState.VLM_ACTIVE, "VLM goal accepted"
                )
                self._publish_status(runtime.task, "vlm_active")
            result_future = handle.get_result_async()
            result_future.add_done_callback(
                lambda completed: self._on_vlm_result(
                    task_id,
                    request_id,
                    attempt,
                    completed,
                )
            )
        except Exception as exc:
            if runtime is not None:
                self._reject(runtime, f"VLM dispatch failed: {exc}")

    def _on_vlm_result(
        self,
        task_id: str,
        request_id: str,
        attempt: int,
        future,
    ) -> None:
        runtime = self._current(task_id, request_id)
        if runtime is None or runtime.attempts_sent != attempt:
            return
        if time.monotonic_ns() >= runtime.deadline_ns:
            self._time_out(runtime, "main inference deadline")
            return
        try:
            result = future.result().result
            if not result.success:
                self._reject(runtime, f"VLM failed: {result.error_message}")
                return
            if result.request_id and result.request_id != request_id:
                self._reject(runtime, "VLM returned a stale request_id")
                return
            if runtime.snapshot is None:
                self._reject(runtime, "trusted context snapshot is missing")
                return
            self._request_writer.write_text(
                runtime.task.run_id,
                runtime.task.task_id,
                request_id,
                f"vlm_attempt_{attempt:02d}.txt",
                result.response_text,
            )
            self._telemetry.mark(request_id, f"vlm_inference_{attempt:02d}")
            with self._lock:
                self._tasks.transition(
                    runtime.task,
                    PipelineState.OUTPUT_VALIDATING,
                    "VLM response received",
                )
            validated = self._validator.validate(
                result.response_text,
                runtime.snapshot,
                expected_request_id=request_id,
            )
            self._request_writer.write_json(
                runtime.task.run_id,
                runtime.task.task_id,
                request_id,
                f"validator_attempt_{attempt:02d}.json",
                validated.validation_record(attempt),
            )
            self._telemetry.mark(request_id, f"validator_{attempt:02d}")
            if not validated.valid:
                self._attempt_repair(runtime, result.response_text, validated.error)
                return
            self._request_writer.write_json(
                runtime.task.run_id,
                runtime.task.task_id,
                request_id,
                "validated_output.json",
                {
                    "schema_version": "flzat.validated_output.v1",
                    "mode": "cognitive",
                    "output": validated.value,
                },
            )
            self._telemetry.mark(request_id, "validated_artifact")
            if validated.kind == OutcomeKind.NEEDS_CLARIFICATION:
                decision_summary = "clarification"
            elif validated.kind == OutcomeKind.RESPONSE_ONLY:
                decision_summary = "speech"
            else:
                decision_summary = "speech+goal" if validated.reply else "goal"
            elapsed_s = (time.monotonic_ns() - runtime.vlm_sent_ns) / 1_000_000_000
            self.get_logger().info(
                f"[BRAIN] {request_id}: decision={decision_summary}; vlm_roundtrip={elapsed_s:.2f}s"
            )
            self._handle_validated(runtime, validated)
        except Exception as exc:
            self._reject(runtime, f"VLM result processing failed: {exc}")

    def _attempt_repair(
        self,
        runtime: RuntimeRequest,
        raw_response: str,
        validation_error: str,
    ) -> None:
        repair_number = self._repair_policy.next_repair_number(
            runtime.attempts_sent
        )
        if repair_number is None:
            self._request_writer.write_json(
                runtime.task.run_id,
                runtime.task.task_id,
                runtime.task.request_id,
                "failure.json",
                {
                    "schema_version": "flzat.request_failure.v1",
                    "reason": "output validation failed after repair limit",
                    "validation_error": validation_error,
                    "attempts": runtime.attempts_sent,
                },
            )
            self._reject(
                runtime,
                "output validation failed after repair limit: " + validation_error,
            )
            self._speak(self._text_parameter("rejection_speech"))
            return
        if time.monotonic_ns() >= runtime.deadline_ns:
            self._time_out(runtime, "request deadline before output repair")
            return
        self.get_logger().warn(
            f"[VLM] {runtime.task.request_id}: invalid output; "
            f"repair {repair_number}/{self._repair_policy.max_repair_attempts}"
        )
        self.get_logger().debug(f"Validation error: {validation_error}")
        base_prompt = runtime.base_prompt
        if base_prompt is None or runtime.frame is None:
            self._reject(runtime, "repair cannot reuse the frozen request")
            return
        try:
            repair_prompt = self._prompts.compose_repair(
                base_prompt,
                raw_response=raw_response,
                validation_error=validation_error,
                repair_number=repair_number,
            )
            runtime.prompt = repair_prompt
            self._write_prompt_metadata(
                runtime,
                repair_prompt,
                runtime.attempts_sent + 1,
            )
            self._telemetry.mark(
                runtime.task.request_id,
                f"repair_prompt_{repair_number:02d}",
            )
            with self._lock:
                if not self._is_current(runtime):
                    return
                self._tasks.transition(
                    runtime.task,
                    PipelineState.VLM_QUEUED,
                    f"output repair {repair_number} queued",
                )
                self._publish_status(runtime.task, f"repair_{repair_number}_queued")
            self._send_vlm(runtime, runtime.frame)
        except Exception as exc:
            self._reject(runtime, f"output repair failed: {exc}")

    def _handle_validated(
        self,
        runtime: RuntimeRequest,
        validated: ValidatedCognitiveOutput,
    ) -> None:
        task = runtime.task
        if validated.kind == OutcomeKind.NEEDS_CLARIFICATION:
            with self._lock:
                if not self._tasks.can_request_clarification(task):
                    self._tasks.force_terminal(
                        task, PipelineState.REJECTED, "UNRESOLVED_ANSWER"
                    )
                    self._active = None
                    self._publish_status(task, "UNRESOLVED_ANSWER")
                    self._finish_metrics(task.request_id, "unresolved_answer")
                    self.get_logger().warn(
                        f"[BRAIN] {task.task_id}: clarification limit reached."
                    )
                    return
                self._tasks.wait_for_clarification(task, validated.reply)
                self._waiting_deadlines[task.task_id] = (
                    time.monotonic_ns() + self._high_clarification_timeout_ns
                )
                self._active = None
                self._publish_status(task, "clarification_required")
                self._finish_metrics(task.request_id, "clarification")
            self._remember_turn(runtime, validated.reply)
            self._speak(validated.reply)
            return
        if validated.kind == OutcomeKind.RESPONSE_ONLY:
            with self._lock:
                self._tasks.transition(task, PipelineState.RESPONSE_ONLY, "response only")
                self._active = None
                self._publish_status(task, "response_only")
                self._finish_metrics(task.request_id, "response_only")
            self._remember_turn(runtime, validated.reply)
            self._speak(validated.reply)
            return
        if validated.task_outcome is None:
            self._reject(runtime, "cognitive validator did not return an outcome")
            return
        self._dispatch_handoff(
            runtime,
            validated.task_outcome,
            validated.reply,
        )

    def _dispatch_handoff(
        self, runtime: RuntimeRequest, outcome, success_speech: str
    ) -> None:
        task = runtime.task
        snapshot = runtime.snapshot
        if snapshot is None:
            self._reject(runtime, "cannot publish without context snapshot")
            return
        try:
            with self._lock:
                self._tasks.transition(
                    task, PipelineState.HANDOFF_READY, "decision validated"
                )
            handoff_id, payload = self._assembler.assemble(task, outcome, snapshot)
            self._request_writer.write_json(
                task.run_id,
                task.task_id,
                task.request_id,
                "handoff_audit.json",
                self._assembler.audit_record(task, handoff_id, snapshot),
            )
            with self._lock:
                self._tasks.mark_handoff(task, handoff_id)
                self._tasks.transition(task, PipelineState.WRITING_ARTIFACT, "writing JSON")
            artifact_path = self._writer.write(
                task.run_id, task.task_id, handoff_id, payload
            ).resolve()
            self._telemetry.mark(task.request_id, "handoff_artifact")
            with self._lock:
                self._tasks.transition(
                    task, PipelineState.HANDOFF_PENDING, "artifact committed"
                )
                runtime.handoff_id = handoff_id
                runtime.handoff_path = str(artifact_path)
                self._dispatched[handoff_id] = task
                self._handoff_paths[handoff_id] = str(artifact_path)
                self._publish_status(task, "handoff_artifact_ready")
            if not self._intent_client.server_is_ready():
                self._fail_handoff(runtime, "Action A server is unavailable")
                return
            goal = HandleIntent.Goal()
            goal.run_id = task.run_id
            goal.task_id = task.task_id
            goal.handoff_id = handoff_id
            goal.json_path = str(artifact_path)
            self._remember_turn(runtime, success_speech)
            # Start accompanying speech before Low can call B. Action A remains
            # independent, while an immediate B observes TTS BUSY and retries.
            self._speak(success_speech)
            future = self._intent_client.send_goal_async(
                goal,
                feedback_callback=lambda feedback: self._on_intent_feedback(
                    task.task_id, task.request_id, feedback
                ),
            )
            future.add_done_callback(
                lambda done: self._on_intent_goal(
                    task.task_id, task.request_id, done
                )
            )
        except Exception as exc:
            self._reject(runtime, f"handoff assembly/write failed: {exc}")

    def _on_intent_goal(self, task_id: str, request_id: str, future) -> None:
        runtime = self._current(task_id, request_id)
        try:
            handle = future.result()
            if runtime is None:
                if handle.accepted:
                    handle.cancel_goal_async()
                return
            if not handle.accepted:
                self._fail_handoff(runtime, "Action A goal was rejected")
                return
            with self._lock:
                if not self._is_current(runtime):
                    handle.cancel_goal_async()
                    return
                runtime.handoff_goal_handle = handle
                self._telemetry.mark(request_id, "action_a_accepted")
                cancel_requested = runtime.handoff_cancel_requested
                self.get_logger().info(f"[ACTION_A] {task_id}: accepted")
            handle.get_result_async().add_done_callback(
                lambda done: self._on_intent_result(task_id, request_id, done)
            )
            if cancel_requested:
                handle.cancel_goal_async()
        except Exception as exc:
            if runtime is not None:
                self._fail_handoff(runtime, f"Action A goal failed: {exc}")

    def _on_intent_feedback(self, task_id: str, request_id: str, message) -> None:
        runtime = self._current(task_id, request_id)
        if runtime is None:
            return
        stage = str(message.feedback.stage).strip()
        detail = str(message.feedback.detail).strip()
        if stage not in HANDLE_INTENT_STAGES:
            self.get_logger().warn(
                f"[ACTION_A] Ignored unsupported feedback stage: {stage!r}"
            )
            return
        self._publish_action_event(
            event="ACTION_A_FEEDBACK",
            task=runtime.task,
            handoff_id=runtime.handoff_id,
            detail=detail,
            stage=stage,
        )

    def _on_intent_result(self, task_id: str, request_id: str, future) -> None:
        runtime = self._current(task_id, request_id)
        if runtime is None:
            return
        try:
            response = future.result()
            result = response.result
            outcome = str(result.outcome).strip()
            detail = str(result.detail).strip()
            if outcome not in HANDLE_INTENT_OUTCOMES:
                raise ValueError(f"unsupported Action A outcome: {outcome!r}")
            expected_status = {
                "STARTED_FAST": GoalStatus.STATUS_SUCCEEDED,
                "STARTED_PLAN": GoalStatus.STATUS_SUCCEEDED,
                "REFUSED": GoalStatus.STATUS_SUCCEEDED,
                "FAILED": GoalStatus.STATUS_ABORTED,
                "CANCELED": GoalStatus.STATUS_CANCELED,
            }[outcome]
            if response.status != expected_status:
                raise ValueError(
                    f"Action A status/outcome mismatch: {response.status}/{outcome}"
                )
            with self._lock:
                if not self._is_current(runtime):
                    return
                task = runtime.task
                self._tasks.transition(
                    task, PipelineState.HANDOFF_ACCEPTED, f"Action A {outcome}"
                )
                if outcome in {"STARTED_FAST", "STARTED_PLAN"}:
                    self._tasks.transition(
                        task, PipelineState.LOW_RUNNING, outcome
                    )
                elif outcome == "REFUSED":
                    self._tasks.transition(task, PipelineState.REFUSED, detail or outcome)
                elif outcome == "FAILED":
                    self._tasks.transition(task, PipelineState.FAILED, detail or outcome)
                else:
                    self._tasks.transition(
                        task, PipelineState.CANCELLED, detail or outcome
                    )
                self._active = None
                self._publish_status(task, f"action_a:{outcome}")
                self._finish_metrics(request_id, outcome.casefold())
            self._publish_action_event(
                event="ACTION_A_RESULT",
                task=runtime.task,
                handoff_id=runtime.handoff_id,
                detail=detail,
                outcome=outcome,
            )
            self._publish_ready_if_available()
        except Exception as exc:
            self._fail_handoff(runtime, f"Action A result failed: {exc}")

    def _fail_handoff(self, runtime: RuntimeRequest, reason: str) -> None:
        with self._lock:
            if not self._is_current(runtime):
                return
            self._tasks.force_terminal(runtime.task, PipelineState.FAILED, reason)
            self._active = None
            self._publish_status(runtime.task, reason)
            self._finish_metrics(runtime.task.request_id, "action_a_transport_failed")
        self.get_logger().error(f"[ACTION_A] {runtime.task.task_id}: {reason}")
        self._publish_ready_if_available()

    def _on_cancel(self, message: String) -> None:
        requested_id = message.data.strip()
        with self._lock:
            interaction = self._interaction
            if interaction is not None and requested_id in {
                "",
                interaction.task_id,
                interaction.handoff_id,
                interaction.interaction_id,
            }:
                interaction.interrupted_error = "INTERRUPTED_BY_USER"
                interaction.answer_event.set()
                if not requested_id:
                    self._stop_speech()
                self._publish_action_event(
                    event="ACTION_B_INTERRUPTED",
                    task=self._dispatched.get(interaction.handoff_id),
                    handoff_id=interaction.handoff_id,
                    interaction_id=interaction.interaction_id,
                    detail="INTERRUPTED_BY_USER",
                )
                return

            if not requested_id:
                self._stop_speech()

            runtime = self._active
            if runtime is not None and requested_id in {
                "",
                runtime.task.task_id,
                runtime.task.request_id,
                runtime.handoff_id,
            }:
                if runtime.handoff_id:
                    runtime.handoff_cancel_requested = True
                    if runtime.handoff_goal_handle is not None:
                        runtime.handoff_goal_handle.cancel_goal_async()
                    self._publish_action_event(
                        event="ACTION_A_CANCEL_REQUESTED",
                        task=runtime.task,
                        handoff_id=runtime.handoff_id,
                        detail="waiting for Low Action A cancellation result",
                    )
                    return
                if runtime.vlm_goal_handle is not None:
                    runtime.vlm_goal_handle.cancel_goal_async()
                self._tasks.force_terminal(
                    runtime.task, PipelineState.CANCELLED, "cancel topic"
                )
                self._active = None
                self._publish_status(runtime.task, "cancelled")
                self._finish_metrics(runtime.task.request_id, "cancelled")
                self._publish_ready_if_available()
                return

            waiting_id = self._tasks.waiting_task_id
            if waiting_id:
                waiting = self._tasks.get(waiting_id)
                if requested_id in {"", waiting.task_id, waiting.request_id}:
                    self._tasks.force_terminal(
                        waiting, PipelineState.CANCELLED, "cancel clarification"
                    )
                    self._waiting_deadlines.pop(waiting.task_id, None)
                    self._publish_status(waiting, "cancelled")
                    self._publish_ready_if_available()
                    return

            for handoff_id, task in reversed(tuple(self._dispatched.items())):
                if requested_id not in {
                    "",
                    handoff_id,
                    task.task_id,
                    task.request_id,
                }:
                    continue
                self._publish_action_event(
                    event="LOW_CANCEL_UNAVAILABLE",
                    task=task,
                    handoff_id=handoff_id,
                    detail=(
                        "Action A has ended; this phase has no Low execution "
                        "cancel contract"
                    ),
                )
                return

    def _on_ask_user_goal(self, goal_request: AskUser.Goal) -> GoalResponse:
        try:
            run_id = validate_safe_identifier(goal_request.run_id, "run_id")
            task_id = validate_safe_identifier(goal_request.task_id, "task_id")
            handoff_id = validate_safe_identifier(
                goal_request.handoff_id, "handoff_id"
            )
            interaction_id = validate_safe_identifier(
                goal_request.interaction_id, "interaction_id"
            )
            kind = str(goal_request.kind).strip().upper()
            if kind not in ASK_USER_KINDS:
                raise ValueError(f"unsupported interaction kind: {kind!r}")
            content_hint = str(goal_request.content_hint).strip()
            if not content_hint:
                raise ValueError("content_hint cannot be empty")
            timeout_s = float(goal_request.timeout_s)
            if not math.isfinite(timeout_s) or timeout_s <= 0.0:
                raise ValueError("timeout_s must be finite and positive")
            with self._lock:
                task = self._dispatched.get(handoff_id)
                if (
                    task is None
                    or task.run_id != run_id
                    or task.task_id != task_id
                ):
                    raise ValueError("Action B does not reference a known handoff")
                busy_reason = ""
                if self._interaction is not None or self._interaction_reserved:
                    busy_reason = "user_interaction_active"
                elif self._speech_busy:
                    busy_reason = "tts_active"
                elif self._active is not None and not (
                    self._active.task is task
                    and self._active.handoff_id == handoff_id
                    and self._active.task.state == PipelineState.HANDOFF_PENDING
                ):
                    busy_reason = "cognitive_request_active"
                if busy_reason:
                    self._publish_busy(
                        event="action_b_rejected",
                        reason=busy_reason,
                        interaction_id=interaction_id,
                    )
                    return GoalResponse.REJECT
                self._interaction_reserved = True
                self._interaction_reservation_deadline_ns = (
                    time.monotonic_ns() + int(timeout_s * 1_000_000_000)
                )
            return GoalResponse.ACCEPT
        except Exception as exc:
            self.get_logger().warn(f"[ACTION_B] Rejected goal: {exc}")
            return GoalResponse.REJECT

    def _on_ask_user_cancel(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_ask_user(self, goal_handle) -> AskUser.Result:
        request = goal_handle.request
        session = UserInteractionSession(
            goal_handle=goal_handle,
            run_id=str(request.run_id).strip(),
            task_id=str(request.task_id).strip(),
            handoff_id=str(request.handoff_id).strip(),
            interaction_id=str(request.interaction_id).strip(),
            kind=str(request.kind).strip().upper(),
            content_hint=str(request.content_hint).strip(),
            options=tuple(str(value).strip() for value in request.options),
            deadline_ns=(
                self._interaction_reservation_deadline_ns
                or time.monotonic_ns()
                + int(float(request.timeout_s) * 1_000_000_000)
            ),
            answer_event=threading.Event(),
        )
        with self._lock:
            if self._interaction is not None:
                self._interaction_reserved = False
                self._interaction_reservation_deadline_ns = 0
                goal_handle.abort()
                return self._ask_user_result(error="INTERACTION_STATE_CONFLICT")
            self._interaction = session
            self._interaction_reserved = False
            self._interaction_reservation_deadline_ns = 0
        self._publish_action_event(
            event="ACTION_B_ACCEPTED",
            task=self._dispatched.get(session.handoff_id),
            handoff_id=session.handoff_id,
            interaction_id=session.interaction_id,
            detail=session.kind,
        )
        try:
            speech_status = self._speak_interaction(session, session.content_hint)
            terminal = self._speech_terminal_result(
                goal_handle, session, speech_status
            )
            if terminal is not None:
                return terminal

            if session.kind not in ASK_USER_LISTENING_KINDS:
                goal_handle.succeed()
                return self._ask_user_result(outcome="ANSWERED")

            while True:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return self._ask_user_result()
                if session.interrupted_error:
                    goal_handle.abort()
                    return self._ask_user_result(error=session.interrupted_error)
                if time.monotonic_ns() >= session.deadline_ns:
                    goal_handle.succeed()
                    return self._ask_user_result(outcome="TIMEOUT")

                feedback = AskUser.Feedback()
                feedback.stage = "listening"
                feedback.detail = "waiting for a natural-language answer"
                session.answer_event.clear()
                session.listening = True
                goal_handle.publish_feedback(feedback)
                self.get_logger().info(
                    f"[ACTION_B] {session.interaction_id}: listening; waiting for user answer"
                )
                while not session.answer_event.wait(timeout=0.05):
                    if goal_handle.is_cancel_requested:
                        session.listening = False
                        goal_handle.canceled()
                        return self._ask_user_result()
                    if time.monotonic_ns() >= session.deadline_ns:
                        session.listening = False
                        goal_handle.succeed()
                        return self._ask_user_result(outcome="TIMEOUT")
                session.listening = False
                if session.interrupted_error:
                    goal_handle.abort()
                    return self._ask_user_result(error=session.interrupted_error)
                assessment = assess_user_answer(session.answer)
                if assessment.disposition == "ANSWERED":
                    goal_handle.succeed()
                    return self._ask_user_result(
                        outcome="ANSWERED", answer=assessment.raw_answer
                    )
                if assessment.disposition == "DECLINED":
                    goal_handle.succeed()
                    return self._ask_user_result(
                        outcome="DECLINED", answer=assessment.raw_answer
                    )
                if session.repeat_count >= self._interaction_max_reprompts:
                    goal_handle.abort()
                    return self._ask_user_result(error="UNRESOLVED_ANSWER")
                session.repeat_count += 1
                session.answer = ""
                repeat = (
                    "I did not understand your answer. " + session.content_hint
                )
                speech_status = self._speak_interaction(session, repeat)
                terminal = self._speech_terminal_result(
                    goal_handle, session, speech_status
                )
                if terminal is not None:
                    return terminal
        except Exception as exc:
            self.get_logger().error(f"[ACTION_B] Execution failed: {exc}")
            try:
                goal_handle.abort()
            except Exception:
                pass
            return self._ask_user_result(error=f"INTERACTION_ERROR:{exc}")
        finally:
            with self._lock:
                if self._interaction is session:
                    self._interaction = None
                self._interaction_reserved = False
                self._interaction_reservation_deadline_ns = 0
            self._publish_action_event(
                event="ACTION_B_FINISHED",
                task=self._dispatched.get(session.handoff_id),
                handoff_id=session.handoff_id,
                interaction_id=session.interaction_id,
                detail="interaction resources released",
            )
            self._publish_ready_if_available()

    def _speak_interaction(
        self, session: UserInteractionSession, text: str
    ) -> str:
        feedback = AskUser.Feedback()
        feedback.stage = "speaking"
        feedback.detail = text
        session.goal_handle.publish_feedback(feedback)
        completed = threading.Event()
        result: dict[str, object] = {"success": False, "error": ""}

        def on_complete(success: bool, error: str) -> None:
            result["success"] = success
            result["error"] = error
            completed.set()

        if not self._speak(text, on_complete=on_complete, allow_queue=False):
            return "FAILED"
        while not completed.wait(timeout=0.05):
            if session.goal_handle.is_cancel_requested:
                self._stop_speech()
                return "CANCELED"
            if session.interrupted_error:
                self._stop_speech()
                return "INTERRUPTED"
            if time.monotonic_ns() >= session.deadline_ns:
                self._stop_speech()
                return "TIMEOUT"
        if not bool(result["success"]):
            self.get_logger().warn(
                f"[ACTION_B] TTS failed: {str(result['error'])}"
            )
            return "FAILED"
        return "COMPLETED"

    def _speech_terminal_result(
        self, goal_handle, session: UserInteractionSession, status: str
    ) -> AskUser.Result | None:
        if session.interrupted_error:
            goal_handle.abort()
            return self._ask_user_result(error=session.interrupted_error)
        if status == "COMPLETED":
            return None
        if status == "CANCELED":
            goal_handle.canceled()
            return self._ask_user_result()
        if status == "TIMEOUT":
            goal_handle.succeed()
            return self._ask_user_result(outcome="TIMEOUT")
        if status == "INTERRUPTED":
            goal_handle.abort()
            return self._ask_user_result(
                error=session.interrupted_error or "INTERRUPTED_BY_USER"
            )
        goal_handle.abort()
        return self._ask_user_result(error="TTS_UNAVAILABLE_OR_FAILED")

    def _ask_user_result(
        self, *, outcome: str = "", answer: str = "", error: str = ""
    ) -> AskUser.Result:
        result = AskUser.Result()
        result.outcome = outcome
        result.answer = answer
        result.error_message = error
        session = self._interaction
        identifier = session.interaction_id if session else "unassigned"
        message = f"[ACTION_B] {identifier}: {outcome or error or 'CANCELED'}"
        if error:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)
        return result

    def _on_timer(self) -> None:
        with self._lock:
            self._expire_speech()
            waiting_id = self._tasks.waiting_task_id
            if waiting_id:
                deadline = self._waiting_deadlines.get(waiting_id, 0)
                if deadline and time.monotonic_ns() >= deadline:
                    waiting = self._tasks.get(waiting_id)
                    self._tasks.force_terminal(
                        waiting,
                        PipelineState.TIMED_OUT,
                        "clarification deadline",
                    )
                    self._waiting_deadlines.pop(waiting_id, None)
                    self._publish_status(waiting, "clarification_timed_out")
                    self._publish_ready_if_available()
        with self._lock:
            runtime = self._active
            if runtime is None or time.monotonic_ns() < runtime.deadline_ns:
                return
        self._time_out(runtime, "request deadline")

    def _time_out(self, runtime: RuntimeRequest, reason: str) -> None:
        with self._lock:
            if not self._is_current(runtime):
                return
            if runtime.vlm_goal_handle is not None:
                runtime.vlm_goal_handle.cancel_goal_async()
            if runtime.handoff_goal_handle is not None:
                runtime.handoff_goal_handle.cancel_goal_async()
            self._tasks.force_terminal(runtime.task, PipelineState.TIMED_OUT, reason)
            self._active = None
            self._publish_status(runtime.task, "timed_out")
            self._finish_metrics(runtime.task.request_id, "timed_out")

    def _reject(self, runtime: RuntimeRequest, reason: str) -> None:
        with self._lock:
            if not self._is_current(runtime):
                return
            if runtime.handoff_goal_handle is not None:
                runtime.handoff_goal_handle.cancel_goal_async()
            self._tasks.force_terminal(runtime.task, PipelineState.REJECTED, reason)
            self._active = None
            self._publish_status(runtime.task, reason)
            self._finish_metrics(runtime.task.request_id, "rejected")
        self.get_logger().error(f"[BRAIN] {runtime.task.request_id}: {reason}")

    def _current(self, task_id: str, request_id: str) -> RuntimeRequest | None:
        with self._lock:
            runtime = self._active
            if (
                runtime is None
                or runtime.task.task_id != task_id
                or runtime.task.request_id != request_id
                or runtime.task.state in TERMINAL_STATES
            ):
                return None
            return runtime

    def _is_current(self, runtime: RuntimeRequest) -> bool:
        return self._active is runtime and runtime.task.state not in TERMINAL_STATES

    def _publish_status(self, task: TaskRecord, reason: str) -> None:
        payload = {
            "schema_version": "flzat.high_status.v2",
            "run_id": task.run_id,
            "task_id": task.task_id,
            "request_id": task.request_id,
            "handoff_id": task.handoff_id,
            "state": task.state.value,
            "reason": reason,
        }
        serialized = canonical_json(payload)
        self._status_pub.publish(String(data=serialized))
        visible_states = {
            PipelineState.WAITING_CLARIFICATION,
            PipelineState.SUCCEEDED,
            PipelineState.CANCELLED,
            PipelineState.TIMED_OUT,
        }
        if task.state in visible_states:
            detail = ""
            if task.state in {
                PipelineState.WAITING_CLARIFICATION,
                PipelineState.REFUSED,
                PipelineState.FAILED,
                PipelineState.CANCELLED,
                PipelineState.TIMED_OUT,
            }:
                detail = f" ({reason})"
            self.get_logger().info(
                f"[BRAIN] {task.task_id}: state={task.state.value}{detail}"
            )
        self.get_logger().debug(f"HIGH_STATUS {serialized}")

    def _publish_busy(
        self,
        *,
        event: str,
        reason: str,
        input_id: str = "",
        interaction_id: str = "",
    ) -> None:
        active = self._active
        interaction = self._interaction
        payload = {
            "schema_version": "flzat.high_availability.v1",
            "run_id": self._tasks.run_id,
            "state": "BUSY",
            "event": event,
            "reason": reason,
            "accepted": False,
            "queued": False,
            "input_id": input_id,
            "interaction_id": interaction_id,
            "active_task_id": active.task.task_id if active else "",
            "active_request_id": active.task.request_id if active else "",
            "active_interaction_id": (
                interaction.interaction_id if interaction else ""
            ),
        }
        self._status_pub.publish(String(data=canonical_json(payload)))
        self.get_logger().info(
            f"[BRAIN] Busy; rejected={event}; reason={reason}; queued=false"
        )

    def _publish_ready_if_available(self) -> None:
        with self._lock:
            if (
                self._active is not None
                or self._interaction is not None
                or self._interaction_reserved
                or self._speech_busy
                or self._tasks.waiting_task_id
            ):
                return
            payload = {
                "schema_version": "flzat.high_availability.v1",
                "run_id": self._tasks.run_id,
                "state": "READY",
                "event": "resources_released",
                "accepted": True,
                "queued": False,
            }
        self._status_pub.publish(String(data=canonical_json(payload)))
        self.get_logger().info("[BRAIN] Ready for user input.")

    def _publish_action_event(
        self,
        *,
        event: str,
        task: TaskRecord | None,
        handoff_id: str,
        interaction_id: str = "",
        detail: str = "",
        stage: str = "",
        outcome: str = "",
    ) -> None:
        payload = {
            "schema_version": "flzat.high_action_event.v1",
            "run_id": task.run_id if task else self._tasks.run_id,
            "task_id": task.task_id if task else "",
            "request_id": task.request_id if task else "",
            "handoff_id": handoff_id,
            "interaction_id": interaction_id,
            "event": event,
            "stage": stage,
            "outcome": outcome,
            "detail": detail,
        }
        self._status_pub.publish(String(data=canonical_json(payload)))
        summary = stage or outcome or detail
        identifier = interaction_id or (task.task_id if task else handoff_id)
        # A's result is summarized once by _finish_metrics. Full event payloads
        # and feedback remain available on ROS topics and at DEBUG.
        self.get_logger().debug(f"HIGH_ACTION_EVENT {canonical_json(payload)}")
        if event not in {"ACTION_A_FEEDBACK", "ACTION_A_RESULT", "ACTION_B_FINISHED"}:
            if event.startswith("ACTION_B_"):
                component, event_name = "ACTION_B", event.removeprefix("ACTION_B_")
            else:
                component, event_name = "ACTION_A", event.removeprefix("ACTION_A_")
            self.get_logger().info(
                f"[{component}] {identifier}: {event_name.lower()}; {summary}".rstrip()
            )

    def _finish_metrics(self, request_id: str, outcome: str) -> None:
        payload = self._telemetry.finish(request_id, outcome)
        if payload is None:
            return
        serialized = canonical_json(payload)
        self._metrics_pub.publish(String(data=serialized))
        component = (
            "ACTION_A"
            if outcome.upper() in HANDLE_INTENT_OUTCOMES or outcome == "action_a_transport_failed"
            else "BRAIN"
        )
        self.get_logger().info(
            f"[{component}] {payload['request_id']}: result={outcome.upper()}; "
            f"high={float(payload['total_ms']) / 1000:.2f}s"
        )
        self.get_logger().debug(f"HIGH_METRICS {serialized}")

    def _publish_health(self) -> None:
        with self._lock:
            active = self._active
            waiting_task_id = self._tasks.waiting_task_id
            dependencies = {
                "memory": self._memory_client.service_is_ready(),
                "frame": self._frame_client.service_is_ready(),
                "tts": self._tts_client.server_is_ready(),
                "vlm": self._vlm_client.server_is_ready(),
                "action_a": self._intent_client.server_is_ready(),
            }
            # Memory and vision are optional context sources in the unified flow.
            base_ready = (
                dependencies["tts"]
                and dependencies["vlm"]
                and dependencies["action_a"]
            )
            mode_readiness = {"cognitive": bool(base_ready)}
            active_mode = "cognitive" if active else ""
            payload = {
                "schema_version": "flzat.high_health.v2",
                "run_id": self._tasks.run_id,
                "node": self.get_name(),
                "ready": mode_readiness["cognitive"],
                "dependencies": dependencies,
                "mode_readiness": mode_readiness,
                "active_mode": active_mode,
                "active_task_id": active.task.task_id if active else "",
                "active_request_id": active.task.request_id if active else "",
                "waiting_clarification_task_id": waiting_task_id,
                "active_interaction_id": (
                    self._interaction.interaction_id if self._interaction else ""
                ),
                "uptime_s": round(
                    (time.monotonic_ns() - self._started_ns) / 1_000_000_000.0,
                    3,
                ),
                "counters": self._telemetry.snapshot(),
            }
        self._health_pub.publish(String(data=canonical_json(payload)))

    def _speak(
        self,
        text: str,
        *,
        on_complete: Callable[[bool, str], None] | None = None,
        allow_queue: bool = True,
    ) -> bool:
        value = text.strip()
        if not value:
            if on_complete is not None:
                on_complete(True, "")
            return True
        with self._lock:
            if self._speech_busy:
                if not allow_queue or on_complete is not None:
                    return False
                if self._speech_pending is not None:
                    self.get_logger().warn("[TTS] Busy; replaced older queued reply.")
                self._speech_pending = (value, time.monotonic())
                return True
            if not self._tts_client.server_is_ready():
                self.get_logger().warn("[TTS] Unavailable; speech was not played.")
                return False
            self._speech_serial += 1
            serial = self._speech_serial
            self._speech_busy = True
            self._speech_completion = on_complete
            self._speech_deadline = time.monotonic() + self._speech_timeout_s
            try:
                self.get_logger().info(f"[TTS] Speaking: {value!r}")
                self._tts_client.send_goal_async(TextToSpeech.Goal(text=value)).add_done_callback(
                    lambda future: self._on_speech_goal(serial, future)
                )
            except Exception as exc:
                self.get_logger().warn(f"[TTS] Dispatch failed: {exc}")
                self._finish_speech(False, f"dispatch failed: {exc}")
                return False
            return True

    def _on_speech_goal(self, serial: int, future) -> None:
        with self._lock:
            self._expire_speech()
            try:
                handle = future.result()
                if serial != self._speech_serial or not self._speech_busy:
                    if handle.accepted:
                        handle.cancel_goal_async()
                    return
                if not handle.accepted:
                    self.get_logger().warn("[TTS] Rejected reply; speech was not played.")
                    self._finish_speech(False, "goal rejected")
                    return
                self._speech_handle = handle
                handle.get_result_async().add_done_callback(
                    lambda result: self._on_speech_result(serial, result)
                )
            except Exception as exc:
                self.get_logger().warn(f"[TTS] Goal failed: {exc}")
                if serial == self._speech_serial:
                    self._finish_speech(False, f"goal failed: {exc}")

    def _on_speech_result(self, serial: int, future) -> None:
        with self._lock:
            self._expire_speech()
            if serial != self._speech_serial or not self._speech_busy:
                return
            try:
                response = future.result()
                if response.status != GoalStatus.STATUS_SUCCEEDED or not response.result.success:
                    self.get_logger().warn(f"[TTS] Reply failed: status={response.status}.")
                    success = False
                    error = f"status={response.status}"
                else:
                    self.get_logger().debug("TTS reply completed.")
                    success = True
                    error = ""
            except Exception as exc:
                self.get_logger().warn(f"[TTS] Result failed: {exc}")
                success = False
                error = f"result failed: {exc}"
            self._finish_speech(success, error)

    def _expire_speech(self) -> None:
        if self._speech_busy and time.monotonic() >= self._speech_deadline:
            self.get_logger().warn("[TTS] Response timed out; discarding queued speech.")
            self._stop_speech()

    def _finish_speech(self, success: bool = True, error: str = "") -> None:
        self._speech_serial += 1
        self._speech_busy = False
        self._speech_handle = None
        completion, self._speech_completion = self._speech_completion, None
        pending, self._speech_pending = self._speech_pending, None
        if completion is not None:
            try:
                completion(success, error)
            except Exception as exc:
                self.get_logger().warn(f"[TTS] Completion callback failed: {exc}")
        if pending is not None:
            text, received_at = pending
            if time.monotonic() - received_at < self._speech_timeout_s:
                self._speak(text)
            else:
                self.get_logger().warn("[TTS] Discarded expired queued reply.")
        else:
            self._publish_ready_if_available()

    def _stop_speech(self) -> None:
        with self._lock:
            if not self._speech_busy and self._speech_pending is None:
                return
            handle = self._speech_handle
            self._speech_pending = None
            self._finish_speech(False, "speech cancelled")
            if handle is not None:
                try:
                    handle.cancel_goal_async()
                except Exception as exc:
                    self.get_logger().warn(f"[TTS] Cancellation failed: {exc}")

    def _remember_turn(self, runtime: RuntimeRequest, assistant: str) -> None:
        if not assistant.strip():
            return
        try:
            self._conversation.append(
                "cognitive",
                runtime.task.latest_transcript,
                assistant,
            )
        except ValueError as exc:
            self.get_logger().warn(f"[BRAIN] Conversation turn was not stored: {exc}")

    def _write_initial_artifacts(self, runtime: RuntimeRequest) -> None:
        task = runtime.task
        self._request_writer.write_json(
            task.run_id,
            task.task_id,
            task.request_id,
            "transcript.json",
            {
                "schema_version": "flzat.transcript.v1",
                "task_id": task.task_id,
                "request_id": task.request_id,
                "mode": "cognitive",
                "transcript": task.latest_transcript,
            },
        )
        self._request_writer.write_json(
            task.run_id,
            task.task_id,
            task.request_id,
            "input.json",
            runtime.input_record,
        )

    def _write_prompt_metadata(
        self,
        runtime: RuntimeRequest,
        prompt: ComposedPrompt,
        attempt: int,
    ) -> None:
        filename = (
            "prompt_metadata.json"
            if attempt == 1
            else f"prompt_metadata_{attempt:02d}.json"
        )
        self._request_writer.write_json(
            runtime.task.run_id,
            runtime.task.task_id,
            runtime.task.request_id,
            filename,
            {
                "schema_version": "flzat.prompt_metadata.v1",
                "attempt": attempt,
                "profile": prompt.profile,
                "version": prompt.version,
                "mode": prompt.mode,
                "output_schema": prompt.output_schema,
                "validator": prompt.validator,
                "prompt_hash": prompt.prompt_hash,
                "sections": list(prompt.sections),
                "context_refs": list(prompt.context_refs),
            },
        )

    @staticmethod
    def _set_context_availability(
        runtime: RuntimeRequest,
        source: str,
        status: str,
        reason: str,
    ) -> None:
        if status not in {"available", "unavailable", "pending"}:
            raise ValueError("unsupported context availability status")
        if runtime.context_availability is None:
            runtime.context_availability = {}
        runtime.context_availability[source] = {
            "status": status,
            "reason": reason.strip()[:500],
        }

    def _empty_snapshot(self, request_id: str) -> ContextSnapshot:
        return ContextSnapshot(
            request_id=request_id,
            environment_id=self._environment_id,
            map_id=self._map_id,
            items=(),
            truncated=False,
        )

    def _trusted_config_path(self, parameter: str, filename: str) -> str:
        configured = self._text_parameter(parameter)
        if configured:
            return configured
        from ament_index_python.packages import get_package_share_directory

        return str(
            Path(get_package_share_directory("flzat_cognitive_pipeline"))
            / "config"
            / filename
        )

    @staticmethod
    def _memory_candidate(message, score: float) -> MemoryCandidate:
        if len(message.attribute_keys) != len(message.attribute_values):
            raise ValueError("memory attributes have mismatched keys and values")
        point = message.map_position.point
        position = None
        if message.map_position.header.frame_id == "map" and all(
            math.isfinite(value) for value in (point.x, point.y, point.z)
        ):
            position = TrustedPosition("map", point.x, point.y, point.z)
        return MemoryCandidate(
            object_id=message.object_id,
            environment_id=message.environment_id,
            map_id=message.map_id,
            label=message.label or message.detector_class,
            description=message.description,
            relationships=tuple(message.relationships),
            scene=message.scene,
            score=score,
            position=position,
            attributes=tuple(zip(message.attribute_keys, message.attribute_values)),
            first_seen_ros_ns=(
                int(message.first_seen.sec) * 1_000_000_000
                + int(message.first_seen.nanosec)
            ),
            last_seen_ros_ns=(
                int(message.last_seen.sec) * 1_000_000_000
                + int(message.last_seen.nanosec)
            ),
            localization_quality=float(message.localization_quality),
            confidence=float(message.confidence),
        )

    def _text_parameter(self, name: str) -> str:
        return str(self.get_parameter(name).value).strip()

    def destroy_node(self) -> None:
        self._stop_speech()
        self._ask_user_server.destroy()
        self._intent_client.destroy()
        self._vlm_client.destroy()
        self._tts_client.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[CognitiveManagerNode] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = CognitiveManagerNode()
        executor = MultiThreadedExecutor(num_threads=6)
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
