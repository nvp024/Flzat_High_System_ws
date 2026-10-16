"""Canonical ROS orchestration for the FLZAT cognitive pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from flzat_high_interfaces.action import RunVlm, TextToSpeech
from flzat_high_interfaces.msg import RoutedCommand
from flzat_high_interfaces.srv import GetFrameNear, QueryMemory
from flzat_cognitive_pipeline.artifacts import (
    CommandArtifactWriter,
    RequestArtifactWriter,
)
from flzat_cognitive_pipeline.command import TrustedCommandAssembler, canonical_json
from flzat_cognitive_pipeline.context_policy import (
    ContextPolicy,
    ContextPolicyError,
    ContextRequirements,
    ROUTE_MODES,
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
    ModeOutputValidator,
    RepairPolicy,
    ValidatedModeOutput,
)
from flzat_cognitive_pipeline.prompt_composer import ComposedPrompt, PromptComposer
from flzat_cognitive_pipeline.routing import (
    LowStatusAdapter,
    build_low_cancel,
)
from flzat_cognitive_pipeline.task_store import TaskRunStore, TaskStateError
from flzat_cognitive_pipeline.telemetry import PipelineTelemetry
from flzat_cognitive_pipeline.transcript import is_usable_transcript


@dataclass
class RuntimeRequest:
    task: TaskRecord
    deadline_ns: int
    policy: ContextRequirements
    router_record: dict[str, object]
    snapshot: ContextSnapshot | None = None
    prompt: ComposedPrompt | None = None
    base_prompt: ComposedPrompt | None = None
    frame: CompressedImage | None = None
    attempts_sent: int = 0
    vlm_goal_handle: object | None = None


class CognitiveManagerNode(Node):
    """Connect speech, RAG, prompt, VLM, validation, artifacts and Low System."""

    def __init__(self) -> None:
        super().__init__("cognitive_manager")
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
        if float(self.get_parameter("health_period_s").value) <= 0.0:
            raise ValueError("health_period_s must be positive")
        self._routed_topic = self._text_parameter("routed_command_topic")

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
        self._context_policy = ContextPolicy.from_yaml(
            self._trusted_config_path("context_policy_path", "intent_routes.yaml")
        )
        self._conversation = ConversationStore(
            max_turns=int(self.get_parameter("conversation_max_turns").value),
            max_text_characters=int(
                self.get_parameter("conversation_max_characters").value
            ),
        )
        skill_registry = TrustedYamlProvider(
            self._trusted_config_path("skill_registry_path", "skill_registry.yaml"),
            "skills",
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
            skill_registry=skill_registry,
            safety_policy=safety_policy,
            user_profile=user_profile,
            conversation_store=self._conversation,
        )
        skills = skill_registry.get()
        allowed_skills = [
            item.get("name", "")
            for item in skills
            if isinstance(item, dict)
        ]
        self._validator = ModeOutputValidator(allowed_skills)
        self._repair_policy = RepairPolicy(
            int(self.get_parameter("max_repair_attempts").value)
        )
        self._assembler = TrustedCommandAssembler()
        output_root = Path(self._text_parameter("output_root"))
        self._writer = CommandArtifactWriter(output_root)
        self._request_writer = RequestArtifactWriter(output_root)
        self._low_status = LowStatusAdapter()
        self._telemetry = PipelineTelemetry(self._tasks.run_id)
        self._started_ns = time.monotonic_ns()
        self._active: RuntimeRequest | None = None
        self._dispatched: dict[str, TaskRecord] = {}
        self._success_speech: dict[str, str] = {}

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
        self._command_pub = self.create_publisher(
            String, self._text_parameter("command_topic"), qos
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
        self._low_cancel_pub = self.create_publisher(
            String, self._text_parameter("low_cancel_topic"), qos
        )
        self._routed_sub = self.create_subscription(
            RoutedCommand,
            self._routed_topic,
            self._on_routed_command,
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
        self._low_sub = self.create_subscription(
            String,
            self._text_parameter("low_status_topic"),
            self._on_low_status,
            qos,
            callback_group=self._cb_group,
        )
        self._vlm_client = ActionClient(
            self, RunVlm, "/vlm/run", callback_group=self._cb_group
        )
        self._tts_client = ActionClient(
            self, TextToSpeech, "/tts_action", callback_group=self._cb_group
        )
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
            "Cognitive pipeline ready: "
            f"run={self._tasks.run_id}, environment={self._environment_id}, "
            f"map={self._map_id}, "
            "frame_source=/vision/get_frame_near."
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "routed_command_topic", "/flzat/high_system/routed_command"
        )
        self.declare_parameter("command_topic", "/flzat/high_system/command")
        self.declare_parameter("low_status_topic", "/flzat/low_system/status")
        self.declare_parameter("low_cancel_topic", "/flzat/low_system/cancel")
        self.declare_parameter("environment_id", "")
        self.declare_parameter("map_id", "")
        self.declare_parameter("rag_top_k", 5)
        self.declare_parameter("rag_minimum_score", -1.0)
        self.declare_parameter("context_max_items", 5)
        self.declare_parameter("context_max_characters", 6000)
        self.declare_parameter("conversation_max_turns", 5)
        self.declare_parameter("conversation_max_characters", 1000)
        self.declare_parameter("max_clarifications", 3)
        self.declare_parameter("max_repair_attempts", 2)
        self.declare_parameter("request_timeout_s", 60.0)
        self.declare_parameter("health_period_s", 2.0)
        self.declare_parameter("prompt_directory", "")
        self.declare_parameter("context_policy_path", "")
        self.declare_parameter("skill_registry_path", "")
        self.declare_parameter("safety_policy_path", "")
        self.declare_parameter("user_profile_path", "")
        self.declare_parameter(
            "output_root", "~/.local/share/flzat/high_system/commands"
        )
        self.declare_parameter(
            "rejection_speech", "I could not safely process that request."
        )

    def _on_routed_command(self, message: RoutedCommand) -> None:
        transcript = message.utterance.strip()
        if not is_usable_transcript(transcript):
            self.get_logger().warn("Ignored routed command without a usable utterance.")
            return
        if (
            len(message.route_names) != len(message.route_scores)
            or not math.isfinite(float(message.confidence))
            or not math.isfinite(float(message.latency_ms))
        ):
            self.get_logger().warn("Ignored malformed routed command metadata.")
            return
        if message.mode == "safety_stop":
            # The router also publishes the canonical high-level cancel event.
            return
        with self._lock:
            waiting_id = self._tasks.waiting_task_id
            waiting = self._tasks.get(waiting_id) if waiting_id else None
        continuation_mode = ""
        if waiting is not None:
            continuation_mode = (
                "task_command" if waiting.mode == "clarification" else waiting.mode
            )
        proposed_flags = {
            "needs_vision": bool(message.needs_vision),
            "needs_task_memory": bool(message.needs_task_memory),
            "needs_conversation_memory": bool(message.needs_conversation_memory),
            "needs_user_profile": bool(message.needs_user_profile),
            "needs_external_knowledge": bool(message.needs_external_knowledge),
        }
        try:
            policy = self._context_policy.resolve(
                message.mode,
                proposed_flags,
                continuation_mode=continuation_mode,
            )
        except ContextPolicyError as exc:
            self.get_logger().warn(f"Routed command rejected by context policy: {exc}")
            return
        if policy.corrected_flags:
            self.get_logger().warn(
                "ContextPolicy corrected untrusted router flags: "
                + ", ".join(policy.corrected_flags)
            )
        self._accept_transcript(
            transcript,
            f"intent_router:{message.mode}",
            policy,
            {
                "schema_version": "flzat.router_record.v1",
                "router_version": message.router_version,
                "routed_mode": message.mode,
                "effective_mode": policy.mode,
                "confidence": float(message.confidence),
                "language": message.language,
                "urgency": message.urgency,
                "route_scores": {
                    name: float(score)
                    for name, score in zip(
                        message.route_names,
                        message.route_scores,
                    )
                },
                "latency_ms": float(message.latency_ms),
            },
        )

    def _accept_transcript(
        self,
        transcript: str,
        source: str,
        policy: ContextRequirements,
        router_record: dict[str, object],
    ) -> None:
        with self._lock:
            if self._active is not None:
                self.get_logger().warn("Ignored transcript while a request is active.")
                return
            try:
                task = self._tasks.accept_transcript(transcript, policy.mode)
            except (ValueError, TaskStateError) as exc:
                self.get_logger().warn(f"Transcript rejected: {exc}")
                return
            runtime = RuntimeRequest(
                task=task,
                deadline_ns=time.monotonic_ns() + self._request_timeout_ns,
                policy=policy,
                router_record=router_record,
            )
            self._telemetry.begin(task.request_id, task.task_id, source)
            self._active = runtime
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
        if policy.needs_task_memory:
            self._query_memory(runtime)
        else:
            self._prepare_prompt(runtime, self._empty_snapshot(task.request_id))

    def _query_memory(self, runtime: RuntimeRequest) -> None:
        if not self._memory_client.service_is_ready():
            self._reject(runtime, "memory query service is unavailable")
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
                self._reject(runtime, f"RAG query failed: {response.error_message}")
                return
            candidates = [
                self._memory_candidate(value, float(score))
                for value, score in zip(response.objects, response.cosine_scores)
            ]
            snapshot = self._context_manager.build_snapshot(
                request_id,
                self._environment_id,
                self._map_id,
                candidates,
            )
            self._prepare_prompt(runtime, snapshot)
        except Exception as exc:
            self._reject(runtime, f"context/prompt processing failed: {exc}")

    def _prepare_prompt(
        self,
        runtime: RuntimeRequest,
        snapshot: ContextSnapshot,
    ) -> None:
        try:
            adaptive = self._adaptive_context.build(
                runtime.policy,
                runtime.task,
                snapshot,
            )
            runtime.snapshot = snapshot
            self._telemetry.mark(
                runtime.task.request_id,
                "rag_context" if runtime.policy.needs_task_memory else "context_build",
            )
            runtime.prompt = self._prompts.compose(
                transcript=runtime.task.original_transcript,
                context_items=adaptive.context_items,
                context_refs=adaptive.context_refs,
                dialogue_history=adaptive.dialogue_history,
                mode=adaptive.mode,
                trusted_context=adaptive.trusted_context,
                include_rag_context=adaptive.include_rag_context,
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
                    "enabled": adaptive.include_rag_context,
                    "environment_id": snapshot.environment_id,
                    "map_id": snapshot.map_id,
                    "context_refs": list(adaptive.context_refs),
                    "items": adaptive.context_items,
                    "truncated": snapshot.truncated,
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
                    f"{runtime.policy.mode} context frozen",
                )
                self._tasks.transition(
                    runtime.task, PipelineState.VLM_QUEUED, "prompt composed"
                )
                self._publish_status(runtime.task, "prompt_ready")
            if runtime.policy.needs_vision:
                self._obtain_frame(runtime)
            else:
                self._telemetry.mark(runtime.task.request_id, "frame_skipped")
                self._send_vlm(runtime, CompressedImage())
        except Exception as exc:
            self._reject(runtime, f"context/prompt processing failed: {exc}")

    def _obtain_frame(self, runtime: RuntimeRequest) -> None:
        if not runtime.policy.needs_vision:
            self._reject(runtime, "ContextPolicy forbids frame lookup for this mode")
            return
        stamp = self.get_clock().now().to_msg()
        if not self._frame_client.service_is_ready():
            self._reject(runtime, "frame service is unavailable")
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
                self._reject(runtime, f"frame lookup failed: {response.message}")
                return
            self._send_vlm(runtime, response.frame)
        except Exception as exc:
            self._reject(runtime, f"frame lookup failed: {exc}")

    def _send_vlm(self, runtime: RuntimeRequest, frame: CompressedImage) -> None:
        if not self._vlm_client.server_is_ready():
            self._reject(runtime, "VLM action server is unavailable")
            return
        prompt = runtime.prompt
        if prompt is None:
            self._reject(runtime, "composed prompt is missing")
            return
        if runtime.policy.needs_vision and not frame.data:
            self._reject(runtime, "vision mode cannot invoke VLM without a frame")
            return
        if not runtime.policy.needs_vision and frame.data:
            self._reject(runtime, "text-only mode received an unexpected frame")
            return
        if runtime.attempts_sent == 0:
            runtime.frame = frame
        if runtime.frame is None:
            self._reject(runtime, "frozen VLM frame is missing")
            return
        if runtime.policy.needs_vision and runtime.attempts_sent == 0:
            self._telemetry.mark(runtime.task.request_id, "frame_lookup")
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
        future = self._vlm_client.send_goal_async(goal)
        future.add_done_callback(
            lambda completed: self._on_vlm_goal(
                runtime.task.task_id,
                runtime.task.request_id,
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
        if runtime is None or runtime.attempts_sent != attempt:
            return
        try:
            handle = future.result()
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
                runtime.policy.mode,
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
                    "mode": validated.mode,
                    "output": validated.value,
                },
            )
            self._telemetry.mark(request_id, "validated_artifact")
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
        validated: ValidatedModeOutput,
    ) -> None:
        task = runtime.task
        if validated.kind == OutcomeKind.NEEDS_CLARIFICATION:
            with self._lock:
                self._tasks.wait_for_clarification(task, validated.reply)
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
        if not runtime.policy.allows_low_command:
            self._reject(
                runtime,
                f"route mode '{runtime.policy.mode}' cannot publish a High Command",
            )
            self._speak(self._text_parameter("rejection_speech"))
            return
        if validated.task_outcome is None:
            self._reject(runtime, "task validator did not return a task outcome")
            return
        self._publish_command(
            runtime,
            validated.task_outcome,
            validated.reply,
        )

    def _publish_command(self, runtime: RuntimeRequest, outcome, success_speech: str) -> None:
        if not runtime.policy.allows_low_command:
            self._reject(runtime, "ContextPolicy blocked a non-task Low command")
            return
        task = runtime.task
        snapshot = runtime.snapshot
        if snapshot is None:
            self._reject(runtime, "cannot publish without context snapshot")
            return
        try:
            with self._lock:
                self._tasks.transition(task, PipelineState.COMMAND_READY, "decision validated")
            command_id, payload = self._assembler.assemble(task, outcome, snapshot)
            with self._lock:
                self._tasks.mark_command(task, command_id)
                self._tasks.transition(task, PipelineState.WRITING_ARTIFACT, "writing JSON")
            self._writer.write(task.run_id, task.task_id, command_id, payload)
            self._telemetry.mark(task.request_id, "command_artifact")
            with self._lock:
                self._tasks.transition(
                    task, PipelineState.PUBLISHING_COMMAND, "artifact committed"
                )
                self._tasks.transition(task, PipelineState.DISPATCHED, "ROS command publish")
                self._dispatched[command_id] = task
                self._success_speech[command_id] = success_speech
                self._active = None
                self._publish_status(task, "dispatched")
            self._command_pub.publish(String(data=payload))
            self._finish_metrics(task.request_id, "dispatched")
        except Exception as exc:
            self._reject(runtime, f"command assembly/write failed: {exc}")

    def _on_low_status(self, message: String) -> None:
        try:
            raw = json.loads(message.data)
            command_id = raw.get("command_id") if isinstance(raw, dict) else ""
            task = self._dispatched.get(command_id)
            if task is None:
                raise ValueError("status references an unknown command")
            status = self._low_status.parse(
                message.data,
                task.run_id,
                command_id,
            )
            with self._lock:
                if task.state in TERMINAL_STATES:
                    return
                if status.state == "running" and task.state == PipelineState.DISPATCHED:
                    self._tasks.transition(task, PipelineState.LOW_RUNNING, "Low running")
                elif status.state == "succeeded":
                    self._tasks.transition(task, PipelineState.SUCCEEDED, "Low succeeded")
                elif status.state == "failed":
                    self._tasks.transition(
                        task, PipelineState.FAILED, status.error or "Low failed"
                    )
                elif status.state == "cancelled":
                    self._tasks.transition(task, PipelineState.CANCELLED, "Low cancelled")
                self._publish_status(task, f"low_{status.state}")
            if status.state == "succeeded":
                self._speak(self._success_speech.pop(command_id, ""))
            elif status.state == "failed":
                self._speak(status.error or "The robot could not complete the task.")
        except Exception as exc:
            self.get_logger().warn(f"Ignored invalid Low status: {exc}")

    def _on_cancel(self, message: String) -> None:
        requested_id = message.data.strip()
        low_cancel_payload = ""
        with self._lock:
            runtime = self._active
            if runtime is not None and requested_id in {
                "",
                runtime.task.task_id,
                runtime.task.request_id,
            }:
                if runtime.vlm_goal_handle is not None:
                    runtime.vlm_goal_handle.cancel_goal_async()
                self._tasks.force_terminal(
                    runtime.task, PipelineState.CANCELLED, "cancel topic"
                )
                self._active = None
                self._publish_status(runtime.task, "cancelled")
                self._finish_metrics(runtime.task.request_id, "cancelled")
                return

            waiting_id = self._tasks.waiting_task_id
            if waiting_id:
                waiting = self._tasks.get(waiting_id)
                if requested_id in {"", waiting.task_id, waiting.request_id}:
                    self._tasks.force_terminal(
                        waiting, PipelineState.CANCELLED, "cancel clarification"
                    )
                    self._publish_status(waiting, "cancelled")
                    return

            for command_id, task in reversed(tuple(self._dispatched.items())):
                if task.state in TERMINAL_STATES:
                    continue
                if requested_id not in {"", command_id, task.task_id, task.request_id}:
                    continue
                low_cancel_payload = build_low_cancel(
                    task.run_id,
                    command_id,
                    task.task_id,
                    task.request_id,
                )
                self._success_speech.pop(command_id, None)
                self._publish_status(task, "low_cancel_requested")
                break
        if low_cancel_payload:
            self._low_cancel_pub.publish(String(data=low_cancel_payload))

    def _on_timer(self) -> None:
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
            self._tasks.force_terminal(runtime.task, PipelineState.TIMED_OUT, reason)
            self._active = None
            self._publish_status(runtime.task, "timed_out")
            self._finish_metrics(runtime.task.request_id, "timed_out")

    def _reject(self, runtime: RuntimeRequest, reason: str) -> None:
        with self._lock:
            if not self._is_current(runtime):
                return
            self._tasks.force_terminal(runtime.task, PipelineState.REJECTED, reason)
            self._active = None
            self._publish_status(runtime.task, reason)
            self._finish_metrics(runtime.task.request_id, "rejected")
        self.get_logger().error(reason)

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
            "command_id": task.command_id,
            "state": task.state.value,
            "reason": reason,
        }
        serialized = canonical_json(payload)
        self._status_pub.publish(String(data=serialized))
        self.get_logger().info(f"HIGH_STATUS {serialized}")

    def _finish_metrics(self, request_id: str, outcome: str) -> None:
        payload = self._telemetry.finish(request_id, outcome)
        if payload is None:
            return
        serialized = canonical_json(payload)
        self._metrics_pub.publish(String(data=serialized))
        self.get_logger().info(f"HIGH_METRICS {serialized}")

    def _publish_health(self) -> None:
        with self._lock:
            active = self._active
            waiting_task_id = self._tasks.waiting_task_id
            dependencies = {
                "memory": self._memory_client.service_is_ready(),
                "frame": self._frame_client.service_is_ready(),
                "intent_router": self.count_publishers(self._routed_topic) > 0,
                "tts": self._tts_client.server_is_ready(),
                "vlm": self._vlm_client.server_is_ready(),
            }
            base_ready = (
                dependencies["intent_router"]
                and dependencies["tts"]
                and dependencies["vlm"]
            )
            mode_readiness: dict[str, bool] = {}
            for mode in ROUTE_MODES:
                requirements = self._context_policy.requirements_for(mode)
                mode_readiness[mode] = bool(
                    base_ready
                    and (
                        not requirements.needs_task_memory
                        or dependencies["memory"]
                    )
                    and (not requirements.needs_vision or dependencies["frame"])
                )
            active_mode = active.policy.mode if active else ""
            payload = {
                "schema_version": "flzat.high_health.v2",
                "run_id": self._tasks.run_id,
                "node": self.get_name(),
                "ready": mode_readiness.get(
                    active_mode, mode_readiness["task_command"]
                ),
                "dependencies": dependencies,
                "mode_readiness": mode_readiness,
                "active_mode": active_mode,
                "active_task_id": active.task.task_id if active else "",
                "active_request_id": active.task.request_id if active else "",
                "waiting_clarification_task_id": waiting_task_id,
                "uptime_s": round(
                    (time.monotonic_ns() - self._started_ns) / 1_000_000_000.0,
                    3,
                ),
                "counters": self._telemetry.snapshot(),
            }
        self._health_pub.publish(String(data=canonical_json(payload)))

    def _speak(self, text: str) -> None:
        value = text.strip()
        if not value:
            return
        if not self._tts_client.server_is_ready():
            self.get_logger().warn("TTS unavailable; speech was not played.")
            return
        goal = TextToSpeech.Goal()
        goal.text = value
        self._tts_client.send_goal_async(goal)

    def _remember_turn(self, runtime: RuntimeRequest, assistant: str) -> None:
        if not assistant.strip():
            return
        try:
            self._conversation.append(
                runtime.policy.mode,
                runtime.task.latest_transcript,
                assistant,
            )
        except ValueError as exc:
            self.get_logger().warn(f"Conversation turn was not stored: {exc}")

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
                "mode": runtime.policy.mode,
                "transcript": task.latest_transcript,
            },
        )
        self._request_writer.write_json(
            task.run_id,
            task.task_id,
            task.request_id,
            "router.json",
            runtime.router_record,
        )
        self._request_writer.write_json(
            task.run_id,
            task.task_id,
            task.request_id,
            "context_policy.json",
            {
                "schema_version": "flzat.context_policy.v1",
                **runtime.policy.as_dict(),
            },
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
        )

    def _text_parameter(self, name: str) -> str:
        return str(self.get_parameter(name).value).strip()

    def destroy_node(self) -> None:
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
