"""Config-driven safety guard and semantic intent routing for FLZAT."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Protocol, Sequence
import unicodedata

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import yaml

from flzat_high_interfaces.msg import RoutedCommand
from flzat_memory.storage.embedding import (
    DeterministicTestEmbedder,
    SentenceTransformerEmbedder,
)


ROUTE_MODES = (
    "task_command",
    "visual_qa",
    "general_qa",
    "social_dialogue",
    "clarification",
)
SAFETY_MODE = "safety_stop"
_FLAG_NAMES = (
    "needs_vision",
    "needs_task_memory",
    "needs_conversation_memory",
    "needs_user_profile",
    "needs_external_knowledge",
)


class RouterConfigError(ValueError):
    pass


class RouterEmbedder(Protocol):
    def encode(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class RoutePolicy:
    mode: str
    examples: tuple[str, ...]
    needs_vision: bool
    needs_task_memory: bool
    needs_conversation_memory: bool
    needs_user_profile: bool
    needs_external_knowledge: bool


@dataclass(frozen=True)
class IntentRouterConfig:
    router_version: str
    model_name: str
    model_revision: str
    embedding_device: str
    embedding_local_files_only: bool
    minimum_confidence: float
    minimum_margin: float
    exact_safety_phrases: frozenset[str]
    safety_patterns: tuple[str, ...]
    emergency_terms: tuple[str, ...]
    routes: tuple[RoutePolicy, ...]

    def policy(self, mode: str) -> RoutePolicy:
        for route in self.routes:
            if route.mode == mode:
                return route
        raise KeyError(mode)


@dataclass(frozen=True)
class RoutedDecision:
    router_version: str
    utterance: str
    mode: str
    confidence: float
    language: str
    urgency: str
    needs_vision: bool
    needs_task_memory: bool
    needs_conversation_memory: bool
    needs_user_profile: bool
    needs_external_knowledge: bool
    route_names: tuple[str, ...]
    route_scores: tuple[float, ...]
    latency_ms: float

    def as_log_value(self) -> dict[str, Any]:
        return {
            "router_version": self.router_version,
            "utterance": self.utterance,
            "mode": self.mode,
            "confidence": round(self.confidence, 6),
            "language": self.language,
            "urgency": self.urgency,
            "needs_vision": self.needs_vision,
            "needs_task_memory": self.needs_task_memory,
            "needs_conversation_memory": self.needs_conversation_memory,
            "needs_user_profile": self.needs_user_profile,
            "needs_external_knowledge": self.needs_external_knowledge,
            "scores": {
                name: round(score, 6)
                for name, score in zip(self.route_names, self.route_scores)
            },
            "latency_ms": round(self.latency_ms, 3),
        }


def _normalized_utterance(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RouterConfigError(f"{name} must be a nonempty string")
    return value.strip()


def _required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RouterConfigError(f"{name} must be a boolean")
    return value


def _finite_threshold(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RouterConfigError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise RouterConfigError(f"{name} must be between {minimum} and {maximum}")
    return result


def load_router_config(path: str | Path) -> IntentRouterConfig:
    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RouterConfigError(f"cannot load router config {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RouterConfigError("router config must be a YAML mapping")

    embedding = raw.get("embedding")
    thresholds = raw.get("thresholds")
    safety = raw.get("safety")
    route_values = raw.get("routes")
    for value, name in (
        (embedding, "embedding"),
        (thresholds, "thresholds"),
        (safety, "safety"),
        (route_values, "routes"),
    ):
        if not isinstance(value, dict):
            raise RouterConfigError(f"{name} must be a mapping")

    if tuple(route_values) != ROUTE_MODES:
        raise RouterConfigError(
            "routes must contain task_command, visual_qa, general_qa, "
            "social_dialogue and clarification in canonical order"
        )

    routes: list[RoutePolicy] = []
    seen_examples: set[str] = set()
    for mode in ROUTE_MODES:
        route = route_values[mode]
        if not isinstance(route, dict):
            raise RouterConfigError(f"routes.{mode} must be a mapping")
        examples = route.get("examples")
        if not isinstance(examples, list) or len(examples) < 2:
            raise RouterConfigError(f"routes.{mode}.examples requires at least two values")
        clean_examples = tuple(
            _required_text(value, f"routes.{mode}.examples") for value in examples
        )
        for example in clean_examples:
            normalized = _normalized_utterance(example)
            if normalized in seen_examples:
                raise RouterConfigError(f"duplicate normalized route example: {example}")
            seen_examples.add(normalized)
        flags = {
            name: _required_bool(route.get(name), f"routes.{mode}.{name}")
            for name in _FLAG_NAMES
        }
        routes.append(RoutePolicy(mode=mode, examples=clean_examples, **flags))

    exact = safety.get("exact_phrases")
    patterns = safety.get("command_patterns")
    emergency = safety.get("emergency_terms")
    if not isinstance(exact, list) or not exact:
        raise RouterConfigError("safety.exact_phrases must be a nonempty list")
    if not isinstance(patterns, list):
        raise RouterConfigError("safety.command_patterns must be a list")
    if not isinstance(emergency, list):
        raise RouterConfigError("safety.emergency_terms must be a list")
    exact_phrases = frozenset(
        _normalized_utterance(_required_text(value, "safety.exact_phrases"))
        for value in exact
    )
    safety_patterns = tuple(
        _required_text(value, "safety.command_patterns") for value in patterns
    )
    try:
        for pattern in safety_patterns:
            re.compile(pattern, flags=re.IGNORECASE)
    except re.error as exc:
        raise RouterConfigError(f"invalid safety command pattern: {exc}") from exc

    return IntentRouterConfig(
        router_version=_required_text(raw.get("router_version"), "router_version"),
        model_name=_required_text(embedding.get("model_name"), "embedding.model_name"),
        model_revision=_required_text(
            embedding.get("revision"), "embedding.revision"
        ),
        embedding_device=_required_text(embedding.get("device"), "embedding.device"),
        embedding_local_files_only=_required_bool(
            embedding.get("local_files_only"), "embedding.local_files_only"
        ),
        minimum_confidence=_finite_threshold(
            thresholds.get("minimum_confidence"),
            "thresholds.minimum_confidence",
            -1.0,
            1.0,
        ),
        minimum_margin=_finite_threshold(
            thresholds.get("minimum_margin"),
            "thresholds.minimum_margin",
            0.0,
            2.0,
        ),
        exact_safety_phrases=exact_phrases,
        safety_patterns=safety_patterns,
        emergency_terms=tuple(
            _normalized_utterance(_required_text(value, "safety.emergency_terms"))
            for value in emergency
        ),
        routes=tuple(routes),
    )


class SemanticIntentRouter:
    """Cache route examples once and classify one normalized utterance at a time."""

    def __init__(self, config: IntentRouterConfig, embedder: RouterEmbedder) -> None:
        self.config = config
        self._embedder = embedder
        self._patterns = tuple(
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in config.safety_patterns
        )
        self._example_vectors: dict[str, tuple[tuple[float, ...], ...]] = {}
        dimension = 0
        for route in config.routes:
            vectors = tuple(
                embedder.encode(_normalized_utterance(example))
                for example in route.examples
            )
            if not vectors:
                raise RouterConfigError(f"route {route.mode} has no cached embeddings")
            for vector in vectors:
                if not vector or not all(math.isfinite(value) for value in vector):
                    raise RouterConfigError("route embeddings must be finite and nonempty")
                if dimension == 0:
                    dimension = len(vector)
                elif len(vector) != dimension:
                    raise RouterConfigError("route embedding dimensions do not match")
            self._example_vectors[route.mode] = vectors
        self.embedding_dimension = dimension
        self.cached_example_count = sum(
            len(vectors) for vectors in self._example_vectors.values()
        )

    def route(self, utterance: str) -> RoutedDecision:
        started = time.perf_counter_ns()
        value = utterance.strip()
        normalized = _normalized_utterance(value)
        if not normalized:
            raise ValueError("utterance must contain alphanumeric text")

        route_names = tuple(route.mode for route in self.config.routes)
        if self._is_safety(normalized):
            zeros = tuple(0.0 for _ in route_names)
            return self._decision(
                value,
                SAFETY_MODE,
                1.0,
                route_names,
                zeros,
                started,
            )

        # Apply the same canonicalization used for cached examples.  Besides
        # improving production consistency, this makes punctuation/case-only
        # variants of a configured command route identically.
        vector = self._embedder.encode(normalized)
        if len(vector) != self.embedding_dimension:
            raise ValueError("utterance embedding dimension does not match route cache")
        scores = tuple(
            max(self._dot(vector, example) for example in self._example_vectors[mode])
            for mode in route_names
        )
        ranked = sorted(
            zip(route_names, scores), key=lambda item: (-item[1], item[0])
        )
        best_mode, best_score = ranked[0]
        second_score = ranked[1][1]
        if (
            best_score < self.config.minimum_confidence
            or best_score - second_score < self.config.minimum_margin
        ):
            # "clarification" is a request for semantic fallback, not permission
            # to ask about an object. The manager resolves it through Qwen.
            selected_mode = "clarification"
        else:
            selected_mode = best_mode
        return self._decision(
            value,
            selected_mode,
            best_score,
            route_names,
            scores,
            started,
        )

    def _decision(
        self,
        utterance: str,
        mode: str,
        confidence: float,
        route_names: tuple[str, ...],
        route_scores: tuple[float, ...],
        started_ns: int,
    ) -> RoutedDecision:
        if mode == SAFETY_MODE:
            flags = {name: False for name in _FLAG_NAMES}
            urgency = self._safety_urgency(_normalized_utterance(utterance))
        else:
            policy = self.config.policy(mode)
            flags = {name: getattr(policy, name) for name in _FLAG_NAMES}
            urgency = "normal"
        return RoutedDecision(
            router_version=self.config.router_version,
            utterance=utterance,
            mode=mode,
            confidence=float(confidence),
            language=self._language(utterance),
            urgency=urgency,
            route_names=route_names,
            route_scores=tuple(float(value) for value in route_scores),
            latency_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
            **flags,
        )

    def _is_safety(self, normalized: str) -> bool:
        return normalized in self.config.exact_safety_phrases or any(
            pattern.fullmatch(normalized) is not None for pattern in self._patterns
        )

    def _safety_urgency(self, normalized: str) -> str:
        if any(term and term in normalized for term in self.config.emergency_terms):
            return "emergency"
        return "high"

    @staticmethod
    def _language(value: str) -> str:
        return "vi" if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", value.casefold()) else "en"

    @staticmethod
    def _dot(left: Sequence[float], right: Sequence[float]) -> float:
        return float(sum(a * b for a, b in zip(left, right)))


class IntentRouterNode(Node):
    def __init__(self) -> None:
        super().__init__("intent_router")
        self.declare_parameter("config_file", "")
        self.declare_parameter("embedding_backend", "sentence_transformer")
        self.declare_parameter("embedding_device", "")
        self.declare_parameter("embedding_local_files_only", False)
        self.declare_parameter("enable_text_command", False)

        config_file = str(self.get_parameter("config_file").value).strip()
        if not config_file:
            from ament_index_python.packages import get_package_share_directory

            config_file = str(
                Path(get_package_share_directory("flzat_cognitive_pipeline"))
                / "config"
                / "intent_routes.yaml"
            )
        config = load_router_config(config_file)
        backend = str(self.get_parameter("embedding_backend").value).strip()
        device_override = str(self.get_parameter("embedding_device").value).strip()
        local_only = bool(self.get_parameter("embedding_local_files_only").value)
        if backend == "sentence_transformer":
            embedder: RouterEmbedder = SentenceTransformerEmbedder(
                config.model_name,
                config.model_revision,
                device=device_override or config.embedding_device,
                local_files_only=(
                    local_only or config.embedding_local_files_only
                ),
            )
        elif backend == "deterministic_test":
            embedder = DeterministicTestEmbedder()
        else:
            raise ValueError(
                "embedding_backend must be sentence_transformer or deterministic_test"
            )
        started = time.perf_counter()
        self._router = SemanticIntentRouter(config, embedder)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._routed_pub = self.create_publisher(
            RoutedCommand, "/flzat/high_system/routed_command", qos
        )
        self._cancel_pub = self.create_publisher(
            String, "/flzat/high_system/cancel", qos
        )
        self._stt_sub = self.create_subscription(
            String,
            "/flzat/high_system/stt_result",
            self._on_utterance,
            qos,
        )
        self._text_sub = None
        if bool(self.get_parameter("enable_text_command").value):
            self._text_sub = self.create_subscription(
                String,
                "/flzat/high_system/text_command",
                self._on_utterance,
                qos,
            )
        self.get_logger().info(
            "Intent Router ready: "
            f"version={config.router_version}, backend={backend}, "
            f"examples={self._router.cached_example_count}, "
            f"startup={time.perf_counter() - started:.3f} s."
        )

    def _on_utterance(self, message: String) -> None:
        try:
            decision = self._router.route(message.data)
            routed = RoutedCommand()
            routed.router_version = decision.router_version
            routed.utterance = decision.utterance
            routed.mode = decision.mode
            routed.confidence = decision.confidence
            routed.language = decision.language
            routed.urgency = decision.urgency
            routed.needs_vision = decision.needs_vision
            routed.needs_task_memory = decision.needs_task_memory
            routed.needs_conversation_memory = decision.needs_conversation_memory
            routed.needs_user_profile = decision.needs_user_profile
            routed.needs_external_knowledge = decision.needs_external_knowledge
            routed.route_names = list(decision.route_names)
            routed.route_scores = list(decision.route_scores)
            routed.latency_ms = decision.latency_ms
            self._routed_pub.publish(routed)
            serialized = json.dumps(
                decision.as_log_value(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            self.get_logger().info(f"[ROUTE] {decision.mode}")
            self.get_logger().debug(f"ROUTED_COMMAND {serialized}")
            if decision.mode == SAFETY_MODE:
                self._cancel_pub.publish(String(data=""))
        except Exception as exc:
            self.get_logger().error(f"Intent routing failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = IntentRouterNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
