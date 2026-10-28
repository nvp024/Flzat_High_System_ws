from pathlib import Path

import pytest

from flzat_memory.storage.embedding import DeterministicTestEmbedder
from flzat_cognitive_pipeline.intent_router import (
    ROUTE_MODES,
    RouterConfigError,
    SemanticIntentRouter,
    load_router_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/intent_routes.yaml"


class CountingEmbedder:
    def __init__(self):
        self.delegate = DeterministicTestEmbedder()
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return self.delegate.encode(text)


class FlatEmbedder:
    def encode(self, _text):
        return (1.0, 0.0)


def build_router():
    config = load_router_config(CONFIG)
    embedder = CountingEmbedder()
    return config, embedder, SemanticIntentRouter(config, embedder)


def test_config_has_all_routes_and_explicit_context_flags():
    config = load_router_config(CONFIG)

    assert tuple(route.mode for route in config.routes) == ROUTE_MODES
    assert config.policy("task_command").needs_vision is True
    assert config.policy("task_command").needs_task_memory is True
    assert config.policy("visual_qa").needs_vision is True
    assert config.policy("general_qa").needs_task_memory is False
    assert config.policy("social_dialogue").needs_conversation_memory is False
    assert config.policy("clarification").needs_conversation_memory is True


def test_examples_are_cached_once_and_only_utterance_is_encoded_per_route():
    _config, embedder, router = build_router()
    startup_calls = embedder.calls

    assert startup_calls == router.cached_example_count
    assert router.route("get the book on the table").mode == "task_command"
    assert embedder.calls == startup_calls + 1
    assert router.route("what is on the table").mode == "visual_qa"
    assert embedder.calls == startup_calls + 2


def test_exact_examples_cover_all_semantic_modes():
    config, _embedder, router = build_router()

    for policy in config.routes:
        decision = router.route(policy.examples[0])
        assert decision.mode == policy.mode
        assert decision.confidence > 0.99
        assert len(decision.route_names) == len(decision.route_scores)


def test_route_normalization_ignores_case_and_terminal_punctuation():
    _config, _embedder, router = build_router()

    decision = router.route("What is on the table?!")

    assert decision.mode == "visual_qa"
    assert decision.confidence > 0.99


def test_safety_guard_precedes_embeddings_and_sets_no_context_flags():
    _config, embedder, router = build_router()
    calls = embedder.calls

    decision = router.route("Emergency stop!")

    assert decision.mode == "safety_stop"
    assert decision.urgency == "emergency"
    assert decision.needs_vision is False
    assert decision.needs_task_memory is False
    assert decision.needs_conversation_memory is False
    assert decision.needs_user_profile is False
    assert decision.needs_external_knowledge is False
    assert embedder.calls == calls


def test_safety_word_inside_a_question_is_not_an_exact_stop_command():
    _config, embedder, router = build_router()
    calls = embedder.calls

    decision = router.route("What does stop mean?")

    assert decision.mode != "safety_stop"
    assert embedder.calls == calls + 1


def test_unrelated_low_similarity_text_falls_back_to_clarification():
    _config, _embedder, router = build_router()

    decision = router.route("quartz nebula xylophone 84729")

    assert decision.mode == "clarification"
    assert decision.needs_conversation_memory is True


def test_tied_top_scores_fall_back_to_clarification():
    config = load_router_config(CONFIG)
    router = SemanticIntentRouter(config, FlatEmbedder())

    decision = router.route("please perform an ambiguous request")

    assert decision.mode == "clarification"
    assert decision.confidence == 1.0


def test_invalid_yaml_fails_fast(tmp_path):
    config_path = tmp_path / "intent_routes.yaml"
    config_path.write_text("routes: [not, a, mapping]\n", encoding="utf-8")

    with pytest.raises(RouterConfigError):
        load_router_config(config_path)
