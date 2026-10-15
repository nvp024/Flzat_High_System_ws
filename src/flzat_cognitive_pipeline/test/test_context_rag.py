import pytest

from flzat_cognitive_pipeline.context_rag import ContextRagManager
from flzat_cognitive_pipeline.domain import MemoryCandidate, TrustedPosition


def candidate(object_id, score, label="cup", environment="demo", map_id="map"):
    return MemoryCandidate(
        object_id=object_id,
        environment_id=environment,
        map_id=map_id,
        label=label,
        description=f"{label} description",
        relationships=("on:table",),
        scene="kitchen",
        score=score,
        position=TrustedPosition("map", 1.0, 2.0, 0.8),
    )


def test_context_is_deduplicated_ranked_bounded_and_immutable():
    manager = ContextRagManager(max_items=2, max_context_characters=2000)
    snapshot = manager.build_snapshot(
        "req-1",
        "demo",
        "map",
        [candidate("b", 0.6), candidate("a", 0.8), candidate("a", 0.9)],
    )

    assert snapshot.refs == ("memory:a", "memory:b")
    assert snapshot.items[0].retrieval_score == 0.9
    assert snapshot.truncated is False
    assert "x" not in snapshot.items[0].prompt_value()
    assert snapshot.items[0].prompt_value()["has_position"] is True


def test_context_budget_and_score_filter_mark_truncation():
    manager = ContextRagManager(
        max_items=1, max_context_characters=2000, minimum_score=0.5
    )
    snapshot = manager.build_snapshot(
        "req-1", "demo", "map", [candidate("a", 0.9), candidate("b", 0.8)]
    )
    assert snapshot.refs == ("memory:a",)
    assert snapshot.truncated is True


def test_ambiguity_and_unknown_ref_are_explicit():
    manager = ContextRagManager()
    snapshot = manager.build_snapshot(
        "req-1",
        "demo",
        "map",
        [candidate("a", 0.9, "bottle"), candidate("b", 0.8, "Bottle")],
    )
    assert manager.ambiguous_labels(snapshot) == ("bottle",)
    with pytest.raises(KeyError, match="unknown context_ref"):
        snapshot.resolve("memory:missing")


def test_cross_environment_candidate_is_rejected():
    with pytest.raises(ValueError, match="environment_id"):
        ContextRagManager().build_snapshot(
            "req-1", "demo", "map", [candidate("a", 0.9, environment="other")]
        )
