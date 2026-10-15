from dataclasses import replace
from pathlib import Path

import pytest

from flzat_memory.storage.memory_record import MapPosition, embedding_text, new_record
from flzat_memory.storage.readonly_memory import CompletedManifest
from flzat_memory.retrieval.search import (
    InMemoryReadOnlySearchStore,
    MemoryQuery,
    RetrievalError,
    SemanticRetriever,
)
from flzat_memory.retrieval.types import StoredMemory
from test_memory_record import observation


class QueryEmbedder:
    model_name = "test"
    revision = "test"
    dimension = 3

    def encode(self, text):
        assert text
        return (1.0, 0.0, 0.0)


def item(object_id, vector, **overrides):
    record = new_record(observation(**overrides), object_id, f"images/{object_id}.jpg")
    return StoredMemory(record, vector, embedding_text(record))


def retriever(items):
    manifest = CompletedManifest(
        environment_id="hotel_demo",
        map_id="map-session",
        map_yaml=Path("hotel.yaml"),
        map_checksum="0" * 64,
        database_path=Path("chroma"),
        object_count=len(items),
        created_at_utc="2026-08-27T10:00:00+00:00",
    )
    return SemanticRetriever(
        InMemoryReadOnlySearchStore(items), QueryEmbedder(), manifest
    )


def test_semantic_query_orders_results_and_caps_top_five():
    items = [
        item(f"object-{index}", (1.0 - index * 0.05, index * 0.05, 0.0))
        for index in range(6)
    ]

    hits = retriever(items).query(MemoryQuery("blue bottle", top_k=5))

    assert len(hits) == 5
    assert hits[0].stored.record.object_id == "object-0"
    assert hits[0].cosine_score == pytest.approx(1.0)


def test_scene_time_radius_and_session_filters_are_structured():
    lobby = item("lobby", (1.0, 0.0, 0.0))
    kitchen_observation = replace(
        observation(
            observation_id="kitchen-observation",
            scene="kitchen",
            observed_ros_ns=200,
            map_position=MapPosition("map", 9.0, 9.0, 0.9),
        ),
        label="coffee_mug",
    )
    kitchen_record = new_record(
        kitchen_observation, "kitchen", "images/kitchen.jpg"
    )
    kitchen = StoredMemory(
        kitchen_record, (0.9, 0.1, 0.0), embedding_text(kitchen_record)
    )
    engine = retriever([lobby, kitchen])

    hits = engine.query(
        MemoryQuery(
            "object",
            scene="hotel_lobby",
            start_ros_ns=100,
            end_ros_ns=150_000_000_000,
            center=MapPosition("map", 5.4, 3.2, 0.9),
            radius_m=0.5,
        )
    )

    assert [hit.stored.record.object_id for hit in hits] == ["lobby"]
    assert engine.query(MemoryQuery("object", map_id="another-map")) == []


def test_read_only_store_exposes_no_mutation_and_rejects_bad_query():
    store = InMemoryReadOnlySearchStore([])
    assert not hasattr(store, "upsert")
    assert not hasattr(store, "delete")

    with pytest.raises(RetrievalError, match="top_k"):
        retriever([]).query(MemoryQuery("object", top_k=6))
    with pytest.raises(RetrievalError, match="cannot be empty"):
        retriever([]).query(MemoryQuery("   "))
    with pytest.raises(RetrievalError, match="finite coordinates"):
        retriever([]).query(
            MemoryQuery(
                "object",
                center=MapPosition("map", float("nan"), 0.0, 0.0),
                radius_m=1.0,
            )
        )


def test_retriever_rejects_manifest_database_count_mismatch():
    manifest = CompletedManifest(
        environment_id="hotel_demo",
        map_id="map-session",
        map_yaml=Path("hotel.yaml"),
        map_checksum="0" * 64,
        database_path=Path("chroma"),
        object_count=1,
        created_at_utc="2026-08-27T10:00:00+00:00",
    )

    with pytest.raises(RuntimeError, match="object count"):
        SemanticRetriever(InMemoryReadOnlySearchStore([]), QueryEmbedder(), manifest)
