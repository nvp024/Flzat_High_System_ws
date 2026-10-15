import importlib.util

import pytest

from flzat_memory.storage.memory_record import embedding_text, new_record
from flzat_memory.retrieval.search import ReadOnlyChromaMemoryStore
from flzat_memory.retrieval.types import COLLECTION_NAME
from test_memory_record import observation


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("chromadb") is None,
    reason="chromadb is not installed on this development host",
)


def test_read_only_chroma_load_and_query(tmp_path):
    import chromadb

    vector = (1.0,) + (0.0,) * 383
    record = new_record(observation(), "object-1", "images/a.jpg")
    database = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(database))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "schema_version": "environment_memory.v1"},
    )
    collection.upsert(
        ids=[record.object_id],
        embeddings=[list(vector)],
        documents=[embedding_text(record)],
        metadatas=[{
            "record_json": record.to_json(),
            "environment_id": record.environment_id,
            "map_id": record.map_id,
            "detector_class": record.detector_class,
        }],
    )

    assert COLLECTION_NAME == "environment_objects_v1"
    read_only = ReadOnlyChromaMemoryStore(database)
    hits = read_only.search(vector, "hotel_demo", "map-session")
    assert len(hits) == 1
    assert hits[0].stored.record == record
    assert hits[0].cosine_score == pytest.approx(1.0)
    assert not hasattr(read_only, "upsert")
