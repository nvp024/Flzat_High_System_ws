"""Rank and filter semantic search results from completed memory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol, Sequence

from flzat_memory.storage.embedding import Embedder
from flzat_memory.storage.memory_record import (
    EMBEDDING_DIMENSION,
    MapPosition,
    MemoryRecord,
    embedding_text,
)
from flzat_memory.storage.readonly_memory import CompletedManifest
from flzat_memory.retrieval.types import COLLECTION_NAME, StoredMemory


MAX_RESULTS = 5


class RetrievalError(ValueError):
    pass


def _cosine_similarity(
    left: Sequence[float], right: Sequence[float]
) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embeddings must have equal nonzero dimensions")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("embeddings must have nonzero norm")
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _map_distance(left: MapPosition, right: MapPosition) -> float:
    if left.frame_id != "map" or right.frame_id != "map":
        raise ValueError("spatial query requires map-frame positions")
    return math.sqrt(
        (left.x - right.x) ** 2
        + (left.y - right.y) ** 2
        + (left.z - right.z) ** 2
    )


@dataclass(frozen=True)
class SearchHit:
    stored: StoredMemory
    cosine_score: float


@dataclass(frozen=True)
class MemoryQuery:
    text: str
    top_k: int = MAX_RESULTS
    environment_id: str = ""
    map_id: str = ""
    scene: str = ""
    start_ros_ns: int | None = None
    end_ros_ns: int | None = None
    center: MapPosition | None = None
    radius_m: float | None = None


class ReadOnlySearchStore(Protocol):
    def search(
        self,
        embedding: Sequence[float],
        environment_id: str,
        map_id: str,
    ) -> list[SearchHit]: ...

    def count(self) -> int: ...


class InMemoryReadOnlySearchStore:
    def __init__(self, items: Sequence[StoredMemory]) -> None:
        self._items = tuple(items)

    def search(
        self,
        embedding: Sequence[float],
        environment_id: str,
        map_id: str,
    ) -> list[SearchHit]:
        return sorted(
            (
                SearchHit(item, _cosine_similarity(item.embedding, embedding))
                for item in self._items
                if item.record.environment_id == environment_id
                and item.record.map_id == map_id
            ),
            key=lambda hit: (-hit.cosine_score, hit.stored.record.object_id),
        )

    def count(self) -> int:
        return len(self._items)


class ReadOnlyChromaMemoryStore:
    """Chroma query adapter exposing no mutation operation to assistant code."""

    def __init__(self, path: Path) -> None:
        if not path.is_dir():
            raise RuntimeError(f"Chroma directory does not exist: {path}")
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is unavailable; install the workspace requirements.txt"
            ) from exc
        self._client = chromadb.PersistentClient(path=str(path))
        try:
            self._collection = self._client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"required Chroma collection {COLLECTION_NAME!r} is unavailable: {exc}"
            ) from exc
        self._records = self._load_records()

    def _load_records(self) -> dict[str, MemoryRecord]:
        result = self._collection.get(include=["metadatas", "documents"])
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        documents = result.get("documents") or []
        if not (len(ids) == len(metadatas) == len(documents)):
            raise RuntimeError("Chroma collection returned inconsistent record arrays")
        records = {}
        for object_id, metadata, document in zip(ids, metadatas, documents):
            if not metadata or "record_json" not in metadata:
                raise RuntimeError(f"stored object {object_id} lacks canonical record")
            record = MemoryRecord.from_json(str(metadata["record_json"]))
            if record.object_id != object_id:
                raise RuntimeError(f"stored object ID mismatch for {object_id}")
            if (
                metadata.get("environment_id") != record.environment_id
                or metadata.get("map_id") != record.map_id
                or metadata.get("detector_class") != record.detector_class
            ):
                raise RuntimeError(
                    f"stored object {object_id} metadata does not match record"
                )
            if str(document) != embedding_text(record):
                raise RuntimeError(
                    f"stored object {object_id} document does not match record"
                )
            records[object_id] = record
        return records

    def search(
        self,
        embedding: Sequence[float],
        environment_id: str,
        map_id: str,
    ) -> list[SearchHit]:
        vector = tuple(float(value) for value in embedding)
        if len(vector) != EMBEDDING_DIMENSION or not all(
            math.isfinite(value) for value in vector
        ):
            raise RetrievalError("query embedding is invalid")
        if not self._records:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=len(self._records),
            where={
                "$and": [
                    {"environment_id": {"$eq": environment_id}},
                    {"map_id": {"$eq": map_id}},
                ]
            },
            include=["distances"],
        )
        ids_rows = result.get("ids") or [[]]
        distance_rows = result.get("distances") or [[]]
        ids = ids_rows[0] if ids_rows else []
        distances = distance_rows[0] if distance_rows else []
        if len(ids) != len(distances):
            raise RuntimeError("Chroma query returned inconsistent result arrays")
        hits = []
        for object_id, distance in zip(ids, distances):
            record = self._records.get(object_id)
            if record is None:
                raise RuntimeError(f"query returned unknown object {object_id}")
            hits.append(
                SearchHit(
                    StoredMemory(record, (), embedding_text(record)),
                    max(-1.0, min(1.0, 1.0 - float(distance))),
                )
            )
        return hits

    def count(self) -> int:
        return len(self._records)

    def records(self) -> tuple[MemoryRecord, ...]:
        """Return a stable read-only snapshot ordered by object ID."""
        return tuple(self._records[key] for key in sorted(self._records))


class SemanticRetriever:
    def __init__(
        self,
        store: ReadOnlySearchStore,
        embedder: Embedder,
        manifest: CompletedManifest,
    ) -> None:
        if store.count() != manifest.object_count:
            raise RuntimeError(
                "database object count does not match completed manifest"
            )
        self.store = store
        self.embedder = embedder
        self.manifest = manifest

    def query(self, query: MemoryQuery) -> list[SearchHit]:
        _validate_query(query)
        environment_id = query.environment_id or self.manifest.environment_id
        map_id = query.map_id or self.manifest.map_id
        if (
            environment_id != self.manifest.environment_id
            or map_id != self.manifest.map_id
        ):
            return []
        embedding = self.embedder.encode(query.text.strip())
        hits = self.store.search(embedding, environment_id, map_id)
        filtered = [hit for hit in hits if _matches(hit.stored.record, query)]
        filtered.sort(
            key=lambda hit: (-hit.cosine_score, hit.stored.record.object_id)
        )
        return filtered[: query.top_k]


def _validate_query(query: MemoryQuery) -> None:
    if not isinstance(query.text, str) or not query.text.strip():
        raise RetrievalError("query text cannot be empty")
    if len(query.text) > 512:
        raise RetrievalError("query text exceeds 512 characters")
    if isinstance(query.top_k, bool) or not 1 <= query.top_k <= MAX_RESULTS:
        raise RetrievalError(f"top_k must be between 1 and {MAX_RESULTS}")
    if query.start_ros_ns is not None and query.start_ros_ns < 0:
        raise RetrievalError("start time cannot be negative")
    if query.end_ros_ns is not None and query.end_ros_ns < 0:
        raise RetrievalError("end time cannot be negative")
    if (
        query.start_ros_ns is not None
        and query.end_ros_ns is not None
        and query.start_ros_ns > query.end_ros_ns
    ):
        raise RetrievalError("start time must not follow end time")
    if (query.center is None) != (query.radius_m is None):
        raise RetrievalError("center and radius must be provided together")
    if query.radius_m is not None:
        if not math.isfinite(query.radius_m) or query.radius_m <= 0.0:
            raise RetrievalError("radius must be finite and positive")
        if query.center.frame_id != "map":
            raise RetrievalError("radius center must use map frame")
        if not all(
            math.isfinite(value)
            for value in (query.center.x, query.center.y, query.center.z)
        ):
            raise RetrievalError("radius center must contain finite coordinates")


def _matches(record: MemoryRecord, query: MemoryQuery) -> bool:
    if query.scene and record.scene != query.scene:
        return False
    if query.start_ros_ns is not None and record.last_seen_ros_ns < query.start_ros_ns:
        return False
    if query.end_ros_ns is not None and record.last_seen_ros_ns > query.end_ros_ns:
        return False
    if query.center is not None and query.radius_m is not None:
        if _map_distance(record.map_position, query.center) > query.radius_m:
            return False
    return True
