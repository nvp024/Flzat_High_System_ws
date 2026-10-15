"""Read-only facade over one completed FLZAT semantic-memory environment."""

from __future__ import annotations

from pathlib import Path

from flzat_memory.retrieval.search import (
    MemoryQuery,
    ReadOnlyChromaMemoryStore,
    SearchHit,
    SemanticRetriever,
)
from flzat_memory.storage.embedding import Embedder, SentenceTransformerEmbedder
from flzat_memory.storage.memory_record import (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
)
from flzat_memory.storage.readonly_memory import (
    CompletedManifest,
    load_completed_manifest,
)


class RagMemoryReader:
    """Validate, open and query an immutable Chroma environment.

    The cognitive manager depends on this facade only through the ROS query
    service. Tests may inject an embedder, while production uses the pinned
    sentence-transformer model.
    """

    def __init__(
        self,
        environment_root: Path,
        environment_id: str,
        map_id: str = "",
        *,
        embedder: Embedder | None = None,
        embedding_device: str = "cpu",
        embedding_local_files_only: bool = False,
    ) -> None:
        self.manifest: CompletedManifest = load_completed_manifest(
            environment_root,
            expected_environment_id=environment_id,
            expected_map_id=map_id,
        )
        resolved_embedder = embedder or SentenceTransformerEmbedder(
            EMBEDDING_MODEL,
            EMBEDDING_MODEL_REVISION,
            device=embedding_device,
            local_files_only=embedding_local_files_only,
        )
        self.store = ReadOnlyChromaMemoryStore(self.manifest.database_path)
        self.retriever = SemanticRetriever(
            self.store,
            resolved_embedder,
            self.manifest,
        )

    @classmethod
    def from_storage_root(
        cls,
        storage_root: Path,
        environment_id: str,
        map_id: str = "",
        **kwargs,
    ) -> "RagMemoryReader":
        return cls(
            (storage_root.expanduser() / environment_id).resolve(),
            environment_id,
            map_id,
            **kwargs,
        )

    def query(self, query: MemoryQuery) -> list[SearchHit]:
        """Return validated semantic hits for one query."""
        return self.retriever.query(query)

    @property
    def object_count(self) -> int:
        return self.store.count()
