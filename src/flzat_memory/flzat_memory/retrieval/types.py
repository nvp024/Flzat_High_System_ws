"""Small immutable types shared by read-only retrieval adapters."""

from __future__ import annotations

from dataclasses import dataclass

from flzat_memory.storage.memory_record import MemoryRecord


COLLECTION_NAME = "environment_objects_v1"


@dataclass(frozen=True)
class StoredMemory:
    record: MemoryRecord
    embedding: tuple[float, ...]
    document: str
