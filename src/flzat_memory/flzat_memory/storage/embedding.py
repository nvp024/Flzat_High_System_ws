"""Provide the pinned multilingual sentence-embedding boundary."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol, Sequence

from flzat_memory.storage.memory_record import EMBEDDING_DIMENSION


class Embedder(Protocol):
    model_name: str
    revision: str
    dimension: int

    def encode(self, text: str) -> tuple[float, ...]: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is unavailable; install the workspace requirements.txt"
            ) from exc
        self.model_name = model_name
        self.revision = revision
        self.dimension = EMBEDDING_DIMENSION
        model_source, source_is_local = _resolve_model_source(
            model_name,
            revision,
            local_files_only=local_files_only,
        )
        load_kwargs = {
            "device": device,
            "trust_remote_code": False,
            "local_files_only": local_files_only,
        }
        # Passing a repository ID to Transformers 4.57 may still trigger a
        # Hub model_info() request while constructing the tokenizer, even when
        # local_files_only=True.  A concrete snapshot path is fully offline and
        # already identifies the requested revision.
        if not source_is_local:
            load_kwargs["revision"] = revision
        self._model = SentenceTransformer(
            model_source,
            **load_kwargs,
        )
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension != self.dimension:
            raise RuntimeError(
                f"embedding dimension mismatch: expected {self.dimension}, got {dimension}"
            )

    def encode(self, text: str) -> tuple[float, ...]:
        vector = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        result = tuple(float(value) for value in vector)
        _validate_embedding(result, self.dimension)
        return result


def _resolve_model_source(
    model_name: str,
    revision: str,
    *,
    local_files_only: bool,
) -> tuple[str, bool]:
    """Resolve an offline Hub model ID to its cached snapshot directory."""
    local_path = Path(model_name).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve()), True
    if not local_files_only:
        return model_name, False
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=model_name,
            revision=revision,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "embedding model is not available in the local Hugging Face cache: "
            f"{model_name}@{revision}"
        ) from exc
    return str(Path(snapshot).resolve()), True


class DeterministicTestEmbedder:
    """Dependency-free 384-D embedder for contract/integration tests only."""

    model_name = "flzat/test-sha256-embedding"
    revision = "v1"
    dimension = EMBEDDING_DIMENSION

    def encode(self, text: str) -> tuple[float, ...]:
        value = text.strip()
        if not value:
            raise ValueError("embedding text cannot be empty")
        raw = bytearray()
        digest_size = hashlib.sha256().digest_size
        for index in range(self.dimension // digest_size):
            raw.extend(
                hashlib.sha256(f"{index}:{value}".encode("utf-8")).digest()
            )
        vector = [((item / 255.0) * 2.0) - 1.0 for item in raw]
        norm = math.sqrt(sum(item * item for item in vector))
        result = tuple(item / norm for item in vector)
        _validate_embedding(result, self.dimension)
        return result


def _validate_embedding(values: Sequence[float], dimension: int) -> None:
    if len(values) != dimension or not all(math.isfinite(value) for value in values):
        raise ValueError(f"embedding must contain {dimension} finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("embedding must be L2-normalized")
