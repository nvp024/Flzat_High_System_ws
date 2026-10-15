"""Bound and freeze untrusted retrieval results into trusted context snapshots."""

from __future__ import annotations

import math

from flzat_cognitive_pipeline.domain import (
    ContextItem,
    ContextSnapshot,
    MemoryCandidate,
)


class ContextRagManager:
    def __init__(
        self,
        *,
        max_items: int = 5,
        max_context_characters: int = 6000,
        minimum_score: float = -1.0,
    ) -> None:
        if not 1 <= max_items <= 20:
            raise ValueError("max_items must be between 1 and 20")
        if max_context_characters < 256:
            raise ValueError("max_context_characters must be at least 256")
        if not math.isfinite(minimum_score) or not -1.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be finite and between -1 and 1")
        self.max_items = max_items
        self.max_context_characters = max_context_characters
        self.minimum_score = minimum_score

    def build_snapshot(
        self,
        request_id: str,
        environment_id: str,
        map_id: str,
        candidates: list[MemoryCandidate],
    ) -> ContextSnapshot:
        if not request_id.strip():
            raise ValueError("request_id cannot be empty")
        if not environment_id.strip() or not map_id.strip():
            raise ValueError("environment_id and map_id cannot be empty")

        best_by_id: dict[str, MemoryCandidate] = {}
        for candidate in candidates:
            self._validate_candidate(candidate, environment_id, map_id)
            if candidate.score < self.minimum_score:
                continue
            current = best_by_id.get(candidate.object_id)
            if current is None or candidate.score > current.score:
                best_by_id[candidate.object_id] = candidate

        ordered = sorted(
            best_by_id.values(), key=lambda item: (-item.score, item.object_id)
        )
        items: list[ContextItem] = []
        used_characters = 0
        truncated = len(ordered) > self.max_items
        for candidate in ordered:
            if len(items) >= self.max_items:
                break
            item = ContextItem(
                context_ref=f"memory:{candidate.object_id}",
                kind="environment_object",
                object_id=candidate.object_id,
                label=candidate.label.strip(),
                description=candidate.description.strip(),
                relationships=tuple(
                    value.strip()
                    for value in candidate.relationships
                    if value.strip()
                ),
                scene=candidate.scene.strip(),
                retrieval_score=float(candidate.score),
                position=candidate.position,
            )
            item_size = len(str(item.prompt_value()))
            if items and used_characters + item_size > self.max_context_characters:
                truncated = True
                break
            if not items and item_size > self.max_context_characters:
                truncated = True
                continue
            items.append(item)
            used_characters += item_size

        return ContextSnapshot(
            request_id=request_id,
            environment_id=environment_id,
            map_id=map_id,
            items=tuple(items),
            truncated=truncated,
        )

    @staticmethod
    def ambiguous_labels(snapshot: ContextSnapshot) -> tuple[str, ...]:
        counts: dict[str, int] = {}
        for item in snapshot.items:
            key = item.label.casefold()
            counts[key] = counts.get(key, 0) + 1
        return tuple(sorted(label for label, count in counts.items() if count > 1))

    @staticmethod
    def _validate_candidate(
        candidate: MemoryCandidate,
        environment_id: str,
        map_id: str,
    ) -> None:
        if candidate.environment_id != environment_id:
            raise ValueError("candidate environment_id does not match request")
        if candidate.map_id != map_id:
            raise ValueError("candidate map_id does not match request")
        if not candidate.object_id.strip() or not candidate.label.strip():
            raise ValueError("candidate identity and label cannot be empty")
        if not math.isfinite(candidate.score) or not -1.0 <= candidate.score <= 1.0:
            raise ValueError("candidate score must be finite and between -1 and 1")
        if candidate.position is not None:
            position = candidate.position
            if position.frame_id != "map" or not all(
                math.isfinite(value) for value in (position.x, position.y, position.z)
            ):
                raise ValueError("trusted position must be finite in map frame")
