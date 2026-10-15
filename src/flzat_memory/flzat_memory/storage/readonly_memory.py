"""Validate completed memory for read-only consumers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


MANIFEST_SCHEMA = "environment_memory.manifest.v1"


class ReadOnlyMemoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompletedManifest:
    environment_id: str
    map_id: str
    map_yaml: Path
    map_checksum: str
    database_path: Path
    object_count: int
    created_at_utc: str


def load_completed_manifest(
    environment_root: Path,
    expected_environment_id: str = "",
    expected_map_id: str = "",
) -> CompletedManifest:
    """Validate a completed map/database binding without writing any files."""
    root = environment_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ReadOnlyMemoryError(f"completed manifest not found under {root}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadOnlyMemoryError(f"cannot read manifest: {exc}") from exc

    required = {
        "schema_version",
        "environment_id",
        "map_id",
        "status",
        "map_yaml",
        "map_checksum",
        "database_path",
        "object_count",
        "created_at_utc",
    }
    if set(payload) != required:
        raise ReadOnlyMemoryError("manifest fields do not match Version 1 schema")
    if payload["schema_version"] != MANIFEST_SCHEMA:
        raise ReadOnlyMemoryError("unsupported manifest schema")
    if payload["status"] != "complete":
        raise ReadOnlyMemoryError("assistant requires a complete manifest")

    environment_id = _safe_id(payload["environment_id"], "environment_id")
    map_id = _safe_id(payload["map_id"], "map_id")
    if expected_environment_id and environment_id != expected_environment_id:
        raise ReadOnlyMemoryError("manifest environment_id does not match")
    if expected_map_id and map_id != expected_map_id:
        raise ReadOnlyMemoryError("manifest map_id does not match")
    object_count = payload["object_count"]
    if isinstance(object_count, bool) or not isinstance(object_count, int):
        raise ReadOnlyMemoryError("manifest object_count must be an integer")
    if object_count < 0:
        raise ReadOnlyMemoryError("manifest object_count cannot be negative")

    map_yaml = _contained_path(root, payload["map_yaml"], "map_yaml")
    database_path = _contained_path(
        root, payload["database_path"], "database_path"
    )
    if not map_yaml.is_file():
        raise ReadOnlyMemoryError(f"manifest map does not exist: {map_yaml}")
    if not database_path.is_dir():
        raise ReadOnlyMemoryError(
            f"manifest database does not exist: {database_path}"
        )
    checksum = payload["map_checksum"]
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ReadOnlyMemoryError("manifest map checksum is invalid")
    actual_checksum = hashlib.sha256(map_yaml.read_bytes()).hexdigest()
    if actual_checksum != checksum:
        raise ReadOnlyMemoryError("saved map checksum does not match manifest")

    created_at = payload["created_at_utc"]
    if not isinstance(created_at, str) or not created_at.strip():
        raise ReadOnlyMemoryError("manifest created_at_utc is invalid")
    return CompletedManifest(
        environment_id=environment_id,
        map_id=map_id,
        map_yaml=map_yaml,
        map_checksum=checksum,
        database_path=database_path,
        object_count=object_count,
        created_at_utc=created_at,
    )


def _safe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ReadOnlyMemoryError(f"manifest {name} is invalid")
    return value


def _contained_path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReadOnlyMemoryError(f"manifest {name} is empty")
    path = (root / value).resolve()
    if path == root or root not in path.parents:
        raise ReadOnlyMemoryError(f"manifest {name} escapes environment root")
    return path
