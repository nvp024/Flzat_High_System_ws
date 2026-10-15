import hashlib
import json

import pytest

from flzat_memory.storage.readonly_memory import (
    ReadOnlyMemoryError,
    load_completed_manifest,
)


def completed_environment(tmp_path, status="complete"):
    root = tmp_path / "hotel_demo"
    maps = root / "maps"
    database = root / "chroma"
    maps.mkdir(parents=True)
    database.mkdir()
    map_yaml = maps / "hotel.yaml"
    map_yaml.write_text("image: hotel.pgm\n", encoding="utf-8")
    payload = {
        "schema_version": "environment_memory.manifest.v1",
        "environment_id": "hotel_demo",
        "map_id": "map-session",
        "status": status,
        "map_yaml": "maps/hotel.yaml",
        "map_checksum": hashlib.sha256(map_yaml.read_bytes()).hexdigest(),
        "database_path": "chroma",
        "object_count": 2,
        "created_at_utc": "2026-08-27T10:00:00+00:00",
    }
    (root / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return root, map_yaml


def test_completed_manifest_is_validated_without_writes(tmp_path):
    root, map_yaml = completed_environment(tmp_path)
    before = (root / "manifest.json").read_bytes()

    manifest = load_completed_manifest(root, "hotel_demo", "map-session")

    assert manifest.map_yaml == map_yaml.resolve()
    assert manifest.database_path == (root / "chroma").resolve()
    assert manifest.object_count == 2
    assert (root / "manifest.json").read_bytes() == before


def test_incomplete_or_checksum_mismatch_is_rejected(tmp_path):
    root, map_yaml = completed_environment(tmp_path, status="incomplete")
    with pytest.raises(ReadOnlyMemoryError, match="complete manifest"):
        load_completed_manifest(root)

    root, map_yaml = completed_environment(tmp_path / "other")
    map_yaml.write_text("changed: true\n", encoding="utf-8")
    with pytest.raises(ReadOnlyMemoryError, match="checksum"):
        load_completed_manifest(root)
