from pathlib import Path

from flzat_memory.rag_reader import RagMemoryReader


def test_reader_resolves_environment_below_storage_root(monkeypatch, tmp_path):
    captured = {}

    def fake_init(self, environment_root, environment_id, map_id="", **kwargs):
        captured.update(
            root=environment_root,
            environment_id=environment_id,
            map_id=map_id,
            kwargs=kwargs,
        )

    monkeypatch.setattr(RagMemoryReader, "__init__", fake_init)

    RagMemoryReader.from_storage_root(
        tmp_path,
        "hotel_demo",
        "map_session",
        embedding_device="cpu",
    )

    assert captured["root"] == (tmp_path / "hotel_demo").resolve()
    assert captured["environment_id"] == "hotel_demo"
    assert captured["map_id"] == "map_session"
    assert captured["kwargs"] == {"embedding_device": "cpu"}


def test_reader_exposes_no_mutation_api():
    assert not hasattr(RagMemoryReader, "upsert")
    assert not hasattr(RagMemoryReader, "delete")
