"""Idempotent High-to-Low handoff artifacts with atomic writes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any


class ArtifactConflictError(RuntimeError):
    pass


class HandoffArtifactWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.expanduser().resolve()
        self._lock = threading.Lock()

    def write(
        self,
        run_id: str,
        task_id: str,
        handoff_id: str,
        payload: str,
    ) -> Path:
        self._validate_id(run_id, "run_id")
        self._validate_id(task_id, "task_id")
        self._validate_id(handoff_id, "handoff_id")
        parsed = json.loads(payload)
        if (
            not isinstance(parsed, dict)
            or parsed.get("schema_version") != "flzat.high_handoff.v2"
            or parsed.get("handoff_id") != handoff_id
        ):
            raise ValueError("artifact schema or handoff ID does not match canonical payload")
        run_directory = self.output_root / run_id
        task_directory = run_directory / task_id
        path = task_directory / f"{handoff_id}.json"
        with self._lock:
            task_directory.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_text(encoding="utf-8") != payload:
                    raise ArtifactConflictError("handoff artifact already has other bytes")
                self._ensure_audit(run_directory, handoff_id, payload)
                return path
            self._atomic_write(path, payload)
            self._ensure_audit(run_directory, handoff_id, payload)
        return path

    def _ensure_audit(
        self,
        run_directory: Path,
        handoff_id: str,
        payload: str,
    ) -> None:
        audit = run_directory / "handoffs.jsonl"
        if audit.exists():
            for line in audit.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if value.get("handoff_id") == handoff_id:
                    if line != payload:
                        raise ArtifactConflictError("JSONL handoff ID has other bytes")
                    return
        with audit.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _atomic_write(path: Path, payload: str) -> None:
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_name = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name and Path(temporary_name).exists():
                Path(temporary_name).unlink()

    @staticmethod
    def _validate_id(value: str, name: str) -> None:
        if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in value):
            raise ValueError(f"{name} contains unsafe characters")


class RequestArtifactWriter:
    """Write immutable per-request evidence without exposing it to Low System."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.expanduser().resolve()
        self._lock = threading.Lock()

    def write_json(
        self,
        run_id: str,
        task_id: str,
        request_id: str,
        filename: str,
        value: dict[str, Any],
    ) -> Path:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return self.write_text(run_id, task_id, request_id, filename, payload)

    def write_text(
        self,
        run_id: str,
        task_id: str,
        request_id: str,
        filename: str,
        payload: str,
    ) -> Path:
        for name, value in (
            ("run_id", run_id),
            ("task_id", task_id),
            ("request_id", request_id),
        ):
            HandoffArtifactWriter._validate_id(value, name)
        self._validate_filename(filename)
        directory = self.output_root / run_id / task_id / request_id
        path = directory / filename
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_text(encoding="utf-8") != payload:
                    raise ArtifactConflictError(
                        f"request artifact already has other bytes: {filename}"
                    )
                return path
            HandoffArtifactWriter._atomic_write(path, payload)
        return path

    @staticmethod
    def _validate_filename(filename: str) -> None:
        path = Path(filename)
        if (
            not filename
            or path.name != filename
            or path.suffix not in {".json", ".txt"}
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
                for character in filename
            )
        ):
            raise ValueError("artifact filename is unsafe")
