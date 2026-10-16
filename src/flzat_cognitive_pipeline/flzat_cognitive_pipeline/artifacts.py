"""Idempotent canonical command artifacts with atomic per-command writes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any


class ArtifactConflictError(RuntimeError):
    pass


class CommandArtifactWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.expanduser().resolve()
        self._lock = threading.Lock()

    def write(
        self,
        run_id: str,
        task_id: str,
        command_id: str,
        payload: str,
    ) -> Path:
        self._validate_id(run_id, "run_id")
        self._validate_id(task_id, "task_id")
        self._validate_id(command_id, "command_id")
        parsed = json.loads(payload)
        if (
            parsed.get("run_id") != run_id
            or parsed.get("task_id") != task_id
            or parsed.get("command_id") != command_id
        ):
            raise ValueError("artifact IDs do not match canonical payload")
        run_directory = self.output_root / run_id
        task_directory = run_directory / task_id
        path = task_directory / f"{command_id}.json"
        with self._lock:
            task_directory.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_text(encoding="utf-8") != payload:
                    raise ArtifactConflictError("command artifact already has other bytes")
                self._ensure_audit(run_directory, command_id, payload)
                return path
            self._atomic_write(path, payload)
            self._ensure_audit(run_directory, command_id, payload)
        return path

    def _ensure_audit(
        self,
        run_directory: Path,
        command_id: str,
        payload: str,
    ) -> None:
        audit = run_directory / "commands.jsonl"
        if audit.exists():
            for line in audit.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if value.get("command_id") == command_id:
                    if line != payload:
                        raise ArtifactConflictError("JSONL command ID has other bytes")
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
            CommandArtifactWriter._validate_id(value, name)
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
            CommandArtifactWriter._atomic_write(path, payload)
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
