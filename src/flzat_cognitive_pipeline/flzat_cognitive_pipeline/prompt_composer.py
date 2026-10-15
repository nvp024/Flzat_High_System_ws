"""Versioned modular prompt composition with deterministic provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComposedPrompt:
    system_prompt: str
    user_prompt: str
    profile: str
    version: str
    prompt_hash: str
    sections: tuple[str, ...]
    context_refs: tuple[str, ...]


class PromptComposer:
    """Load immutable prompt sections and append escaped runtime JSON."""

    _SYSTEM_FILES = (
        "system/identity.txt",
        "system/capabilities.txt",
        "policies/safety.txt",
        "policies/uncertainty.txt",
        "policies/high_low_boundary.txt",
    )
    _TASK_FILE = "tasks/intent_router.txt"
    _RAG_FILE = "context/rag_context.txt"
    _OUTPUT_FILE = "output/model_decision_v1.txt"
    _DIALOGUE_FILE = "context/dialogue_history.txt"

    def __init__(self, prompt_directory: str = "") -> None:
        if prompt_directory.strip():
            self.root = Path(prompt_directory).expanduser().resolve()
        else:
            from ament_index_python.packages import get_package_share_directory

            self.root = (
                Path(get_package_share_directory("flzat_cognitive_pipeline"))
                / "prompts"
                / "flzat_cognitive_v1"
            )
        if not self.root.is_dir():
            raise FileNotFoundError(f"cognitive prompt profile not found: {self.root}")
        manifest = self._load_manifest(self.root / "manifest.json")
        self.profile = self._required_text(manifest, "profile")
        self.version = self._required_text(manifest, "version")
        self._system_sections = tuple(
            (filename, self._read(filename)) for filename in self._SYSTEM_FILES
        )
        self._task_section = (self._TASK_FILE, self._read(self._TASK_FILE))
        self._rag_section = (self._RAG_FILE, self._read(self._RAG_FILE))
        self._output_section = (self._OUTPUT_FILE, self._read(self._OUTPUT_FILE))
        self._dialogue_section = (
            self._DIALOGUE_FILE,
            self._read(self._DIALOGUE_FILE),
        )

    def compose(
        self,
        *,
        transcript: str,
        context_items: list[dict[str, Any]],
        context_refs: tuple[str, ...],
        dialogue_history: list[dict[str, str]],
    ) -> ComposedPrompt:
        if not transcript.strip():
            raise ValueError("transcript is required")
        runtime: dict[str, Any] = {
            "english_transcript": transcript.strip(),
            "rag_context": context_items,
        }
        user_sections = [self._task_section]
        if dialogue_history:
            runtime["dialogue_history"] = dialogue_history
            user_sections.append(self._dialogue_section)
        user_sections.append(self._rag_section)
        user_sections.append(self._output_section)
        runtime_json = json.dumps(
            runtime,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        system_prompt = self._join(self._system_sections)
        active_user_sections = tuple(user_sections)
        user_prompt = self._join(active_user_sections) + "\n\nRUNTIME_INPUT_JSON\n" + runtime_json
        digest = hashlib.sha256(
            (system_prompt + "\0" + user_prompt).encode("utf-8")
        ).hexdigest()
        sections = tuple(
            filename
            for filename, _ in self._system_sections + active_user_sections
        )
        return ComposedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            profile=self.profile,
            version=self.version,
            prompt_hash=digest,
            sections=sections,
            context_refs=context_refs,
        )

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot load prompt manifest {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("prompt manifest must be a JSON object")
        return value

    @staticmethod
    def _required_text(value: dict[str, Any], name: str) -> str:
        result = value.get(name)
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"prompt manifest {name} is invalid")
        return result.strip()

    def _read(self, relative: str) -> str:
        path = self.root / relative
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"cannot read prompt section {path}: {exc}") from exc
        if not value:
            raise ValueError(f"prompt section is empty: {path}")
        return value

    @staticmethod
    def _join(sections: tuple[tuple[str, str], ...]) -> str:
        return "\n\n".join(
            f"## {filename}\n{content}" for filename, content in sections
        )
