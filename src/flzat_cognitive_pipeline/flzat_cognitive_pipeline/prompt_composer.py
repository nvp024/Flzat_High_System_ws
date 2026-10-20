"""Manifest-driven modular prompt composition with deterministic provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from flzat_cognitive_pipeline.profile_contracts import PROFILE_CONTRACTS


@dataclass(frozen=True)
class PromptProfile:
    name: str
    version: str
    mode: str
    output_schema: str
    validator: str
    system_sections: tuple[str, ...]
    user_sections: tuple[str, ...]
    optional_sections: dict[str, str]
    output_section: str


@dataclass(frozen=True)
class ComposedPrompt:
    system_prompt: str
    user_prompt: str
    profile: str
    version: str
    mode: str
    output_schema: str
    validator: str
    prompt_hash: str
    sections: tuple[str, ...]
    context_refs: tuple[str, ...]


class PromptComposer:
    """Discover profiles, enforce trusted contracts and append escaped runtime JSON."""

    def __init__(self, prompt_directory: str = "") -> None:
        self.root = self._resolve_root(prompt_directory)
        self._profiles = self._discover_profiles()
        if set(self._profiles) != set(PROFILE_CONTRACTS):
            raise ValueError(
                "prompt profiles must define exactly: "
                + ", ".join(PROFILE_CONTRACTS)
            )

    @property
    def profiles(self) -> dict[str, PromptProfile]:
        return dict(self._profiles)

    def compose(
        self,
        *,
        transcript: str,
        context_items: list[dict[str, Any]],
        context_refs: tuple[str, ...],
        dialogue_history: list[dict[str, str]],
        mode: str = "task_command",
        trusted_context: dict[str, Any] | None = None,
        include_rag_context: bool = True,
    ) -> ComposedPrompt:
        if not transcript.strip():
            raise ValueError("transcript is required")
        route_mode = mode.strip()
        profile = self._profiles.get(route_mode)
        if profile is None:
            raise ValueError(f"unsupported prompt route mode: {route_mode}")
        if not include_rag_context and (context_items or context_refs):
            raise ValueError("RAG context was supplied to a mode that disabled it")
        if include_rag_context and "rag_context" not in profile.optional_sections:
            raise ValueError(f"prompt profile {profile.name} does not allow RAG context")
        if dialogue_history and "dialogue_history" not in profile.optional_sections:
            raise ValueError(
                f"prompt profile {profile.name} does not allow dialogue history"
            )
        if trusted_context and "trusted_context" not in profile.optional_sections:
            raise ValueError(
                f"prompt profile {profile.name} does not allow trusted context"
            )

        runtime: dict[str, Any] = {
            "english_transcript": transcript.strip(),
            "router_mode": route_mode,
        }
        user_files = list(profile.user_sections)
        if dialogue_history:
            runtime["dialogue_history"] = dialogue_history
            user_files.append(profile.optional_sections["dialogue_history"])
        if include_rag_context:
            runtime["rag_context"] = context_items
            user_files.append(profile.optional_sections["rag_context"])
        if trusted_context:
            runtime["trusted_context"] = trusted_context
            user_files.append(profile.optional_sections["trusted_context"])
        user_files.append(profile.output_section)

        runtime_json = json.dumps(
            runtime,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        system_prompt = self._join(profile.system_sections)
        user_prompt = self._join(tuple(user_files))
        user_prompt += "\n\nRUNTIME_INPUT_JSON\n" + runtime_json
        return self._build(
            profile,
            system_prompt,
            user_prompt,
            tuple(user_files),
            context_refs,
        )

    def compose_repair(
        self,
        base_prompt: ComposedPrompt,
        *,
        raw_response: str,
        validation_error: str,
        repair_number: int,
    ) -> ComposedPrompt:
        if repair_number not in {1, 2}:
            raise ValueError("repair_number must be 1 or 2")
        profile = self._profiles.get(base_prompt.mode)
        if profile is None or profile.name != base_prompt.profile:
            raise ValueError("repair prompt profile does not match the base prompt")
        repair_payload = json.dumps(
            {
                "repair_number": repair_number,
                "validation_error": validation_error.strip()[:1000],
                "invalid_response": raw_response.strip()[:8000],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        user_prompt = (
            base_prompt.user_prompt
            + "\n\nFORMAT_REPAIR\n"
            + "Correct only the JSON format/schema error described below. "
            + "Use the same request and context. Do not invent facts, coordinates, "
            + "targets, skills, or context references. Return JSON only.\n"
            + repair_payload
        )
        sections = base_prompt.sections + (f"repair_attempt_{repair_number}",)
        return self._build(
            profile,
            base_prompt.system_prompt,
            user_prompt,
            sections,
            base_prompt.context_refs,
            sections_are_complete=True,
        )

    def compose_routing(
        self, transcript: str, pending: dict[str, Any] | None,
    ) -> ComposedPrompt:
        """Compose a short, text-only classifier request without RAG or vision."""
        system = self._join(("routing/system.txt",))
        user = self._join(("routing/task.txt", "routing/output.txt"))
        user += "\n\nRUNTIME_INPUT_JSON\n" + json.dumps(
            {"latest_utterance": transcript, "pending_request": pending},
            ensure_ascii=False, sort_keys=True, allow_nan=False,
        )
        return ComposedPrompt(
            system_prompt=system, user_prompt=user, profile="flzat_routing_v1",
            version="1.0.0", mode="routing", output_schema="SemanticRoute.v1",
            validator="semantic_route_v1",
            prompt_hash=hashlib.sha256((system + "\0" + user).encode()).hexdigest(),
            sections=("routing/system.txt", "routing/task.txt", "routing/output.txt"),
            context_refs=(),
        )

    def _discover_profiles(self) -> dict[str, PromptProfile]:
        profiles: dict[str, PromptProfile] = {}
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            profile = self._load_profile(manifest_path)
            if profile.mode in profiles:
                raise ValueError(f"duplicate prompt mode: {profile.mode}")
            profiles[profile.mode] = profile
        return profiles

    def _load_profile(self, path: Path) -> PromptProfile:
        manifest = self._load_manifest(path)
        name = self._required_text(manifest, "profile")
        version = self._required_text(manifest, "version")
        mode = self._required_text(manifest, "mode")
        output_schema = self._required_text(manifest, "output_schema")
        validator = self._required_text(manifest, "validator")
        contract = PROFILE_CONTRACTS.get(mode)
        if contract is None:
            raise ValueError(f"manifest has unsupported mode: {mode}")
        if output_schema != contract.output_schema or validator != contract.validator:
            raise ValueError(
                f"manifest contract mismatch for {mode}: {output_schema}/{validator}"
            )
        system_sections = self._file_list(manifest, "system_sections")
        user_sections = self._file_list(manifest, "user_sections")
        output_section = self._required_file(manifest, "output_section")
        optional_raw = manifest.get("optional_sections", {})
        if not isinstance(optional_raw, dict):
            raise ValueError("optional_sections must be a mapping")
        optional: dict[str, str] = {}
        allowed_optional = {"rag_context", "dialogue_history", "trusted_context"}
        if any(key not in allowed_optional for key in optional_raw):
            raise ValueError("manifest contains an unsupported optional section")
        for key, value in optional_raw.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"optional section {key} is invalid")
            optional[key] = self._checked_relative_file(value)
        if mode == "task_command" and "rag_context" not in optional:
            raise ValueError("task_command profile requires rag_context")
        return PromptProfile(
            name=name,
            version=version,
            mode=mode,
            output_schema=output_schema,
            validator=validator,
            system_sections=system_sections,
            user_sections=user_sections,
            optional_sections=optional,
            output_section=output_section,
        )

    def _build(
        self,
        profile: PromptProfile,
        system_prompt: str,
        user_prompt: str,
        sections: tuple[str, ...],
        context_refs: tuple[str, ...],
        *,
        sections_are_complete: bool = False,
    ) -> ComposedPrompt:
        digest = hashlib.sha256(
            (system_prompt + "\0" + user_prompt).encode("utf-8")
        ).hexdigest()
        complete_sections = (
            sections if sections_are_complete else profile.system_sections + sections
        )
        return ComposedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            profile=profile.name,
            version=profile.version,
            mode=profile.mode,
            output_schema=profile.output_schema,
            validator=profile.validator,
            prompt_hash=digest,
            sections=complete_sections,
            context_refs=context_refs,
        )

    def _join(self, files: tuple[str, ...]) -> str:
        return "\n\n".join(
            f"## {filename}\n{self._read(filename)}" for filename in files
        )

    def _file_list(self, manifest: dict[str, Any], name: str) -> tuple[str, ...]:
        values = manifest.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"manifest {name} must be a nonempty list")
        return tuple(self._checked_relative_file(value) for value in values)

    def _required_file(self, manifest: dict[str, Any], name: str) -> str:
        return self._checked_relative_file(self._required_text(manifest, name))

    def _checked_relative_file(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("prompt section path must be a nonempty string")
        relative = Path(value.strip())
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("prompt section path must stay inside the prompt root")
        # Do not resolve the final file here.  A ROS 2 ``--symlink-install``
        # intentionally points installed prompt files back into the source tree,
        # so resolving the symlink would make a valid packaged file appear to be
        # outside the install-space prompt root.  Absolute paths and ``..`` are
        # rejected above; the remaining path is package-owned and must be a file.
        path = self.root / relative
        if not path.is_file():
            raise ValueError(f"prompt section does not exist: {relative}")
        return relative.as_posix()

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
    def _resolve_root(prompt_directory: str) -> Path:
        if prompt_directory.strip():
            configured = Path(prompt_directory).expanduser().resolve()
            root = (
                configured.parent
                if (configured / "manifest.json").is_file()
                else configured
            )
        else:
            from ament_index_python.packages import get_package_share_directory

            root = (
                Path(get_package_share_directory("flzat_cognitive_pipeline"))
                / "prompts"
            )
        if not root.is_dir():
            raise FileNotFoundError(f"cognitive prompt root not found: {root}")
        return root
