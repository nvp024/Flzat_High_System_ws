"""Small in-memory conversation context kept separate from environment RAG."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationTurn:
    mode: str
    user: str
    assistant: str

    def prompt_value(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "user": self.user,
            "assistant": self.assistant,
        }


class ConversationStore:
    def __init__(self, max_turns: int = 5, max_text_characters: int = 1000) -> None:
        if not 1 <= max_turns <= 20:
            raise ValueError("max_turns must be between 1 and 20")
        if not 64 <= max_text_characters <= 10_000:
            raise ValueError("max_text_characters must be between 64 and 10000")
        self.max_turns = max_turns
        self.max_text_characters = max_text_characters
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def append(self, mode: str, user: str, assistant: str) -> None:
        values = (mode.strip(), user.strip(), assistant.strip())
        if not all(values):
            raise ValueError("conversation turn fields cannot be empty")
        self._turns.append(
            ConversationTurn(
                mode=values[0][:64],
                user=values[1][: self.max_text_characters],
                assistant=values[2][: self.max_text_characters],
            )
        )

    def recent(self, limit: int | None = None) -> list[dict[str, str]]:
        count = self.max_turns if limit is None else int(limit)
        if not 1 <= count <= self.max_turns:
            raise ValueError("conversation limit is outside configured bounds")
        return [turn.prompt_value() for turn in tuple(self._turns)[-count:]]

    def __len__(self) -> int:
        return len(self._turns)

