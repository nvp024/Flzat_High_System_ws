"""Pure policies shared by cognitive ingress and Action B sessions."""

from __future__ import annotations

from dataclasses import dataclass
import re


ASK_USER_KINDS = frozenset({"ASK", "CONFIRM", "REFUSE", "INFORM"})
ASK_USER_LISTENING_KINDS = frozenset({"ASK", "CONFIRM"})
ASK_USER_OUTCOMES = frozenset({"ANSWERED", "DECLINED", "TIMEOUT", "SUPERSEDED"})
HANDLE_INTENT_OUTCOMES = frozenset(
    {"STARTED_FAST", "STARTED_PLAN", "REFUSED", "FAILED", "CANCELED"}
)
HANDLE_INTENT_STAGES = frozenset(
    {"deciding", "validating", "planning", "waiting_for_user"}
)

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_STOP_PATTERNS = (
    re.compile(r"^stop(?: now| the robot| the task| the mission)?$"),
    re.compile(r"^cancel(?: the)?(?: task| mission| command)?$"),
    re.compile(r"^emergency(?: stop)?$"),
    re.compile(r"^dừng(?: lại| robot| nhiệm vụ)?$"),
)
_STOP_EXACT = frozenset(
    {
        "abort",
        "freeze",
        "halt",
        "hủy nhiệm vụ",
        "ngừng",
        "nguy hiểm",
    }
)
_DECLINE_PATTERNS = (
    re.compile(r"^(?:i (?:do not|don t) want to answer|i refuse(?: to answer)?|skip)$"),
    re.compile(r"^(?:tôi không muốn trả lời|tôi từ chối|bỏ qua)$"),
)
_UNCLEAR_PATTERNS = (
    re.compile(r"^(?:i do not know|i don t know|not sure|unsure|what|huh|maybe)$"),
    re.compile(r"^(?:không biết|tôi không biết|không chắc|gì cơ|hả)$"),
)


def normalize_utterance(text: str) -> str:
    """Normalize only for policy matching; never replace the retained raw answer."""
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", text.casefold())).strip()


def is_stop_command(text: str) -> bool:
    value = normalize_utterance(text)
    return value in _STOP_EXACT or any(pattern.fullmatch(value) for pattern in _STOP_PATTERNS)


@dataclass(frozen=True)
class AnswerAssessment:
    disposition: str
    raw_answer: str


def assess_user_answer(text: str) -> AnswerAssessment:
    """Classify only explicit decline/unclear replies; preserve all useful language.

    A negative CONFIRM answer such as "No, use the blue bottle instead" remains
    ANSWERED because it contains task information and is not a refusal to answer.
    """
    raw = text.strip()
    if not raw:
        return AnswerAssessment("UNCLEAR", raw)
    value = normalize_utterance(raw)
    if any(pattern.fullmatch(value) for pattern in _DECLINE_PATTERNS):
        return AnswerAssessment("DECLINED", raw)
    if any(pattern.fullmatch(value) for pattern in _UNCLEAR_PATTERNS):
        return AnswerAssessment("UNCLEAR", raw)
    return AnswerAssessment("ANSWERED", raw)


def validate_safe_identifier(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in normalized
    ):
        raise ValueError(f"{name} must be a nonempty safe identifier")
    return normalized
