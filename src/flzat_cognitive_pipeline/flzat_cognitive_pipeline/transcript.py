"""Small, dependency-free transcript validation shared by ROS inputs."""

from __future__ import annotations


def is_usable_transcript(transcript: str) -> bool:
    """Return true when a transcript contains at least one alphanumeric token."""

    text = transcript.strip()
    return bool(text) and any(character.isalnum() for character in text)
