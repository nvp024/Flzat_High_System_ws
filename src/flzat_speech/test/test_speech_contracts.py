from pathlib import Path

from flzat_speech.stt_node import (
    DEFAULT_WHISPER_MODEL,
    EXPECTED_SAMPLE_RATE,
    _build_transcribe_options,
    _extract_confident_transcript,
)
from flzat_speech.tts_node import _split_speech_text


ROOT = Path(__file__).resolve().parents[1]


def test_stt_is_pinned_to_english_and_16khz():
    source = (ROOT / "flzat_speech/stt_node.py").read_text(encoding="utf-8")

    assert 'declare_parameter("language", "en")' in source
    assert EXPECTED_SAMPLE_RATE == 16_000
    assert DEFAULT_WHISPER_MODEL == "base.en.pt"


def test_stt_uses_deterministic_beam_search_for_independent_commands():
    options = _build_transcribe_options("en", 5, "FLZAT, Nav2")

    assert options == {
        "task": "transcribe",
        "fp16": False,
        "temperature": 0.0,
        "beam_size": 5,
        "condition_on_previous_text": False,
        "sample_len": 64,
        "verbose": None,
        "language": "en",
        "initial_prompt": "FLZAT, Nav2",
    }


def test_stt_rejects_high_no_speech_probability_and_low_logprob():
    result = {
        "text": "You get the book",
        "segments": [
            {"text": "You", "no_speech_prob": 0.77, "avg_logprob": -0.72},
            {
                "text": "get the book",
                "no_speech_prob": 0.05,
                "avg_logprob": -0.25,
            },
            {"text": "noise", "no_speech_prob": 0.10, "avg_logprob": -1.40},
        ],
    }

    transcript = _extract_confident_transcript(result, 0.65, -1.0, 2.0)

    assert transcript == "get the book"


def test_stt_rejects_repetitive_or_implausibly_fast_transcript():
    repetitive = {
        "segments": [
            {
                "text": "yum " * 40,
                "no_speech_prob": 0.05,
                "avg_logprob": -0.20,
                "compression_ratio": 3.1,
            }
        ]
    }
    too_fast = {
        "segments": [
            {
                "text": "one two three four five six seven eight nine ten eleven",
                "no_speech_prob": 0.05,
                "avg_logprob": -0.20,
                "compression_ratio": 1.0,
            }
        ]
    }

    assert _extract_confident_transcript(repetitive, 0.65, -1.0, 4.0) == ""
    assert _extract_confident_transcript(too_fast, 0.65, -1.0, 2.0) == ""


def test_vad_publishes_generated_speech_audio_contract():
    source = (ROOT / "flzat_speech/vad_node.py").read_text(encoding="utf-8")

    assert "from flzat_high_interfaces.msg import SpeechAudio" in source
    assert 'create_publisher(SpeechAudio, "/audio_events"' in source


def test_speech_ingress_bridges_audio_through_stt_to_text_topic():
    source = (ROOT / "flzat_speech/speech_ingress.py").read_text(
        encoding="utf-8"
    )

    assert '"/audio_events"' in source
    assert '"/stt_action"' in source
    assert '"/flzat/high_system/stt_result"' in source
    assert "SpeechToText" in source


def test_tts_splits_at_sentence_boundaries_without_rewriting_text():
    text = "I found the box. It is on the table, next to the book."

    chunks = _split_speech_text(text, 240)

    assert chunks == (
        "I found the box.",
        "It is on the table, next to the book.",
    )
    assert " ".join(chunks) == text


def test_tts_bounds_long_chunks_and_preserves_all_words():
    text = "Move the blue box to the right side of the table, then return to me."

    chunks = _split_speech_text(text, 40)

    assert all(len(chunk) <= 40 for chunk in chunks)
    assert " ".join(chunks).replace(", ", " ") == text.replace(", ", " ")


def test_tts_uses_piper_as_its_only_backend():
    source = (ROOT / "flzat_speech/tts_node.py").read_text(encoding="utf-8")

    assert "class _PiperEngine" in source
    assert "self._engine = _PiperEngine(" in source
