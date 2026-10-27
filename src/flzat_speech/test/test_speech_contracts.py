from pathlib import Path
from types import SimpleNamespace

import pytest

from flzat_speech.stt_node import (
    DEFAULT_WHISPER_MODEL,
    EXPECTED_SAMPLE_RATE,
    _build_transcribe_options,
    _extract_confident_transcript,
    _transcribe_bounded,
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


def test_stt_rejects_entire_command_instead_of_dropping_words():
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

    with pytest.raises(ValueError):
        _extract_confident_transcript(result, 0.65, -1.0, 2.0)


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

    with pytest.raises(ValueError, match="repetitive"):
        _extract_confident_transcript(repetitive, 0.65, -1.0, 4.0)
    with pytest.raises(ValueError, match="word rate"):
        _extract_confident_transcript(too_fast, 0.65, -1.0, 2.0)


def segment(text, **overrides):
    return dict(text=text, no_speech_prob=0.1, avg_logprob=-0.2,
                compression_ratio=1.0, **overrides)


def test_uncertain_negation_does_not_become_positive_command():
    first = segment("Do not")
    first["avg_logprob"] = -1.5
    with pytest.raises(ValueError, match="low segment confidence"):
        _extract_confident_transcript(
            {"segments": [first, segment("pick up the red box.")]}, 0.65, -1.0, 3.0
        )


def test_confident_command_preserves_all_segments():
    assert _extract_confident_transcript(
        {"segments": [segment("Do not"), segment("pick up the red box.")]},
        0.65, -1.0, 3.0,
    ) == "Do not pick up the red box."


@pytest.mark.parametrize("result", [
    {"text": "pick up the box"}, {"segments": [None]},
    {"segments": [{"text": "pick up the box"}]},
    {"segments": [{**segment("go"), "no_speech_prob": float("nan")}]},
    {"segments": [{**segment("go"), "avg_logprob": float("inf")}]},
    {"segments": [{**segment("go"), "compression_ratio": None}]},
])
def test_stt_rejects_missing_or_invalid_metadata(result):
    with pytest.raises(ValueError):
        _extract_confident_transcript(result, 0.65, -1.0, 2.0)


def test_stt_rejects_repetition_across_segments():
    with pytest.raises(ValueError, match="repeated phrase"):
        _extract_confident_transcript(
            {"segments": [segment("I can tell you") for _ in range(4)]},
            0.65, -1.0, 10.0,
        )


@pytest.mark.parametrize("count", [31, 32])
def test_decode_limit_checked_before_segment_slicing_and_method_restored(count):
    class Model:
        def decode(self, *args, **kwargs):
            return SimpleNamespace(tokens=list(range(count)))

        def transcribe(self, audio, **options):
            self.decode(audio)
            return {"segments": [segment("go")]}  # Sliced result hides token count.

    model = Model()
    options = _build_transcribe_options("en", 5, "", 32)
    assert options["sample_len"] == 32
    if count == 32:
        with pytest.raises(ValueError, match="token limit"):
            _transcribe_bounded(model, [], options)
    else:
        assert _transcribe_bounded(model, [], options)["segments"]
    assert "decode" not in model.__dict__
    assert len(model.decode().tokens) == count


def test_shared_launch_defaults_and_typed_overrides(tmp_path, monkeypatch):
    from launch import LaunchContext
    from flzat_speech.launch_config import speech_arguments, speech_parameters

    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    context = LaunchContext()
    arguments = speech_arguments()
    assert len({arg.name for arg in arguments}) == len(arguments)
    for argument in arguments:
        argument.execute(context)
    assert speech_parameters("stt")["language"].evaluate(context) == "en"
    assert speech_parameters("stt")["whisper_model_path"].evaluate(context) == ""
    assert speech_parameters("stt")["beam_size"].evaluate(context) == 5
    assert speech_parameters("ingress")["pending_audio_max_age_s"].evaluate(context) == 2.0
    context.launch_configurations["speech_max_duration_s"] = "12.0"
    context.launch_configurations["whisper_max_decode_tokens"] = "96"
    assert speech_parameters("vad")["max_segment_duration_s"].evaluate(context) == 12.0
    assert speech_parameters("stt")["max_audio_duration_s"].evaluate(context) == 12.0
    assert speech_parameters("stt")["max_decode_tokens"].evaluate(context) == 96
    context.launch_configurations["piper_length_scale"] = "0.85"
    context.launch_configurations["piper_use_cuda"] = "false"
    assert speech_parameters("tts")["piper_length_scale"].evaluate(context) == 0.85
    assert speech_parameters("tts")["piper_use_cuda"].evaluate(context) is False


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
