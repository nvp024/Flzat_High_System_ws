from pathlib import Path

from flzat_speech.stt_node import EXPECTED_SAMPLE_RATE


ROOT = Path(__file__).resolve().parents[1]


def test_stt_is_pinned_to_english_and_16khz():
    source = (ROOT / "flzat_speech/stt_node.py").read_text(encoding="utf-8")

    assert 'declare_parameter("language", "en")' in source
    assert 'transcribe_options["language"] = self._language' in source
    assert EXPECTED_SAMPLE_RATE == 16_000


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
