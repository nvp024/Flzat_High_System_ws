import sys
import threading
from types import SimpleNamespace

from flzat_speech.tts_node import _PiperEngine, _resample_pcm_int16
import numpy as np


class _FakeStream:
    instances = []

    def __init__(self, *, samplerate, channels, dtype):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.writes = []
        self.aborted = False
        self.__class__.instances.append(self)

    def start(self):
        return None

    def write(self, samples):
        self.writes.append(samples.copy())

    def stop(self):
        return None

    def abort(self):
        self.aborted = True

    def close(self):
        return None


class _FakeVoice:
    load_calls = 0

    @classmethod
    def load(cls, model_path, config_path=None, use_cuda=False):
        cls.load_calls += 1
        return cls()

    def synthesize(self, text, syn_config=None):
        samples = np.array([1, -1, 2, -2], dtype=np.int16)
        yield SimpleNamespace(
            sample_rate=16_000,
            sample_width=2,
            sample_channels=1,
            audio_int16_bytes=samples.tobytes(),
        )


class _FakeSynthesisConfig:
    def __init__(self, **values):
        self.values = values


def _install_fake_modules(monkeypatch):
    _FakeVoice.load_calls = 0
    _FakeStream.instances = []
    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(
            PiperVoice=_FakeVoice,
            SynthesisConfig=_FakeSynthesisConfig,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(OutputStream=_FakeStream),
    )


def test_piper_loads_once_and_reuses_voice(monkeypatch, tmp_path):
    _install_fake_modules(monkeypatch)
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"test")
    words = []
    engine = _PiperEngine(
        threading.Event(),
        str(model),
        "",
        volume=1.0,
        length_scale=1.0,
        output_sample_rate=48_000,
        sentence_pause_ms=10,
        max_chunk_characters=240,
        use_cuda=False,
    )

    assert engine.speak("First sentence. Second sentence.", words.append)
    assert engine.speak("Another response.")

    assert _FakeVoice.load_calls == 1
    assert words == ["First", "sentence.", "Second", "sentence."]
    assert len(_FakeStream.instances) == 2
    assert all(stream.samplerate == 48_000 for stream in _FakeStream.instances)
    assert any(
        np.all(samples == 0)
        for samples in _FakeStream.instances[0].writes
    )


def test_piper_stop_aborts_active_stream(monkeypatch, tmp_path):
    _install_fake_modules(monkeypatch)
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"test")
    stop_event = threading.Event()
    engine = _PiperEngine(
        stop_event,
        str(model),
        "",
        volume=1.0,
        length_scale=1.0,
        output_sample_rate=48_000,
        sentence_pause_ms=0,
        max_chunk_characters=240,
        use_cuda=False,
    )
    stream = _FakeStream(samplerate=16_000, channels=1, dtype="int16")
    engine._stream = stream

    engine.request_stop()

    assert stop_event.is_set()
    assert stream.aborted


def test_resample_pcm_int16_preserves_channels_and_duration():
    samples = np.array([[0, 100], [1000, -1000], [0, 100]], dtype=np.int16)

    result = _resample_pcm_int16(samples, 16_000, 48_000)

    assert result.dtype == np.int16
    assert result.shape == (9, 2)
    assert np.array_equal(result[0], samples[0])
    assert np.array_equal(result[-1], samples[-1])
