# Runtime model assets

This directory is owned by `flzat_high_system_ws`; it is not a symbolic link
to a legacy workspace.

Required local files:

- `base.en.pt`: default English Whisper checkpoint used by
  `flzat_speech/stt_node.py`.
- `snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit`: Silero VAD
  TorchScript model used by `flzat_speech/vad_node.py`.
- `tts/en_US-lessac-medium.onnx`: default English Piper voice used by
  `flzat_speech/tts_node.py`.
- `tts/en_US-lessac-medium.onnx.json`: Piper voice configuration downloaded
  alongside the ONNX model.

Only runtime weights are retained. Qwen3-VL and sentence-transformer weights
are resolved through their configured Hugging Face model IDs/cache and are not
duplicated here.

The Silero VAD license is preserved at
`snakers4_silero-vad_master/LICENSE`.

Install `piper-tts` and download the default voice from the workspace root:

```bash
python3 -m pip install "piper-tts==1.8.0"
python3 -m piper.download_voices \
  --download-dir models/tts \
  en_US-lessac-medium
```

Download the default English Whisper checkpoint from the workspace root:

```bash
python3 -c "import whisper; whisper.load_model('base.en', download_root='models')"
```

Piper and each downloaded voice have their own licenses. Review the generated
license/metadata files before redistributing model assets.
