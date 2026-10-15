# Runtime model assets

This directory is owned by `flzat_high_system_ws`; it is not a symbolic link
to a legacy workspace.

Required local files:

- `tiny.pt`: Whisper Tiny checkpoint used by `flzat_speech/stt_node.py`.
- `snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit`: Silero VAD
  TorchScript model used by `flzat_speech/vad_node.py`.

Only runtime weights are retained. Qwen3-VL and sentence-transformer weights
are resolved through their configured Hugging Face model IDs/cache and are not
duplicated here.

The Silero VAD license is preserved at
`snakers4_silero-vad_master/LICENSE`.
