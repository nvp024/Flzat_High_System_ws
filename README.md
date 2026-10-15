# FLZAT High System

Unified ROS 2 workspace for the High-Level Cognitive Pipeline:

```text
Voice → VAD → English STT → Cognitive Manager
→ read-only RAG → modular prompt → Qwen3-VL
→ validate/normalize JSON → clarification or High-Level Command
→ JSON artifact + ROS 2 topic → Low System
```

`flzat_high_system_ws` is the canonical source workspace. Older workspaces are
kept only as references and are not runtime dependencies.

## Workspace structure

- `flzat_speech`: VAD, English STT, and TTS.
- `flzat_vision_context`: camera buffering and `/vision/get_frame_near`.
- `flzat_memory`: read-only access to an existing Chroma database through
  `QueryMemory`.
- `flzat_cognitive_pipeline`: manager, RAG context, prompt composition,
  validation, and command generation.
- `flzat_vlm`: Qwen3-VL server for cognitive inference.
- `flzat_high_bringup`: the production launch file and test adapters/mocks.

There is one production manager, `cognitive_manager`, and one canonical prompt
profile for cognitive commands, `flzat_cognitive_v1/manifest.json`.

## Install dependencies and build

```bash
conda activate py312
source /opt/ros/jazzy/setup.bash
cd /home/phucnv/Documents/JTR/High_system/flzat_high_system_ws

pip install -r requirements.txt
python -m pip check
colcon build --symlink-install
source install/setup.bash
```

Full voice mode uses these runtime model files stored directly in this
workspace; `models/` is a real directory and does not link to an older
workspace:

```text
models/tiny.pt
models/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit
```

Only the two weights required at runtime are kept here. Qwen3-VL and the
sentence-transformer model remain managed through their model cache according
to the configured model IDs.

`environment_id` and `map_id` are intentionally required launch arguments;
there is no silent fallback to an arbitrary or newest database.

## Local test without Qwen, a camera, or a microphone

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:=/home/phucnv/Documents/JTR/High_system \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  embedding_backend:=deterministic_test \
  enable_text_command:=true \
  use_fake_speech:=true \
  use_fake_vlm:=true \
  use_low_mock:=true \
  frame_source:=static
```

Send a command from another terminal:

```bash
ros2 topic pub --once /flzat/high_system/text_command \
  std_msgs/msg/String "{data: 'get the book on the table'}"
```

## Test the full pipeline with an existing image

This runs the standard ROS 2 pipeline. The static frame server replaces only
the camera; it does not bypass RAG, the Cognitive Manager, PromptComposer, or
output processing.

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:=/path/to/flzat_memory_data \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  enable_text_command:=true \
  use_fake_speech:=true \
  use_fake_vlm:=false \
  use_low_mock:=true \
  frame_source:=static \
  static_image_path:=/path/to/image.png \
  model_id:=Qwen/Qwen3-VL-2B-Instruct \
  device:=cuda \
  dtype:=bfloat16
```

Copy this workspace and the database directory to the GPU/cloud machine, then
run the same ROS 2 commands. There is no separate cloud pipeline or dependency
file.

## Test the VLM server with an image and prompt

This command tests only the model server; it does not test RAG or the cognitive
pipeline:

```bash
ros2 launch flzat_vlm vlm_server.launch.py \
  model_id:=Qwen/Qwen3-VL-2B-Instruct device:=cuda dtype:=bfloat16

ros2 run flzat_vlm vlm_test_client \
  --image /path/to/image.png \
  --prompt "Describe the objects visible in this image."
```

## Production

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:=$HOME/.local/share/flzat/environment_memory \
  environment_id:=hotel_demo_14 \
  map_id:=<MAP_ID> \
  model_id:=Qwen/Qwen3-VL-2B-Instruct
```

The Low System subscribes to `/flzat/high_system/command`, publishes
`/flzat/low_system/status`, and receives `/flzat/low_system/cancel`.

Runtime IDs use readable formats such as `run-20261014T153000Z`, `task-0001`,
`task-0001-r01`, and `task-0001-c01`. Artifacts are stored per run so sequence
numbers never overwrite data from a previous process invocation.

`qwen3_vl` is the only production VLM backend. `audio_loopback` and
`acceptance_client` are test tools that use the same production ROS contracts.

See `docs/ARCHITECTURE.md`, `docs/INTERFACES.md`, and `docs/TEST_REPORT.md`.
