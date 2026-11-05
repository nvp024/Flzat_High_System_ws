# FLZAT High System

ROS 2 companion-robot pipeline: microphone → VAD/STT → image + RAG context →
Qwen3-VL → speech and high-level goals handed to Low through Action A.

Last updated: **5 November 2026**.

## Overview and structure

`vlm_brain` collects the user's request, a camera frame and relevant RAG records,
composes the prompt and asks Qwen to respond, clarify or form a high-level goal.
Speech goes to TTS; action goals become JSON files passed to Low. Low handles
planning and execution and can ask the user through Action B.

```text
src/
├── flzat_high_bringup/        # Launch files and runtime configuration
├── flzat_high_interfaces/    # ROS messages, services and Action A/B
├── flzat_cognitive_pipeline/ # Brain, modular prompts, validation and JSON writing
├── flzat_vlm/                # Qwen3-VL inference backend
├── flzat_memory/             # Read-only Chroma RAG retrieval
├── flzat_speech/             # VAD, Whisper STT and Piper TTS
└── flzat_vision_context/     # Camera capture and nearest-frame service
models/                      # Local model assets
demo_1/                      # Bundled RAG database
output/                      # Generated JSON and request artifacts
```

## Run the full pipeline

Use the Python environment in which the dependencies and models were installed.
The machine needs an NVIDIA GPU with BF16 support, a microphone, a camera and
audio output. Whisper, Silero VAD and Piper model files must be available.

From the directory containing the workspace:

```bash
cd flzat_high_system_ws
source /opt/ros/jazzy/setup.bash

colcon build
source install/setup.bash

ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:="$PWD" \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  output_root:="$PWD/output" \
  embedding_backend:=sentence_transformer \
  enable_text_command:=false \
  use_fake_speech:=false \
  use_fake_vlm:=false \
  use_low_mock:=false \
  frame_source:=camera \
  run_camera_node:=true \
  run_vad_node:=true \
  run_stt_node:=true \
  run_tts_node:=true \
  whisper_language:=en \
  model_id:=Qwen/Qwen3-VL-2B-Instruct \
  device:=cuda \
  dtype:=bfloat16 \
  quantization:=none \
  max_new_tokens:=256 \
  log_level:=info
```

Use the ROS distribution installed on your machine in the source command.
The bundled RAG database is `demo_1/`; change `storage_root`,
`environment_id` and `map_id` when using another database.

Wait for the models and devices to initialize, then speak English into the
microphone. The pipeline captures a camera frame and retrieves RAG context
for the request. Stop the launch with `Ctrl+C`.

Low must run its own Action A server at `/llm/handle_intent` and have access
to the JSON files under `output_root`. This command starts no Low mock.
Low can request user interaction through Action B at `/brain/ask_user`.

## JSON handoff to Low

With the command above, files are saved under `flzat_high_system_ws/output/`:

```text
<output_root>/<run_id>/<task_id>/<handoff_id>.json
output/run-20261105T080000Z/task-0001/task-0001-c01.json
```

High writes the complete file before sending Action A. Low receives a
`flzat_high_interfaces/action/HandleIntent` goal containing `run_id`, `task_id`,
`handoff_id` and the absolute `json_path`. Read that exact path rather than
searching for the latest file. A speech-only reply creates no handoff.

Inside Low's Action A server, read the payload from the goal:

```python
import json
from pathlib import Path

request = goal_handle.request
path = Path(request.json_path).resolve(strict=True)
payload = json.loads(path.read_text(encoding="utf-8"))

if payload.get("schema_version") != "flzat.high_handoff.v2":
    raise ValueError("Unsupported handoff schema")
if payload.get("handoff_id") != request.handoff_id:
    raise ValueError("Handoff ID does not match Goal A")

user_request = payload["user_request"]
goal = payload["goal"]
context = payload["context"]
entities = payload["entities"]
```

`user_request` preserves the original words and clarification answers; `goal`
contains the objective, constraints and missing information. `entities` contains
selected RAG objects, their relationships and each object's stored
`position.frame_id/x/y/z` when available. Positions come from memory and are not
confirmed current observations.

Low should also validate the path suffix against Goal A's IDs:
`<run_id>/<task_id>/<handoff_id>.json`.
Across machines or containers, mount the shared output directory at the same
absolute path so Low can open `json_path`; the Action does not transfer the file.
Keep files available until Low has finished reading them.
