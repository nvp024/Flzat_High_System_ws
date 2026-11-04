# FLZAT High System

ROS 2 High-Level Cognitive Pipeline for the FLZAT companion robot.

Last updated: **4 November 2026**.

## Current scope

The workspace implements one canonical High pipeline:

```text
Voice or text
  → VAD / STT
  → cognitive manager
  → bounded conversation + RAG + current image + trusted policies
  → modular prompt
  → Qwen3-VL
  → CognitiveDecision.v1 validation and bounded format repair
  → atomic compact flzat.high_handoff.v2 JSON
  → Action A /llm/handle_intent (JSON path only)

Low needs user input
  → Action B /brain/ask_user
  → High TTS → user answer → optional interpretation
  → Result B returned to the same Low request
```

High understands the user, maintains the interaction context, grounds memory
references and hands a natural-language goal to Low. High does **not** select
navigation/manipulation intents, FAST-SKILL, PLANNED-TASK, behavior trees or
controllers. Those responsibilities belong to Low.

The repository contains a deterministic Low mock for integration testing. It
does not contain the production Low LLM or executor.

## Packages

| Package | Responsibility |
|---|---|
| `flzat_high_interfaces` | ROS messages, services and Action A/B contracts |
| `flzat_speech` | VAD, STT, TTS and speech ingress |
| `flzat_vision_context` | Camera/static frame provider and `GetFrameNear` |
| `flzat_memory` | Read-only Chroma semantic-memory query service |
| `flzat_vlm` | Qwen3-VL action server and backend |
| `flzat_cognitive_pipeline` | Context, prompt, validation, artifacts and orchestration |
| `flzat_high_bringup` | Full launch, fake servers and Low mock |

Bundled test data:

- `demo_1/`: completed semantic-memory fixture, including `chroma.sqlite3`;
- `images/`: static images for cloud or headless testing.

Only `flzat_brain_v1` is shipped as the production prompt profile. It is
modular, but every request reaches the same cognitive VLM contract; there is no
separate intent router or legacy technical-intent pipeline.

## Installation

Use one Python environment consistently for dependency installation, `colcon
build` and runtime. Source the ROS distribution installed on the machine.

```bash
cd flzat_high_system_ws

source /opt/ros/$ROS_DISTRO/setup.bash

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install
source install/setup.bash
```

For an NVIDIA host, install the PyTorch wheel matching its CUDA runtime before
installing `requirements.txt`.

## Portable test paths

The commands below contain no username-specific absolute path:

```bash
cd flzat_high_system_ws
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash

export FLZAT_WS="$PWD"
export FLZAT_MEMORY_ROOT="$FLZAT_WS"
export FLZAT_OUTPUT_ROOT="$FLZAT_WS/test_output"
export FLZAT_TEST_IMAGE="$FLZAT_WS/images/image1.png"
```

`storage_root` points to the parent of `demo_1`, not directly to its Chroma
directory.

## Full pipeline: microphone, camera and real Qwen3-VL

This starts the complete High pipeline and uses the Low mock only as the Action
A server:

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:="$FLZAT_MEMORY_ROOT" \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  output_root:="$FLZAT_OUTPUT_ROOT" \
  embedding_backend:=sentence_transformer \
  enable_text_command:=false \
  use_fake_speech:=false \
  use_fake_vlm:=false \
  use_low_mock:=true \
  frame_source:=camera \
  run_camera_node:=true \
  run_vad_node:=true \
  run_stt_node:=true \
  run_tts_node:=true \
  model_id:=Qwen/Qwen3-VL-2B-Instruct \
  device:=cuda \
  dtype:=bfloat16 \
  quantization:=none \
  max_new_tokens:=256
```

## Cloud test: text, static image, RAG and real Qwen3-VL

This bypasses microphone/VAD/STT and uses a bundled static image. The text is
still processed by the normal cognitive manager, RAG, prompt composer,
validator, artifact writer and Action A flow.

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:="$FLZAT_MEMORY_ROOT" \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  output_root:="$FLZAT_OUTPUT_ROOT" \
  embedding_backend:=sentence_transformer \
  enable_text_command:=true \
  use_fake_speech:=true \
  use_fake_vlm:=false \
  use_low_mock:=true \
  frame_source:=static \
  static_image_path:="$FLZAT_TEST_IMAGE" \
  model_id:=Qwen/Qwen3-VL-2B-Instruct \
  device:=cuda \
  dtype:=bfloat16 \
  quantization:=none \
  max_new_tokens:=256
```

Send a user request from another sourced terminal:

```bash
ros2 topic pub --once \
  /flzat/high_system/text_command \
  std_msgs/msg/String \
  "{data: 'get the book on the table'}"
```

## Local test without loading Qwen

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:="$FLZAT_MEMORY_ROOT" \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  output_root:="$FLZAT_OUTPUT_ROOT" \
  enable_text_command:=true \
  use_fake_speech:=true \
  use_fake_vlm:=true \
  use_low_mock:=true \
  frame_source:=static \
  static_image_path:="$FLZAT_TEST_IMAGE"
```

The default fake response is deterministic and intended to verify transport,
state and artifact handling—not model quality.

## Inspect the High-to-Low JSON

Action A sends an absolute JSON path to Low. It does not publish the complete
payload on a legacy command topic. Watch newly committed artifacts with:

```bash
python3 tools/pretty_handoff_watcher.py \
  --output-root "$FLZAT_OUTPUT_ROOT"
```

A handoff has this high-level shape:

```json
{
  "schema_version": "flzat.high_handoff.v2",
  "handoff_id": "task-0001-c01",
  "user_request": {
    "original": "get the book on the table",
    "clarifications": []
  },
  "goal": {
    "description": "Bring the requested book from the table to the user",
    "constraints": [],
    "missing_information": []
  },
  "context": {
    "environment_id": "demo_1",
    "map_id": "demo_1_test",
    "source": "environment_memory",
    "current_positions_confirmed": false
  },
  "entities": []
}
```

When RAG references are selected, `entities` contains `id` (the memory object ID),
`label`, and nonempty description, attributes, relationships and scene. Each
entity retains its own trusted `position`/`frame_id` when available. Relationships
remain the original strings; High does not invent structured relation targets.
Coordinates come from the validated snapshot, never the VLM. Stored observations
do not imply confirmed current positions. Empty entity details and duplicate
description/label text are omitted; meaningful descriptions are retained.

The VLM output schema and its `selected_context_refs` are unchanged. High removes
transport/debug redundancy only when assembling the Low file. `goal.constraints`
and `goal.missing_information` remain arrays, and clarification question/answer
pairs remain under `user_request` so user details are not lost.

Action A still carries `run_id`, `task_id`, `handoff_id`, and `json_path`.
The compact file keeps only `handoff_id`; its required layout is
`<output_root>/<run_id>/<task_id>/<handoff_id>.json`. The Low mock verifies those
directory/file names against Goal A as well as the payload ID and v2 schema.
Shared mounts may change the root but must preserve this suffix.

Full snapshot scores, observation times, quality and reference mappings are
retained locally in `<run_id>/<task_id>/<request_id>/handoff_audit.json`.
`handoffs.jsonl` remains a byte-identical journal of compact handoffs. Neither
audit file is the path sent to Low. The watcher supports old v1 files for viewing,
but new handoffs and the Low mock use v2; production Low must update its reader.

## ROS contracts

### Action A — `/llm/handle_intent`

- High is the client; Low is the server.
- Goal: `run_id`, `task_id`, `handoff_id`, absolute `json_path`.
- Feedback stages: `deciding`, `validating`, `planning`, `waiting_for_user`.
- Business outcomes: `STARTED_FAST`, `STARTED_PLAN`, `REFUSED`, `FAILED`,
  `CANCELED`.
- `STARTED_*` means Low started execution; it does not mean task completion.

### Action B — `/brain/ask_user`

- Low is the client; High is the server.
- Kinds: `ASK`, `CONFIRM`, `INFORM`, `REFUSE`.
- `ASK`/`CONFIRM`: speak, listen and return the answer to the same B request.
- `INFORM`/`REFUSE`: speak and finish without waiting for STT.
- Only one B session is admitted at a time; there is no B queue.
- A B answer never creates a new A goal.

See [docs/HIGH_LOW_ACTION_CONTRACT.md](docs/HIGH_LOW_ACTION_CONTRACT.md) for
the exact mock-tested rules and items that still require agreement with the
production Low team.

## State and cancellation

Observe concise state/action events:

```bash
ros2 topic echo /flzat/high_system/status std_msgs/msg/String --full-length
```

Stop/cancel bypasses normal BUSY admission:

```bash
ros2 topic pub --once \
  /flzat/high_system/cancel \
  std_msgs/msg/String \
  "{data: ''}"
```

Canceling a still-active Action A waits for the Action result. After Low has
returned `STARTED_*`, this phase has no separate production contract for
monitoring or canceling Low execution.

## Tests

Run logic and mock tests:

```bash
python3 -m pytest -q src
```

Build all affected ROS packages:

```bash
colcon build --packages-up-to flzat_high_bringup --symlink-install
```

The two live ROS integration tests are opt-in and should use an isolated domain:

```bash
export ROS_DOMAIN_ID=93
export FLZAT_RUN_ROS_TESTS=1
python3 -m pytest -q \
  src/flzat_high_bringup/test/test_routing_ros_integration.py \
  src/flzat_high_bringup/test/test_step2_actions_ros_integration.py
```

Real Qwen quality and latency must be measured on the target GPU, but they use
the same manager, prompt, validator, artifact and Action A/B path shown above.

## Deferred work

- production Low LLM, planner, skills, BT and controllers;
- execution-status/cancel protocol after Action A returns `STARTED_*`;
- multiple concurrent B sessions or a B queue;
- task replacement, topic switching, voice barge-in and multi-task scheduling;
- online semantic-memory update loop;
- real-hardware end-to-end validation with production Low.
