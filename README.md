# FLZAT High System

`flzat_high_system_ws` is the main ROS 2 workspace for the FLZAT High-Level
Cognitive Pipeline. It receives an English voice or text command, gathers the
required visual and memory context, calls Qwen3-VL, validates the model output,
and publishes a normalized JSON command for the Low System.

```text
Microphone or text command
  -> VAD
  -> English STT
  -> Safety Guard
  -> Intent Router
  -> Optional text-only Qwen routing (uncertain input or pending clarification)
  -> Context selection (RAG / image / dialogue as required)
  -> Modular Prompt Composer
  -> Qwen3-VL
  -> JSON validation and normalization
  -> TTS response or High-Level Command topic
```

This repository is self-contained at runtime. Older FLZAT workspaces are
references only and are not imported by this workspace.

## Current status

Last updated: **20 October 2026**

Available now:

- complete ROS 2 voice path from VAD and English STT to cognitive inference;
- direct text-command input for testing without a microphone, VAD, or STT;
- live-camera and static-image frame providers;
- semantic intent routing for task commands, visual QA, general QA, social
  dialogue, with Qwen fallback for uncertain routing and clarification replies;
- read-only retrieval from an existing Chroma semantic-memory database;
- modular, manifest-driven prompt composition;
- real Qwen3-VL and fake VLM backends;
- strict JSON validation, bounded repair attempts, and normalized command
  publication;
- TTS output and a Low System mock for end-to-end testing;
- per-request artifacts and runtime metrics.

Not implemented yet:

- continuous memory updates from new observations;
- reconciliation, invalidation, and deduplication of changing environment data;
- a complete long-term dynamic context loop;
- integration with the production Low System.

The current memory integration is read-only. It can retrieve existing semantic
records but does not update the Chroma database.

### Conditional Qwen routing and clarification

Confident embedding routes proceed directly to the selected prompt profile.
An embedding `clarification` result now triggers a short, text-only Qwen
classification call before any RAG retrieval or camera lookup. Every reply while
a clarification is pending also goes through this step, even if its standalone
embedding route is confident.

The routing call selects `general_qa`, `visual_qa`, `task_command`, or
`social_dialogue`, decides `continue` versus `new_request`, and asks a relevant
question only if information is actually missing. An unresolved category may be
`null` while asking for clarification. There is no automatic
`clarification -> task_command` conversion.
When there is no pending task, Manager normalizes the relationship to
`new_request` even if the model returns `continue`. This does not change the
selected mode or relax validation of other routing fields.

For a continuation, the main inference receives the original transcript and
question/answer history with the resolved mode's specialized prompt. An
independent request cancels the pending task and starts a new task with clean
history. The same loaded VLM serves both calls; confident requests still need
only the main inference. `stop`/cancel bypasses routing inference.

Example: `Who is the president?` -> `Which country do you mean?` -> `America`
stays in general QA. Asking `What is the capital of Japan?` instead starts a new
general-QA task. General QA does not query object memory or request an image.
This routing change does not add a live external-knowledge provider.

Routing goals use `<request_id>-routing`; main inference keeps `<request_id>`.
The request directory contains `routing_prompt.txt`, `routing_response.txt`,
and `resolved_routing.json` (when routing succeeds), alongside normal inference
artifacts. For a topic switch, `routing_source_request` links the new task to
the classification performed under the previous task. Initial `router.json`
records the embedding decision; `resolved_routing.json` records the final mode
and effective context policy. Invalid routing output fails closed without RAG
or Low-System dispatch. Routing and main inference share the request timeout.

Fake VLM uses separate routing and main response fixtures. Configure the routing
fixture with `fake_routing_response` at launch, or set the running node:

```bash
ros2 param set /fake_vlm_server routing_response_text \
  '{"mode":"general_qa","relation_to_pending":"new_request","needs_clarification":false,"clarification_question":null}'
ros2 param set /fake_vlm_server response_text \
  '{"reply":"This is a scripted general-QA response.","speak":true,"requires_robot_action":false}'
```

Use `relation_to_pending:"continue"` to test a clarification answer. The fake
server also accepts independent `routing_response_sequence` and
`response_sequence` string-array parameters for scripted multi-turn tests.
These fixtures test orchestration, not semantic classification accuracy.

## Workspace layout

```text
flzat_high_system_ws/
|-- demo_1/                         Ready-to-use demo Chroma/RAG database
|-- images/                         Static images for repeatable VLM tests
|-- models/                         Local Whisper and Silero VAD weights
|-- src/
|   |-- flzat_high_interfaces/      Custom ROS 2 messages, services and actions
|   |-- flzat_speech/               VAD, English STT, TTS and speech ingress
|   |-- flzat_vision_context/       Camera buffer and GetFrameNear service
|   |-- flzat_memory/               Read-only Chroma/RAG retrieval
|   |-- flzat_cognitive_pipeline/   Router, context, prompts and manager
|   |-- flzat_vlm/                  Qwen3-VL action server
|   `-- flzat_high_bringup/         Launch files, adapters and test mocks
|-- tools/                          Standalone development and inspection tools
|-- requirements.txt                Python dependencies
`-- README.md
```

The primary production entry point is:

```text
src/flzat_high_bringup/launch/high_system.launch.py
```

## Important ROS 2 interfaces

| Interface | Type | Purpose |
|---|---|---|
| `/flzat/high_system/text_command` | Topic | Direct text input for testing |
| `/flzat/high_system/command` | Topic | Validated JSON command for the Low System |
| `/flzat/high_system/status` | Topic | High System state transitions |
| `/vlm/raw_response` | Topic | Raw text returned by the VLM backend |
| `/vlm/run` | Action | VLM inference request |
| `/environment_memory/query` | Service | Read-only semantic-memory query |
| `/vision/get_frame_near` | Service | Frame nearest to a request timestamp |

Only actionable task commands are published to
`/flzat/high_system/command`. Visual QA, general QA, social dialogue, and
clarification normally produce a user-facing response instead of a Low System
command.

## Supported environment

The primary development environment is:

- Ubuntu 24.04;
- ROS 2 Jazzy;
- Python 3.12 in a Conda environment named `py312`;
- an NVIDIA CUDA GPU for real Qwen3-VL inference.

The architecture can run without a camera and microphone by using direct text
input and a static image. Fake speech and fake VLM backends are also available
for CPU-only integration tests.

## Installation

### 1. Install system packages

```bash
sudo apt update
sudo apt install -y \
  git \
  git-lfs \
  python3-rosdep \
  python3-colcon-common-extensions \
  ffmpeg \
  espeak-ng \
  libsndfile1 \
  portaudio19-dev \
  libgl1
```

Initialize Git LFS and download the tracked model files:

```bash
git lfs install
git lfs pull
```

The following local model files are required for the full voice pipeline:

```text
models/base.en.pt
models/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit
models/tts/en_US-lessac-medium.onnx
models/tts/en_US-lessac-medium.onnx.json
```

Download the default neural English voice after installing `requirements.txt`:

```bash
python3 -m piper.download_voices \
  --download-dir models/tts \
  en_US-lessac-medium
```

Download the default English Whisper checkpoint:

```bash
python3 -c "import whisper; whisper.load_model('base.en', download_root='models')"
```

The real TTS node uses Piper with the `en_US-lessac-medium` English voice.

Qwen3-VL and the sentence-transformer model are downloaded into the Hugging
Face cache on first use unless they are already cached.

### 2. Create the Python environment

Create the environment once:

```bash
conda create -n py312 python=3.12 -y
```

Activate it before installing, building, or running the workspace:

```bash
conda activate py312
source /opt/ros/jazzy/setup.bash
```

For an NVIDIA machine, install the PyTorch and TorchVision wheels appropriate
for its CUDA driver first. Then install the repository dependencies:

```bash
cd <path-to-repository>/flzat_high_system_ws

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
python3 -m pip check
```

Do not independently upgrade `empy`. ROS 2 interface generation requires the
pinned `empy==3.3.4` API; incompatible Empy versions can cause errors such as
`module 'em' has no attribute 'BUFFERED_OPT'`.

### 3. Install ROS dependencies

`rosdep init` is required only once per machine:

```bash
sudo rosdep init
rosdep update
```

Install package dependencies from the workspace root:

```bash
rosdep install \
  --from-paths src \
  --ignore-src \
  -r -y \
  --skip-keys ament_python
```

### 4. Build the workspace

```bash
colcon build --symlink-install
source install/setup.bash
```

The Conda environment must be active while building. This ensures the generated
ROS 2 Python executables use the interpreter containing PyTorch, ChromaDB, and
the model dependencies.

### 5. Run repository tests

```bash
colcon test
colcon test-result --verbose
```

## Included test data

The repository includes a ready-to-use semantic-memory database and a set of
static test images. They allow a new user to install, build, and exercise the
complete cognitive path without first generating a new environment memory.

The included database has this layout:

```text
flzat_high_system_ws/
|-- demo_1/
|   |-- manifest.json
|   |-- observations.jsonl
|   |-- maps/
|   `-- chroma/
|       `-- chroma.sqlite3
`-- images/
    |-- image1.png
    `-- ...
```

The bundled manifest defines:

```text
environment_id = demo_1
map_id         = demo_1_test
object_count   = 59
```

For this bundled database, `storage_root` is the workspace root because the
`demo_1` environment directory is located directly inside it.

An external production database may still be used. A complete environment
directory must contain a completed manifest, observations, map data, and its
Chroma database. When using external data, the launch parameters must satisfy
these rules:

- `storage_root` is the parent directory containing the environment directory;
- `environment_id` is the environment directory name;
- `map_id` matches the completed map recorded in `manifest.json`.

There is no automatic fallback to the newest database. Both `environment_id`
and `map_id` are required intentionally.

## Portable path configuration

The commands below do not contain a username or a machine-specific home path.
Enter the workspace and define the paths once in each new terminal session.
The default values use the bundled `demo_1` database and `images/image1.png`:

```bash
cd <path-to-repository>/flzat_high_system_ws

export FLZAT_WS="$(pwd)"
export FLZAT_MEMORY_ROOT="${FLZAT_WS}"
export FLZAT_OUTPUT_ROOT="${FLZAT_WS}/runtime_output"
export FLZAT_TEST_IMAGE="${FLZAT_WS}/images/image1.png"
```

No path changes are required for the included test data. To use an external
database or another image, override only `FLZAT_MEMORY_ROOT` or
`FLZAT_TEST_IMAGE`. After this step, all launch commands use the same portable
variables.

Every new terminal that uses this workspace must also run:

```bash
conda activate py312
source /opt/ros/jazzy/setup.bash
source "${FLZAT_WS}/install/setup.bash"
```

## Run the full pipeline

This mode starts the real microphone input, VAD, English STT, camera, RAG
retrieval, Qwen3-VL, output validation, and TTS. It assumes camera index `0`
and the default audio devices are available.

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
  run_camera_node:=true \
  run_vad_node:=true \
  run_stt_node:=true \
  run_tts_node:=true \
  piper_model_path:="$FLZAT_WS/models/tts/en_US-lessac-medium.onnx" \
  tts_output_sample_rate:=48000 \
  tts_sentence_pause_ms:=200 \
  frame_source:=camera \
  model_id:=Qwen/Qwen3-VL-2B-Instruct \
  device:=cuda \
  dtype:=bfloat16 \
  quantization:=none \
  max_new_tokens:=256
```

Use `use_low_mock:=true` while the production Low System is not connected.
Change it to `false` when a real Low System subscribes to the command topic and
publishes its own execution status.

### Speech-only check

Test microphone → VAD → Whisper → Piper without camera, RAG, or VLM:

```bash
ros2 launch flzat_speech speech_loopback_test.launch.py \
  whisper_language:=en \
  whisper_beam_size:=5 \
  whisper_max_decode_tokens:=64 \
  vad_silence_ms:=500
```

The default checkpoint remains `models/base.en.pt`; override it with
`whisper_model_path:=/path/to/model.pt`. Loopback echoes accepted transcripts
only and does not publish commands to the cognitive pipeline. Do not run it
alongside the full pipeline (both own the microphone and speech actions).

Speech launch settings are shared in `flzat_speech/launch_config.py`:

- `pending_audio_max_age_s:=2.0`: retain only the latest pending utterance;
  discard it if it has waited too long since this node received it.
- `stt_timeout_s:=30.0`: discard late results and request cancellation. This
  does **not** interrupt Whisper computation; the server stays busy until it returns.
- `whisper_max_decode_tokens:=64`: reject a decode window reaching this cap
  instead of forwarding potentially truncated text. Increase it for longer commands.
- `tts_resume_delay_s:=0.8`: discard microphone backlog while TTS speaks and
  during cooldown. This is half-duplex, not echo cancellation; spoken stop
  commands cannot be heard during playback.
- Piper settings (`piper_length_scale`, `tts_sentence_pause_ms`, model path,
  output rate) are shared by full, local I/O, services and loopback launches.
- `tts_speech_timeout_s:=60.0` bounds the Piper action; `tts_timeout_s:=65.0`
  bounds client waiting (leave room for `tts_cancel_grace_s:=2.0`). The manager
  checks acceptance/results and keeps at most the latest pending reply; replaced,
  expired or failed replies are logged. A global text `stop` also cancels TTS.

Low-confidence segments, repetitive output and implausible word rates reject
the **whole utterance**, rather than removing words such as “not”. Logs include
the rejection reason. These are safeguards, not a guarantee of transcription
accuracy; microphone/noise testing is still required. VAD also discards stale
mic backlog (>0.5 s). A dropped-sample gap or `speech_max_duration_s:=30.0`
discards the interrupted utterance and ignores its tail until silence; truncated
audio is not sent to STT. Graceful shutdown waits for active Whisper/Piper work
before destroying its ROS handles; an unresponsive engine can delay shutdown.

### Timestamped speech and camera context

`/flzat/high_system/stt_result` now uses
`flzat_high_interfaces/msg/SpeechTranscript` (`header`, `transcript`), not
`std_msgs/String`. `RoutedCommand` also carries this header. Rebuild all packages
and restart all participating nodes after updating; do not mix old/new interfaces.
`/flzat/high_system/text_command` remains `std_msgs/String`.

For speech, the timestamp is the VAD segment endpoint before STT, approximately
the end of the utterance including endpoint silence (not sample-accurate audio
capture synchronization). Text commands use their receipt time. Vision requests
use that original stamp, including after routing inference, rather than the time
STT finishes. The camera buffer defaults to `camera_buffer_duration_s:=35.0`;
larger buffers cost RAM. If inference/routing outlasts the buffer, the request
fails explicitly instead of silently selecting a newer scene. Camera and speech
publishers must share the same ROS clock; split machines need synchronized clocks.
Static images remain a test fixture, not a time-synchronized live camera.

Test the Piper voice without running VAD, STT, RAG, or Qwen:

```bash
ros2 run flzat_speech tts_node --ros-args \
  -p piper_model_path:="$FLZAT_WS/models/tts/en_US-lessac-medium.onnx" \
  -p output_sample_rate:=48000
```

In another sourced terminal, send one English sentence:

```bash
ros2 action send_goal --feedback \
  /tts_action \
  flzat_high_interfaces/action/TextToSpeech \
  "{text: 'Hello. I am FLZAT, your robotic assistant.'}"
```

## Inspect the final command JSON

Start the JSON viewer before speaking or publishing a command:

```bash
cd "$FLZAT_WS"
python3 tools/pretty_command_listener.py
```

It subscribes to `/flzat/high_system/command` and prints each final command as
indented JSON. A typical message is:

```json
{
  "schema_version": "flzat.high_command.v2",
  "task_id": "task-0001",
  "request_id": "task-0001-r01",
  "command_id": "task-0001-c01",
  "intent": "fetch_object",
  "target": {
    "type": "environment_object",
    "name": "book",
    "memory_id": "83c7998a-f665-48b3-ba1a-1fb9ae80222f"
  },
  "trusted_context": {
    "environment_id": "demo_1",
    "map_id": "demo_1_test",
    "room": "indoor_environment",
    "position": {
      "frame_id": "map",
      "x": 9.966,
      "y": -0.642,
      "z": 0.595
    }
  },
  "execution_mode": "direct",
  "missing_information": [],
  "status": "validated"
}
```

To inspect the model response before command assembly, use:

```bash
ros2 topic echo --full-length /vlm/raw_response std_msgs/msg/String
```

The raw VLM response and the final High System command are intentionally
different. The Cognitive Manager validates the model response and enriches the
final command with trusted coordinates and metadata retrieved from memory.

## Test without VAD and STT

This mode disables microphone capture, VAD, and STT. A text command enters the
same Intent Router, RAG, prompt, Qwen3-VL, validation, and command-publication
path as an STT transcript. The live camera and TTS remain enabled.

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:="$FLZAT_MEMORY_ROOT" \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  output_root:="$FLZAT_OUTPUT_ROOT" \
  embedding_backend:=sentence_transformer \
  enable_text_command:=true \
  use_fake_speech:=false \
  use_fake_vlm:=false \
  use_low_mock:=true \
  run_camera_node:=true \
  run_vad_node:=false \
  run_stt_node:=false \
  run_tts_node:=true \
  frame_source:=camera \
  model_id:=Qwen/Qwen3-VL-2B-Instruct \
  device:=cuda \
  dtype:=bfloat16 \
  quantization:=none \
  max_new_tokens:=256
```

Publish a user command from another sourced terminal:

```bash
ros2 topic pub --once \
  /flzat/high_system/text_command \
  std_msgs/msg/String \
  "{data: 'get the book on the table'}"
```

Other examples:

```bash
ros2 topic pub --once /flzat/high_system/text_command \
  std_msgs/msg/String "{data: 'go to the bedroom'}"

ros2 topic pub --once /flzat/high_system/text_command \
  std_msgs/msg/String "{data: 'stop'}"

ros2 topic pub --once /flzat/high_system/text_command \
  std_msgs/msg/String "{data: 'what can you see in front of you'}"
```

The visual-QA example produces a user-facing response and normally does not
publish a Low System command.

## Test on a cloud machine with a static image

This mode is useful when the machine has a GPU but no camera, microphone, or
speaker. It still exercises RAG retrieval, prompt composition, real Qwen3-VL,
validation, and output processing.

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:="$FLZAT_MEMORY_ROOT" \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  output_root:="$FLZAT_OUTPUT_ROOT" \
  embedding_backend:=sentence_transformer \
  enable_text_command:=true \
  use_fake_speech:=false \
  use_fake_vlm:=false \
  use_low_mock:=true \
  run_camera_node:=false \
  run_vad_node:=false \
  run_stt_node:=false \
  run_tts_node:=false \
  frame_source:=static \
  static_image_path:="$FLZAT_TEST_IMAGE" \
  model_id:=Qwen/Qwen3-VL-2B-Instruct \
  device:=cuda \
  dtype:=bfloat16 \
  quantization:=none \
  max_new_tokens:=256
```

Send commands through `/flzat/high_system/text_command` and monitor the final
JSON with `tools/pretty_command_listener.py`.

## CPU-only integration test with fake backends

This verifies ROS 2 routing and orchestration without loading Qwen3-VL or
accessing camera and audio hardware:

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:="$FLZAT_MEMORY_ROOT" \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  output_root:="$FLZAT_OUTPUT_ROOT" \
  embedding_backend:=deterministic_test \
  enable_text_command:=true \
  use_fake_speech:=true \
  use_fake_vlm:=true \
  use_low_mock:=true \
  run_camera_node:=false \
  frame_source:=static \
  static_image_path:="$FLZAT_TEST_IMAGE"
```

## Runtime artifacts

Each request writes diagnostic artifacts below:

```text
<output_root>/<run_id>/<task_id>/<request_id>/
```

The artifacts include the transcript, routed intent, selected context, composed
prompt, raw VLM response, validation result, and final command when applicable.
They are intended for debugging and evaluation and should not be committed to
Git.
