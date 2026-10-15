# FLZAT High System

Workspace ROS 2 thống nhất cho High-Level Cognitive Pipeline:

```text
Voice → VAD → English STT → Cognitive Manager
→ read-only RAG → modular prompt → Qwen3-VL
→ validate/normalize JSON → clarification hoặc High-Level Command
→ JSON artifact + ROS 2 topic → Low System
```

`flzat_high_system_ws` là source chính. Các workspace cũ chỉ còn là tài liệu
tham khảo và không phải runtime dependency.

## Cấu trúc chính

- `flzat_speech`: VAD, STT tiếng Anh, TTS.
- `flzat_vision_context`: camera buffer và `/vision/get_frame_near`.
- `flzat_memory`: đọc Chroma database hiện có qua `QueryMemory`.
- `flzat_cognitive_pipeline`: manager, RAG context, prompt, validation, command.
- `flzat_vlm`: Qwen3-VL server cho suy luận cognitive.
- `flzat_high_bringup`: một launch production và các adapter/mock để test.

Chỉ có một manager production: `cognitive_manager`. Chỉ có một prompt profile
cho cognitive command: `flzat_cognitive_v1/manifest.json`.

## Cài dependency và build

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

## Test local không tải Qwen và không cần camera/mic

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

Gửi lệnh ở terminal khác:

```bash
ros2 topic pub --once /flzat/high_system/text_command \
  std_msgs/msg/String "{data: 'get the book on the table'}"
```

## Test full pipeline bằng ảnh có sẵn

Đây vẫn là pipeline ROS 2 bình thường. Static frame server chỉ thay camera,
không bypass RAG, Manager, PromptComposer hay output processing.

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

Copy nguyên workspace và thư mục database lên máy GPU/cloud rồi chạy chính
những lệnh ROS 2 này; không có pipeline hoặc requirements riêng cho cloud.

## Test riêng VLM bằng ảnh + prompt

Lệnh này chỉ test model server, không test RAG/cognitive pipeline:

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

Low System subscribe `/flzat/high_system/command`, publish
`/flzat/low_system/status` và nhận `/flzat/low_system/cancel`.

ID runtime dùng dạng dễ đọc: `run-20261014T153000Z`, `task-0001`,
`task-0001-r01`, `task-0001-c01`. Artifact được lưu theo từng run để số thứ tự
không ghi đè dữ liệu của lần khởi động trước.

`qwen3_vl` là backend production duy nhất. `audio_loopback` và
`acceptance_client` chỉ là test tool dùng cùng ROS contract production.

Xem `docs/ARCHITECTURE.md`, `docs/INTERFACES.md` và `docs/TEST_REPORT.md`.
