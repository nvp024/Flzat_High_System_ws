# Architecture

## Một runtime flow

```text
/audio_events → CognitiveManager → STT
text hook ────────────────────────┘
                    │
                    ├─ QueryMemory → read-only Chroma
                    ├─ GetFrameNear → camera hoặc static-frame adapter
                    ├─ PromptComposer → RunVlm → output normalization/validation
                    ├─ clarification/response → TTS
                    └─ trusted command → JSON/JSONL + ROS topic → Low System
```

Không còn `foundation_manager` hoặc `multimodal_manager`. VLM node không tự
chọn prompt hội thoại; nó chỉ nhận prompt đã được `PromptComposer` tạo và kiểm
tra SHA-256 trước inference.

Prompt profile chỉ ghép context đang thật sự có: RAG luôn hiện diện và dialogue
chỉ được thêm khi có dữ liệu. Ảnh camera được gửi trực tiếp qua `RunVlm`, không
bị chuyển thành text context.

## Trust boundary

- Qwen chọn intent, strategy và `context_ref`.
- Tọa độ không được lấy từ model output.
- `ContextRagManager` đóng băng kết quả RAG theo request nội bộ.
- `TrustedCommandAssembler` resolve tọa độ từ snapshot đáng tin cậy.
- Output chấp nhận JSON thuần hoặc đúng một outer fence ` ```json ... ``` `;
  nội dung bên trong vẫn phải đúng strict schema, không tự đoán hoặc repair.
- Artifact được tách theo `output_root/run_id/task_id/command_id.json`; audit
  JSONL nằm trong chính thư mục của run để ID tuần tự không đụng lần chạy cũ.

## Test adapters

`static_frame_server` implement đúng `/vision/get_frame_near`, vì vậy ảnh file
đi qua cùng đường với camera thật. Fake STT/VLM/TTS/Low dùng cùng ROS contract
production. Chúng là adapter test, không phải pipeline thứ hai.

Qwen3-VL là backend production duy nhất phía sau `VlmNode`.

## Phạm vi

High System sở hữu voice, context/RAG, prompt, reasoning, clarification và
High-Level Command. SLAM, Nav2, depth/TF localization, manipulation và actuator
thuộc Low System.
