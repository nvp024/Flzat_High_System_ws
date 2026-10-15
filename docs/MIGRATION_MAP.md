# Migration Map

Source đã tái sử dụng:

| Source PoC | Module hiện tại |
|---|---|
| `audio_pipeline` | `flzat_speech` |
| `vision_pipeline` | `flzat_vision_context` |
| `vlm_pipeline/backends` | `flzat_vlm.backends` |
| VLM cancellation/priority broker | `flzat_vlm` |
| environment memory storage/retrieval | `flzat_memory` |
| prompt/manager ideas | `flzat_cognitive_pipeline` |

Sau consolidation, code chạy chỉ nằm trong `flzat_high_system_ws`. Những bản
thử `foundation_manager`, `multimodal_manager`, companion prompt fallback và
standalone cloud pipeline đã bị loại bỏ. `cognitive_manager`,
`flzat_cognitive_v1`, `high_system.launch.py` và `requirements.txt` là bốn đầu
mối canonical.

`RunVlm` dùng contract tối thiểu một ảnh/một transcript; wrapper
`MultimodalEvent` của PoC đã được loại bỏ. Prompt optional dialogue chỉ được
ghép khi có dữ liệu. Grounding/bounding-box thử nghiệm và các backend model cũ
đã được loại khỏi runtime canonical.

Workspace cũ không bị sửa và không phải dependency lúc chạy.
