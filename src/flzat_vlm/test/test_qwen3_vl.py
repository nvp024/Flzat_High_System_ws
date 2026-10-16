import contextlib

from flzat_vlm.backends.base import BackendConfig, GenerationRequest
from flzat_vlm.backends.qwen3_vl import Qwen3VlBackend


class _InputIds:
    shape = (1, 3)


class _Inputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=_InputIds())

    def to(self, _device):
        return self


class _OutputIds:
    def __getitem__(self, key):
        assert key == (slice(None), slice(3, None))
        return [[101, 102]]


class _ImageProcessor:
    patch_size = 14


class _Processor:
    image_processor = _ImageProcessor()

    def __init__(self) -> None:
        self.messages = None
        self.images = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "formatted"

    def __call__(self, *, text, images=None, videos=None, **_kwargs):
        assert text == ["formatted"]
        self.images = images
        if images is not None:
            assert videos == []
        return _Inputs()

    @staticmethod
    def batch_decode(_ids, **_kwargs):
        return ['{"objects": []}']


class _Torch:
    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class _Model:
    @staticmethod
    def generate(**_kwargs):
        return _OutputIds()


def test_qwen3_vl_uses_official_vision_preprocessing_contract() -> None:
    processor = _Processor()
    observed = {}
    backend = Qwen3VlBackend(BackendConfig(model_id="fake"))
    backend._model = _Model()
    backend._processor = processor
    backend._torch = _Torch()
    backend._input_device = "cpu"
    backend._stopping_criteria_list = list

    def process(messages, **kwargs):
        observed["messages"] = messages
        observed["kwargs"] = kwargs
        return ["resized-image"], []

    backend._process_vision_info = process
    response = backend.generate(
        GenerationRequest(
            images=(object(),),
            system_prompt="json only",
            user_prompt="ground objects",
            max_new_tokens=32,
        )
    )

    assert response == '{"objects": []}'
    assert observed["kwargs"] == {"image_patch_size": 14}
    assert processor.images == ["resized-image"]
    image_part = observed["messages"][1]["content"][0]
    assert image_part["min_pixels"] == 256 * 256
    assert image_part["max_pixels"] == 768 * 768


def test_qwen3_vl_supports_text_only_generation() -> None:
    processor = _Processor()
    backend = Qwen3VlBackend(BackendConfig(model_id="fake"))
    backend._model = _Model()
    backend._processor = processor
    backend._torch = _Torch()
    backend._input_device = "cpu"
    backend._stopping_criteria_list = list

    def unexpected_vision_call(*_args, **_kwargs):
        raise AssertionError("text-only generation called vision preprocessing")

    backend._process_vision_info = unexpected_vision_call
    response = backend.generate(
        GenerationRequest(
            images=(),
            system_prompt="answer briefly",
            user_prompt="what is the capital of China",
            max_new_tokens=32,
        )
    )

    assert response == '{"objects": []}'
    assert processor.images is None
    assert processor.messages[1]["content"] == [
        {"type": "text", "text": "what is the capital of China"}
    ]
