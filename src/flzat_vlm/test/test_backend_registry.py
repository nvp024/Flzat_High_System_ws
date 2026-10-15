import pytest

from flzat_vlm.backends import available_backends, create_backend
from flzat_vlm.backends.base import BackendConfig


def test_qwen3_vl_backend_is_created_without_loading_model_libraries() -> None:
    config = BackendConfig(model_id="Qwen/Qwen3-VL-2B-Instruct")
    backend = create_backend("qwen3_vl", config)
    assert backend.name == "qwen3_vl"
    assert backend.device_description == "not loaded"
    assert backend.model_revision == "main"
    assert available_backends() == ("qwen3_vl",)


def test_unknown_backend_has_helpful_error() -> None:
    with pytest.raises(
        ValueError,
        match="Available backends: qwen3_vl",
    ):
        create_backend("qwen-not-added-yet", BackendConfig(model_id="unused"))
