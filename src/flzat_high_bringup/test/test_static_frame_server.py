from pathlib import Path

import pytest

from flzat_high_bringup.static_frame_server import StaticFrameServer


def test_empty_path_uses_builtin_png():
    data, image_format, source = StaticFrameServer._load_image("")
    assert data.startswith(b"\x89PNG")
    assert image_format == "png"
    assert "fixture" in source


def test_static_image_loader_validates_extension_and_content(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg-fixture")
    data, image_format, source = StaticFrameServer._load_image(str(image))
    assert data == b"jpeg-fixture"
    assert image_format == "jpeg"
    assert source == str(image.resolve())

    unsupported = tmp_path / "frame.txt"
    unsupported.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="JPEG, PNG or WebP"):
        StaticFrameServer._load_image(str(unsupported))


def test_main_guards_shutdown_cleanup_from_repeated_sigint():
    source_path = (
        Path(__file__).parents[1]
        / "flzat_high_bringup/static_frame_server.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert source.count("except KeyboardInterrupt:") >= 2
