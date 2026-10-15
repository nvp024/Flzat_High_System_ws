"""ROS frame-provider adapter for repeatable tests with one static image."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from flzat_high_interfaces.srv import GetFrameNear


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Wl2nkwAAAAASUVORK5CYII="
)
_FORMATS = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
}


class StaticFrameServer(Node):
    """Expose a local image through the same service used by a real camera."""

    def __init__(self) -> None:
        super().__init__("static_frame_server")
        self.declare_parameter("image_path", "")
        self.declare_parameter("frame_id", "static_camera_optical_frame")
        self._frame_id = str(self.get_parameter("frame_id").value).strip()
        if not self._frame_id:
            raise ValueError("frame_id cannot be empty")
        self._data, self._format, source = self._load_image(
            str(self.get_parameter("image_path").value)
        )
        self._service = self.create_service(
            GetFrameNear,
            "/vision/get_frame_near",
            self._get_frame,
        )
        self.get_logger().info(f"Static frame provider ready: {source}.")

    @staticmethod
    def _load_image(image_path: str) -> tuple[bytes, str, str]:
        if not image_path.strip():
            return _ONE_PIXEL_PNG, "png", "built-in 1 px fixture"
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"static image not found: {path}")
        image_format = _FORMATS.get(path.suffix.casefold())
        if image_format is None:
            raise ValueError("static image must be JPEG, PNG or WebP")
        data = path.read_bytes()
        if not data:
            raise ValueError(f"static image is empty: {path}")
        return data, image_format, str(path)

    def _get_frame(self, _request, response):
        frame = CompressedImage()
        frame.header.stamp = self.get_clock().now().to_msg()
        frame.header.frame_id = self._frame_id
        frame.format = self._format
        frame.data = self._data
        response.success = True
        response.message = "static frame"
        response.frame = frame
        return response

    def destroy_node(self) -> None:
        self.destroy_service(self._service)
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[StaticFrameServer] = None
    try:
        node = StaticFrameServer()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        try:
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
