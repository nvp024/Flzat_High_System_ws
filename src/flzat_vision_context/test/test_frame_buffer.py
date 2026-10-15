from flzat_vision_context.frame_buffer import BufferedFrame, FrameRingBuffer


def test_buffer_evicts_frames_outside_duration() -> None:
    buffer = FrameRingBuffer(duration_s=2.0, max_frames=10)
    buffer.append(BufferedFrame(0, b"old", "camera"))
    buffer.append(BufferedFrame(1_000_000_000, b"middle", "camera"))
    buffer.append(BufferedFrame(3_100_000_000, b"new", "camera"))

    assert len(buffer) == 1
    assert buffer.nearest(3_000_000_000, 0.2).jpeg_data == b"new"


def test_buffer_enforces_frame_count_and_max_age() -> None:
    buffer = FrameRingBuffer(duration_s=10.0, max_frames=2)
    buffer.append(BufferedFrame(1_000_000_000, b"one", "camera"))
    buffer.append(BufferedFrame(2_000_000_000, b"two", "camera"))
    buffer.append(BufferedFrame(3_000_000_000, b"three", "camera"))

    assert len(buffer) == 2
    assert buffer.nearest(2_100_000_000, 0.2).jpeg_data == b"two"
    assert buffer.nearest(9_000_000_000, 1.0) is None
