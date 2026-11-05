"""VAD shutdown releases the microphone without logging to a closed context."""

from unittest.mock import Mock

import pytest
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions

from flzat_speech import vad_node


@pytest.mark.parametrize("interrupt,context_alive", [
    (KeyboardInterrupt, True),
    (KeyboardInterrupt, False),
    (ExternalShutdownException, False),
])
def test_interrupt_cleans_up_and_never_shuts_down_twice(monkeypatch, interrupt, context_alive):
    node = Mock()
    ros = Mock()
    ros.spin.side_effect = interrupt()
    ros.ok.return_value = context_alive
    monkeypatch.setattr(vad_node, "VadNode", Mock(return_value=node))
    monkeypatch.setattr(vad_node, "rclpy", ros)

    vad_node.main(args=[])

    ros.init.assert_called_once_with(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node.destroy_node.assert_called_once()
    ros.try_shutdown.assert_called_once()
    ros.shutdown.assert_not_called()
    if context_alive:
        node.get_logger().info.assert_called_once()
    else:
        node.get_logger.assert_not_called()


def test_context_cleanup_runs_even_if_microphone_cleanup_fails(monkeypatch):
    node = Mock()
    node.destroy_node.side_effect = RuntimeError("microphone cleanup failed")
    ros = Mock()
    monkeypatch.setattr(vad_node, "VadNode", Mock(return_value=node))
    monkeypatch.setattr(vad_node, "rclpy", ros)

    with pytest.raises(RuntimeError, match="microphone cleanup failed"):
        vad_node.main()

    ros.try_shutdown.assert_called_once()
