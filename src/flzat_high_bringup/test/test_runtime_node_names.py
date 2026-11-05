"""Keep ROS node names aligned without renaming executables or interfaces."""

import ast
from pathlib import Path

import pytest
import yaml


SRC = Path(__file__).resolve().parents[2]
NODE_NAMES = {
    "cognitive_manager": "vlm_brain",
    "vlm_node": "qwen_inference",
    "speech_ingress": "speech_input",
    "camera_node": "perception",
    "memory_query_node": "knowledge_server",
    "tts_node": "speech_output",
}


def parse(relative_path):
    return ast.parse((SRC / relative_path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path,expected", [
    ("flzat_cognitive_pipeline/flzat_cognitive_pipeline/cognitive_manager.py", "vlm_brain"),
    ("flzat_vlm/flzat_vlm/vlm_node.py", "qwen_inference"),
    ("flzat_speech/flzat_speech/speech_ingress.py", "speech_input"),
    ("flzat_speech/flzat_speech/tts_node.py", "speech_output"),
    ("flzat_vision_context/flzat_vision_context/camera_node.py", "perception"),
    ("flzat_memory/flzat_memory/retrieval/memory_query_server.py", "knowledge_server"),
])
def test_direct_node_construction_uses_architecture_name(path, expected):
    names = []
    for call in ast.walk(parse(path)):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "__init__"
            and isinstance(call.func.value, ast.Call)
            and isinstance(call.func.value.func, ast.Name)
            and call.func.value.func.id == "super"
            and call.args
        ):
            continue
        name = call.args[0]
        if isinstance(name, ast.IfExp):
            # The standalone speech loopback keeps its diagnostic node name.
            assert ast.literal_eval(name.body) == "audio_loopback_node"
            name = name.orelse
        names.append(ast.literal_eval(name))
    assert expected in names
    assert not set(names).intersection(NODE_NAMES)


def test_all_launches_keep_executables_but_use_new_runtime_names():
    seen = set()
    for path in SRC.glob("*/launch/*.launch.py"):
        for call in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "Node"
            ):
                continue
            kwargs = {item.arg: item.value for item in call.keywords}
            executable = kwargs.get("executable")
            if not isinstance(executable, ast.Constant):
                continue
            if executable.value in NODE_NAMES:
                assert ast.literal_eval(kwargs["name"]) == NODE_NAMES[executable.value], path
                seen.add(executable.value)
    assert seen == set(NODE_NAMES)


def test_parameter_yaml_targets_renamed_nodes():
    config = yaml.safe_load(
        (SRC / "flzat_high_bringup/config/high_system.yaml").read_text(encoding="utf-8")
    )
    assert set(config) == {"vlm_brain", "knowledge_server"}
    assert config["vlm_brain"]["ros__parameters"]["rag_top_k"] == 5
    assert config["knowledge_server"]["ros__parameters"]["embedding_device"] == "cpu"
