from glob import glob

from setuptools import find_packages, setup

package_name = "flzat_speech"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FLZAT High System",
    maintainer_email="high-system@example.com",
    description="Hardware interfacing nodes: VAD, STT, TTS",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vad_node = flzat_speech.vad_node:main",
            "stt_node = flzat_speech.stt_node:main",
            "tts_node = flzat_speech.tts_node:main",
            "audio_loopback_node = flzat_speech.audio_loopback_node:main",
        ],
    },
)
