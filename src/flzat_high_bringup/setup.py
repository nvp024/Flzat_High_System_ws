from glob import glob

from setuptools import find_packages, setup


package_name = "flzat_high_bringup"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FLZAT High System",
    maintainer_email="high-system@example.com",
    description="FLZAT High System launch and deterministic mock nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "fake_stt_server = flzat_high_bringup.fake_servers:stt_main",
            "fake_tts_server = flzat_high_bringup.fake_servers:tts_main",
            "fake_vlm_server = flzat_high_bringup.fake_servers:vlm_main",
            "static_frame_server = flzat_high_bringup.static_frame_server:main",
            "low_system_mock = flzat_high_bringup.low_system_mock:main",
            "acceptance_client = flzat_high_bringup.acceptance_client:main",
        ],
    },
)
