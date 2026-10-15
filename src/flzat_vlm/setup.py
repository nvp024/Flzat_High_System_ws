from glob import glob

from setuptools import find_packages, setup


package_name = "flzat_vlm"

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
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FLZAT High System",
    maintainer_email="high-system@example.com",
    description="Qwen3-VL action server for the FLZAT cognitive pipeline",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vlm_node = flzat_vlm.vlm_node:main",
            "vlm_test_client = flzat_vlm.vlm_test_client:main",
        ],
    },
)
