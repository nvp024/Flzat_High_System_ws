from setuptools import find_packages, setup


package_name = "flzat_vision_context"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/flzat_vision_context.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FLZAT High System",
    maintainer_email="high-system@example.com",
    description="Non-blocking camera capture and nearest-frame service",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_node = flzat_vision_context.camera_node:main",
        ],
    },
)
