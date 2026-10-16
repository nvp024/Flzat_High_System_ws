from glob import glob
from pathlib import Path

from setuptools import find_packages, setup


package_name = "flzat_cognitive_pipeline"


def prompt_data_files():
    groups = {}
    for path in Path("prompts").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".txt"}:
            continue
        destination = f"share/{package_name}/{path.parent.as_posix()}"
        groups.setdefault(destination, []).append(path.as_posix())
    return [(destination, sorted(files)) for destination, files in groups.items()]

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        *prompt_data_files(),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FLZAT High System",
    maintainer_email="high-system@example.com",
    description="FLZAT cognitive orchestration and trusted command pipeline",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "cognitive_manager = flzat_cognitive_pipeline.cognitive_manager:main",
            "intent_router = flzat_cognitive_pipeline.intent_router:main",
        ],
    },
)
