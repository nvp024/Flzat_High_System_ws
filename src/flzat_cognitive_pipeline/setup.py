from glob import glob

from setuptools import find_packages, setup


package_name = "flzat_cognitive_pipeline"

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
        (
            f"share/{package_name}/prompts/flzat_cognitive_v1",
            glob("prompts/flzat_cognitive_v1/*.*"),
        ),
        *[
            (
                f"share/{package_name}/prompts/flzat_cognitive_v1/{section}",
                glob(f"prompts/flzat_cognitive_v1/{section}/*.txt"),
            )
            for section in ("system", "policies", "tasks", "context", "output")
        ],
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
        ],
    },
)
