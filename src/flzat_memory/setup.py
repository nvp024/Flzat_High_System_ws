from setuptools import find_packages, setup


package_name = "flzat_memory"

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
        (f"share/{package_name}/launch", ["launch/memory_query.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FLZAT High System",
    maintainer_email="high-system@example.com",
    description="Read-only Chroma RAG adapter and ROS 2 query service",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "memory_query_node = flzat_memory.retrieval.memory_query_server:main",
        ],
    },
)
