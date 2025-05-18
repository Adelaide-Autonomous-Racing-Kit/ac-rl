from setuptools import setup, find_packages

setup(
    name="ac-rl",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scipy",
        "ruamel.yaml",
        "PyQt6",
    ],
)
