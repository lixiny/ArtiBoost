from setuptools import setup, find_packages

setup(
    name="anakin",
    version="0.0.1",
    python_requires=">=3.7.0",
    packages=find_packages(exclude=("manotorch", "assets", "common", "config", "data", "exp", "scripts")),
)
