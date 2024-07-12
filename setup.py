from setuptools import setup, find_packages

import pkg_resources

import json
import pathlib


def requirements(filepath: str):
    with pathlib.Path(filepath).open() as requirements_txt:
        return [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(requirements_txt)
        ]


setup(
    name="KSR_BERT_768",
    version="0.0.1",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.11",
    install_requires=requirements("requirements.txt"),
)
