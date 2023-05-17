#!/usr/bin/env python3
from pathlib import Path

import setuptools
from setuptools import setup

this_dir = Path(__file__).parent

requirements = []
requirements_path = this_dir / "requirements.txt"
if requirements_path.is_file():
    with open(requirements_path, "r", encoding="utf-8") as requirements_file:
        requirements = requirements_file.read().splitlines()

# -----------------------------------------------------------------------------

setup(
    name="wyoming_openai_audio_api",
    version="0.0.1",
    description="Wyoming Server for OpenAI Audio API",
    url="http://github.com/rhasspy/rhasspy3",
    author="Gregorio Hernández Caso",
    author_email="gregoriohc@gmail.com",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="rhasspy wyoming openai",
)
