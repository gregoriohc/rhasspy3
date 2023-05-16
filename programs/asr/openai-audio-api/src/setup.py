import os

from setuptools import find_packages, setup


def get_requirements(path):
    with open(path, encoding="utf-8") as requirements:
        return [requirement.strip() for requirement in requirements]


base_dir = os.path.dirname(os.path.abspath(__file__))
install_requires = get_requirements(os.path.join(base_dir, "requirements.txt"))

setup(
    name="openai-audio-api",
    version="0.1.0",
    description="OpenAI API audio transcription",
    author="Gregorio Hernández Caso",
    python_requires=">=3.7",
    install_requires=install_requires,
    packages=find_packages(),
)
