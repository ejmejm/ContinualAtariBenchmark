from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="continual-atari-benchmark",
    version="0.1.0",
    author="Edan Meyer",
    description="A benchmark for continual learning on sequences of Atari games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ejmejm/ContinualAtariBenchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.2",
        "numpy>=1.24.4",
        "ale-py>=0.8.1",
    ],
)

