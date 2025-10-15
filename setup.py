from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="grpo-dataset-builder",
    version="0.1.0",
    author="jwest33",
    description="A comprehensive dataset builder for GRPO LoRA fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwest33/grpo-dataset-builder",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.23.2",
            "black>=23.12.1",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "embeddings": [
            "sentence-transformers>=2.2.2",
        ],
        "selenium": [
            "selenium>=4.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dataset-builder=cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "dataset_builder": [
            "generation/templates/*.yaml",
        ],
        "": ["configs/*.yaml", "templates/*.yaml"],
    },
)
