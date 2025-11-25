"""
Setup script for Basketball Trajectory Analyzer
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    # Filter out comments and empty lines
    requirements = [req for req in requirements if not req.startswith("#") and req.strip()]

setup(
    name="basketball-trajectory-analyzer",
    version="1.0.0",
    author="Basketball Analyzer Team",
    author_email="contact@basketballanalyzer.com",
    description="A comprehensive basketball trajectory analysis system using YOLO detection and physics-based prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/basketball-trajectory-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
        "performance": [
            "numba>=0.57.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "basketball-analyzer=scripts.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "basketball_analyzer": ["*.yaml", "*.yml"],
    },
)
