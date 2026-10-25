#!/usr/bin/env python3
"""
Setup script for Synobotix VLA PoC
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="snn-vla-poc",
    version="0.1.0",
    description="Proof of Concept for Synobotix Vision-Language-Action Model",
    author="Synobotix R&D Team",
    author_email="rnd@synobotix.com",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        r for r in requirements 
        if r and not r.startswith('#') and not r.startswith('//')
    ],
    entry_points={
        "console_scripts": [
            "snn-train=scripts.train:main",
            "snn-calibrate=scripts.calibrate:main",
            "snn-evaluate=scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
