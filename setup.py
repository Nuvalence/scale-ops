#!/usr/bin/env python
from pathlib import Path

from setuptools import setup, find_packages
import setuptools

if __name__ == "__main__":
    setuptools.setup()

cur_dir = Path(__file__).parent
long_description = (cur_dir / "README.md").read_text()

setup(
        name='scaleops',
        version='0.0.1',
        author='Richard Bellamy',
        author_email='rbellamy@pteradigm.com',
        description='Scalability Operations',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/rbellamy/scale-ops',
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
)
