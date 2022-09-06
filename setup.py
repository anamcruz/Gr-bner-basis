"""Build script for setuptools."""

import setuptools

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f.readlines()]
    requirements = list(
        filter(lambda line: not line.startswith("-"), requirements))

setuptools.setup(name="grobner",
                 version="0.1.0",
                 description="Buchberger algorithm optimization.",
                 packages=setuptools.find_packages(),
                 install_requires=requirements)
