"""Setup file for ssunet."""

from setuptools import find_packages, setup

setup(
    name="ssunet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
