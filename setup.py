"""Setup configuration for TAS-Research-Suite."""
from setuptools import setup, find_packages

setup(
    packages=find_packages(include=["source", "source.*"]),
)
