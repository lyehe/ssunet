"""Configure pytest for the project."""

import os


def pytest_configure(config):
    """Configure pytest."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FORCE_CPU"] = "1"
