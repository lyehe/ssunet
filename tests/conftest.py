"""Configure pytest for the project."""

import os


def pytest_configure(config):
    """Configure pytest."""
    if os.environ.get("SSUNET_TEST_NATIVE") == "1":
        # Opt-in native tests honor the caller's GPU selection and test deterministic GEMM.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORCE_CPU"] = "1"
