"""Scoped compatibility for checkpoints created with repository-style imports."""

import importlib
import sys
import threading
import types
from contextlib import contextmanager

import torch

_IMPORT_LOCK = threading.RLock()


@contextmanager
def checkpoint_config_imports(config_class):
    """Resolve the two historical ModelConfig paths during checkpoint loading.

    Installed wheels expose ``ssunet`` while older training scripts saved the
    dataclass as ``src.ssunet.configs.model_config.ModelConfig``. Aliases exist
    only while loading and never replace modules already imported by an app.
    The restricted-loader allowlist covers this known dataclass alone.
    """
    base = config_class.__module__.removesuffix(".configs.model_config")
    package = importlib.import_module(base)
    configs = importlib.import_module(f"{base}.configs")
    model_config = importlib.import_module(f"{base}.configs.model_config")
    source_package = types.ModuleType("src")
    source_package.__path__ = []
    aliases = {
        "src": source_package,
        "src.ssunet": package,
        "src.ssunet.configs": configs,
        "src.ssunet.configs.model_config": model_config,
        "ssunet": package,
        "ssunet.configs": configs,
        "ssunet.configs.model_config": model_config,
    }
    allowed = [
        (config_class, "src.ssunet.configs.model_config.ModelConfig"),
        (config_class, "ssunet.configs.model_config.ModelConfig"),
    ]
    with _IMPORT_LOCK:
        added = {}
        try:
            for name, module in aliases.items():
                if name not in sys.modules:
                    sys.modules[name] = module
                    added[name] = module
            existing = set(torch.serialization.get_safe_globals())
            scoped = [entry for entry in allowed if entry not in existing]
            with torch.serialization.safe_globals(scoped):
                yield
        finally:
            for name, module in reversed(tuple(added.items())):
                if sys.modules.get(name) is module:
                    del sys.modules[name]
