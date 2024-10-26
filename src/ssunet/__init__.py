"""SSUnet utils package."""

from pathlib import Path

from . import constants, dataloader, exceptions, losses, models, modules
from .configs import configs, load_config
from .dataloader import BinomDataset, ValidationDataset
from .models import Bit2Bit


def train_from_config(path: Path):
    """Train the model from a configuration file."""
    config = load_config(path)
    data = config.path_config.load_data_only()
    validation_data = config.path_config.load_reference_and_ground_truth()
    model = Bit2Bit(config.model_config)
    train_data = BinomDataset(data, config.data_config, config.split_params)
    validation_data = ValidationDataset(validation_data, config.data_config)
    train_loader = config.loader_config.loader(train_data)
    validation_loader = config.loader_config.loader(validation_data)
    trainer = config.train_config.trainer
    trainer.fit(model, train_loader, validation_loader)


__all__ = [
    "configs",
    "constants",
    "dataloader",
    "exceptions",
    "losses",
    "models",
    "modules",
]
