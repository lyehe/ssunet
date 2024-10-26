"""SSUnet utils package."""

from pathlib import Path

from . import constants, dataloader, exceptions, losses, models, modules
from .configs import configs, load_config
from .dataloader import BinomDataset, ValidationDataset
from .models import Bit2Bit


def train_from_config(path: Path):
    config = load_config("config/config.yml")
    data = config.path_config.load_ssunet_data()
    model = Bit2Bit(config.model_config)
    train_data = BinomDataset(data, config.data_config, config.split_params)
    train_loader = config.loader_config.loader(train_data)
    trainer = config.train_config.trainer
    trainer.fit(model, train_loader, train_loader)


__all__ = [
    "configs",
    "constants",
    "dataloader",
    "exceptions",
    "losses",
    "models",
    "modules",
]
