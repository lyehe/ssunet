from dataclasses import dataclass, field
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

import torch.utils.data as dt
import pytorch_lightning as pl


@dataclass
class LoaderConfig:
    batch_size: int = 100
    shuffle: bool = False
    pin_memory: bool = True
    drop_last: bool = True
    num_workers: int = 0
    persistent_workers: bool = False

    @property
    def config(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
        }

    @property
    def name(self) -> str:
        name_str = [
            f"bs={self.batch_size}",
            f"sh={self.shuffle}",
            f"pm={self.pin_memory}",
            f"dl={self.drop_last}",
            f"nw={self.num_workers}",
            f"pw={self.persistent_workers}",
        ]
        return "_".join(name for name in name_str if name is not None and name != "")


@dataclass
class TrainConfig:
    epochs: int = 50
    device_number: int = 0
    precision: str | int = 32
    matmul_precision: str = "high"
    optimizer_config: dict | str = "default"
    time_stamp: str = field(init=False)
    note: str = ""

    def __post_init__(self):
        self.time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    @property
    def device(self) -> torch.device:
        try:
            output = torch.device(f"cuda:{self.device_number}")
        except RuntimeError:
            raise ValueError("No GPU available")
        return output

    @property
    def name(self) -> str:
        name_str = [
            f"{self.time_stamp}",
            f"e={self.epochs}",
            f"p={self.precision}",
            f"n={self.note}" if self.note != "" else None,
        ]
        return "_".join(name for name in name_str if name is not None and name != "")
