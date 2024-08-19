import torch
import numpy as np

from ssunet.config import (
    PathConfig,
    SingleVolumeConfig,
    SplitParams,
    ModelConfig,
    LoaderConfig,
    TrainConfig,
    load_yaml,
    load_config,
)
from ssunet.dataloader import BinomDataset
import torch.utils.data as dt
from lightning.pytorch.loggers import TensorBoardLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)

from ssunet.models import SSUnet

if __name__ == "__main__":
    example_data = np.ones((512, 512, 512)).astype(np.float32)
    data_config = SingleVolumeConfig(example_data, 32, 32)
    split_params = SplitParams()
    model_config = ModelConfig()
    loader_config = LoaderConfig()
    train_config = TrainConfig()

    train_data = BinomDataset(data_config, split_params=split_params)

    train_loader = dt.DataLoader(train_data, **loader_config.to_dict)
    test_name = train_config.name
    logger = TensorBoardLogger(save_dir="./model_dir", name="test_name")

    trainer = pl.Trainer(
        default_root_dir="./model_dir",
        accelerator="cuda",
        gradient_clip_val=1,
        precision=train_config.precision,  # type: ignore
        devices=train_config.devices,
        max_epochs=train_config.max_epochs,
        callbacks=train_config.callbacks,
        logger=logger,  # type: ignore
        profiler="simple",
        limit_val_batches=20,
        log_every_n_steps=20,
        # enable_model_summary=True,
        # enable_checkpointing=True,
    )
    print(f"input_size: {tuple(next(iter(train_loader))[1].shape)}")

    model = SSUnet(model_config)

    trainer.fit(model, train_loader)
