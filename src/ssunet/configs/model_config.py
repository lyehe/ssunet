"""Configuration for the project."""

from dataclasses import dataclass, field

from ..constants import DEFAULT_OPTIMIZER_CONFIG


@dataclass
class ModelConfig:
    """Configuration for the SSUnet model."""

    channels: int = 1
    depth: int = 4
    start_filts: int = 24
    depth_scale: int = 2
    depth_scale_stop: int = 10
    z_conv_stage: int = 5
    group_norm: int = 4
    skip_depth: int = 0
    dropout_p: float = 0.0
    scale_factor: float = 10.0
    sin_encoding: bool = True
    signal_levels: int = 10
    masked: bool = True
    partial_conv: bool = True
    down_checkpointing: bool = False
    up_checkpointing: bool = False
    loss_function: str = "photon"
    up_mode: str = "transpose"
    merge_mode: str = "concat"
    down_mode: str = "maxpool"
    activation: str = "relu"
    block_type: str = "tri"
    note: str = ""
    optimizer_config: dict = field(default_factory=lambda: DEFAULT_OPTIMIZER_CONFIG)

    @property
    def name(self) -> str:
        """Generate the name of the model."""
        name_str = [
            f"l={self.signal_levels}",
            f"d={self.depth}",
            f"sf={self.start_filts}",
            f"ds={self.depth_scale}at{self.depth_scale_stop}",
            f"f={self.scale_factor}",
            f"z={self.z_conv_stage}",
            f"g={self.group_norm}",
            f"sd={self.skip_depth}",
            f"b={self.block_type}",
            f"a={self.activation}",
        ]
        return "_".join(name_str)
