"""GAP dataset that splits the data into target and noise components."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from numpy.random import choice, rand, seed
from torch.distributions.binomial import Binomial

from ssunet.constants import EPSILON, LOGGER

from .singlevolume import SingleVolumeDataset


class InvalidPValueError(ValueError):
    """Exception raised when p value is invalid."""

    def __init__(self, message: str):
        super().__init__(message)
        LOGGER.error(f"InvalidPValueError: {message}")


class MissingPListError(ValueError):
    """Exception raised when p_list is missing for list method."""

    def __init__(self):
        super().__init__("p_list must be provided when method is list")
        LOGGER.error("MissingPListError: p_list must be provided when method is list")


@dataclass
class SplitParams:
    """Configuration for splitting the data into target and noise components."""

    method: str = "signal"
    min_p: float = EPSILON
    max_p: float = 1 - EPSILON
    p_list: list[float] | None = field(default_factory=list)
    normalize_target: bool = True
    seed: int | None = None

    def __post_init__(self):
        """Validate and initialize the split parameters."""
        if self.min_p > self.max_p:
            LOGGER.warning("min_p should be less than max_p, swapping values")
            self.min_p, self.max_p = self.max_p, self.min_p
        if self.seed is not None:
            seed(self.seed)


class BinomDataset(SingleVolumeDataset):
    """Dataset that splits the data into target and noise using binomial sampling."""

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Get the target and noise components for the input image."""
        index = self._index(index)
        input = self.data[index : index + self.z_size]
        # Combine the input and ground truth data for cropping
        if self.reference is None:
            output = self._crop_list(self._rotate_list([input]))
            output = self._split(output[0])
        else:
            reference = self.reference[index : index + self.z_size]
            output = self._crop_list(self._rotate_list([input, reference]))
            image, noise = self._split(output[0])
            output = [image, noise, output[1]]

        return self._add_channel_dim(self._augment_list(output))

    def __post_init__(self):
        """Validate and initialize the dataset."""
        if self.split_params.seed is not None:
            seed(self.split_params.seed)

    @property
    def split_params(self) -> SplitParams:
        """Get the split parameters."""
        return self.kwargs.get("split_params", SplitParams())

    @property
    def data_size(self) -> int:
        """Get the data size."""
        return self.data.shape[0] - self.z_size + 1

    def _split(self, input: torch.Tensor) -> list[torch.Tensor]:
        """Split the input image into the image and noise components using the distribution."""
        p_value = self._sample_p(input)
        noise = self._sample_noise(input, p_value)
        target = (input - noise).float()
        if self.split_params.normalize_target:
            target = self.normalize_by_mean(target)
        return [target, noise.float()]

    def _sample_p(self, input: torch.Tensor) -> float:
        """Random sample the p level for the input image."""
        # User can pass a custom function to sample the p level
        p_sampling_method: Callable | None = self.kwargs.get("p_sampling_method", None)
        if p_sampling_method is not None and callable(p_sampling_method):
            return p_sampling_method(input, **self.kwargs)
        elif self.split_params.method == "db":
            return self._sample_db(input)
        elif self.split_params.method == "signal":
            return self._sample_signal()
        elif self.split_params.method == "fixed":
            return self._sample_fixed()
        elif self.split_params.method == "list":
            return self._sample_list()
        else:
            LOGGER.warning(f"Method {self.split_params.method} not supported, using default")
            return self._sample_signal()

    @staticmethod
    def _validate_p(p_value: float) -> float:
        """Check the p level is within the valid range."""
        if p_value <= 0:
            raise InvalidPValueError("p<=0")
        if p_value >= 1:
            raise InvalidPValueError("p>=1")
        return p_value

    def _sample_db(self, input: torch.Tensor) -> float:
        """Random sample the PSNR level for the input image."""
        min_psnr = min(self.split_params.min_p, self.split_params.max_p)
        max_psnr = max(self.split_params.min_p, self.split_params.max_p)
        uniform = rand() * (max_psnr - min_psnr) + min_psnr
        p_value = (10 ** (uniform / 10.0)) / (input.float().mean().item() + EPSILON)
        return self._validate_p(p_value)

    def _sample_signal(self) -> float:
        """Random sample the signal level for the input image."""
        min_p = min(self.split_params.min_p, self.split_params.max_p)
        max_p = max(self.split_params.min_p, self.split_params.max_p)
        offset = min_p
        scale = max_p - min_p
        p_value = rand() * scale + offset
        return self._validate_p(p_value)

    def _sample_fixed(self) -> float:
        """Random sample the fixed level for the input image."""
        p_value = (
            self.split_params.p_list[0] if self.split_params.p_list else self.split_params.min_p
        )
        return self._validate_p(p_value)

    def _sample_list(self) -> float:
        """Random sample the choice level for the input image."""
        p_list = self.split_params.p_list
        if p_list is None or len(p_list) == 0:
            raise MissingPListError()
        p_value = choice(p_list)
        return self._validate_p(p_value)

    @staticmethod
    def _sample_noise(input: torch.Tensor, p_value: float) -> torch.Tensor:
        """Sample the noise data for the input image using a binomial distribution."""
        input = torch.floor_(input.float())
        binom = Binomial(total_count=input, probs=torch.tensor([p_value]))  # type: ignore
        return binom.sample()


class BernoulliDataset(BinomDataset):
    """Special case of the BinomDataset where the noise is sampled with a Bernoulli distribution."""

    @staticmethod
    def _sample_noise(input: torch.Tensor, p_value: float) -> torch.Tensor:
        return torch.bernoulli(input * p_value)