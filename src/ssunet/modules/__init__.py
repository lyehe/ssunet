"""Module init."""

from .base_modules import conv111, partial333
from .module_blocks import (
    DownConvDual3D,
    DownConvTri3D,
    LKDownConv3D,
    UpConvDual3D,
    UpConvTri3D,
)

BLOCK = {
    "dual": (DownConvDual3D, UpConvDual3D),
    "tri": (DownConvTri3D, UpConvTri3D),
    "LK": (LKDownConv3D, UpConvTri3D),
}

__all__ = [
    "DownConvDual3D",
    "UpConvDual3D",
    "DownConvTri3D",
    "UpConvTri3D",
    "LKDownConv3D",
    "conv111",
    "partial333",
    "BLOCK",
]
