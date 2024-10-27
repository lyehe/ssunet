"""Module init."""

from .modules import (
    DownConvDual3D,
    DownConvTri3D,
    LKDownConv3D,
    PartialDownConv3D,
    PartialUpConv3D,
    UpConvDual3D,
    UpConvTri3D,
)
from .modulets import conv111, partial333

BLOCK = {
    "dual": (DownConvDual3D, UpConvDual3D),
    "tri": (DownConvTri3D, UpConvTri3D),
    "LK": (LKDownConv3D, UpConvTri3D),
    "partial": (PartialDownConv3D, PartialUpConv3D),
}

__all__ = [
    "DownConvDual3D",
    "UpConvDual3D",
    "DownConvTri3D",
    "UpConvTri3D",
    "LKDownConv3D",
    "PartialDownConv3D",
    "PartialUpConv3D",
    "conv111",
    "partial333",
    "BLOCK",
]
