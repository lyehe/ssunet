from .modules import (
    DownConvDual3D,
    UpConvDual3D,
    DownConvTri3D,
    UpConvTri3D,
    LKDownConv3D,
)
from .modulets import conv111

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
    "BLOCK",
]
