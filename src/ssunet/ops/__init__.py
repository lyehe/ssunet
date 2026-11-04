"""Optional native operations; importing this package does not compile extensions."""

from .convnd import Conv4d, ConvNd, conv_nd, load_native_convnd

__all__ = ["Conv4d", "ConvNd", "conv_nd", "load_native_convnd"]
