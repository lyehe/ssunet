"""Consistent detector colors for confocal data views."""


def detector_color(name):
    """Use blue for HyD 2 and green for HyD 4, with gray for other detectors."""
    key = str(name).replace(" ", "").casefold()
    return {"hyd2": "blue", "hyd4": "green"}.get(key, "gray")


def detector_matplotlib_colormap(name):
    """Match napari's black-to-color detector ramps in exported scientific images."""
    from matplotlib.colors import LinearSegmentedColormap

    color = detector_color(name)
    endpoint = {"blue": "#0000ff", "green": "#00ff00", "gray": "#ffffff"}[color]
    return LinearSegmentedColormap.from_list(f"detector_{color}", ["#000000", endpoint])


def configure_detector_layout(viewer, channels, *, comparisons=1, overlay=True):
    """Overlay detectors additively while keeping different observations in separate panels.

    Layers must be ordered by comparison, then detector: observed HyD2/HyD4,
    followed by predicted HyD2/HyD4. Additive blending disables depth occlusion
    between the colored volumes, including their black backgrounds.
    """
    if channels < 1 or comparisons < 1 or len(viewer.layers) != channels * comparisons:
        message = "Expected one detector layer per channel in each comparison"
        raise ValueError(message)
    for layer in viewer.layers:
        layer.blending = "additive"
    viewer.grid.enabled = comparisons > 1 or not overlay
    viewer.grid.shape = (1, comparisons) if overlay else (comparisons, channels)
    viewer.grid.stride = channels if overlay else 1
