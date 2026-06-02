"""Tiny dependency-free colormaps (no matplotlib).

apply(gray, name) maps a 2-D float array in [0,1] to an (H, W, 3) uint8 RGB image by
linearly interpolating between a handful of anchor colors. Used to render the fractal
and error heatmaps in the evaluator preview and the dashboard.
"""

import numpy as np

# Anchor stops: (position in [0,1], (r, g, b) in [0,1]). Approximations of the
# matplotlib perceptual colormaps, good enough for visualization.
_MAPS = {
    "inferno": [
        (0.00, (0.001, 0.000, 0.014)), (0.15, (0.157, 0.043, 0.329)),
        (0.30, (0.396, 0.082, 0.431)), (0.45, (0.624, 0.165, 0.388)),
        (0.60, (0.831, 0.282, 0.259)), (0.75, (0.961, 0.490, 0.082)),
        (0.90, (0.980, 0.757, 0.153)), (1.00, (0.988, 1.000, 0.643)),
    ],
    "magma": [
        (0.00, (0.001, 0.000, 0.014)), (0.15, (0.146, 0.054, 0.361)),
        (0.30, (0.355, 0.094, 0.494)), (0.45, (0.569, 0.165, 0.502)),
        (0.60, (0.788, 0.243, 0.443)), (0.75, (0.946, 0.408, 0.349)),
        (0.90, (0.996, 0.667, 0.451)), (1.00, (0.987, 0.991, 0.749)),
    ],
    "turbo": [
        (0.000, (0.190, 0.072, 0.232)), (0.125, (0.275, 0.420, 0.890)),
        (0.250, (0.157, 0.690, 0.906)), (0.375, (0.165, 0.875, 0.659)),
        (0.500, (0.478, 0.949, 0.329)), (0.625, (0.882, 0.894, 0.094)),
        (0.750, (0.988, 0.655, 0.067)), (0.875, (0.882, 0.310, 0.031)),
        (1.000, (0.478, 0.016, 0.012)),
    ],
    "gist_heat": [  # black -> red -> orange -> white (matplotlib gist_heat)
        (0.000, (0.000, 0.000, 0.000)), (0.250, (0.375, 0.000, 0.000)),
        (0.500, (0.750, 0.000, 0.000)), (0.667, (1.000, 0.333, 0.000)),
        (0.750, (1.000, 0.500, 0.000)), (1.000, (1.000, 1.000, 1.000)),
    ],
}

# Defaults used across the project.
VALUE_CMAP = "inferno"     # for target / prediction images
ERROR_CMAP = "gist_heat"   # for the absolute-error heatmap (so it visually pops)


def apply(gray: np.ndarray, name: str = VALUE_CMAP) -> np.ndarray:
    """Map a [0,1] float array to an RGB uint8 image of shape (*gray.shape, 3)."""
    if name not in _MAPS:
        raise KeyError(f"unknown colormap {name!r}; have {list(_MAPS)}")
    g = np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    stops = _MAPS[name]
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops])  # (K, 3)
    rgb = np.stack([np.interp(g, pos, cols[:, c]) for c in range(3)], axis=-1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)
