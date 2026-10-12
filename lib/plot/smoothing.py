"""
Array smoothing helpers for heat-map display.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage


def smooth_matrix(matrix: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian-smooth `matrix` with reflect boundary handling.
    """
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    radius = int(3 * sigma) + 1
    return scipy.ndimage.gaussian_filter(
        matrix,
        sigma=sigma,
        radius=radius,
        mode="reflect",
    )
