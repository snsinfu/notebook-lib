"""
Kernel functions for nonparametric density and regression estimates.

All kernels integrate to one. The `make_*` factories return numba-compiled
callables; for an input of shape `(..., d)` they reduce the last axis and
return shape `(...)`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numba
import numpy as np
import scipy.special

Kernel = Callable[[np.ndarray], np.ndarray]


@numba.njit
def scalar_gaussian_kernel(x: float | np.ndarray) -> float | np.ndarray:
    """
    Standard 1-D Gaussian kernel `exp(-x**2 / 2) / sqrt(2*pi)`.
    """
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2 * np.pi)


@numba.njit
def scalar_epanechnikov_kernel(x: float | np.ndarray) -> float | np.ndarray:
    """
    Standard 1-D Epanechnikov kernel `0.75 * max(0, 1 - x**2)`.
    """
    return 0.75 * np.maximum(0, 1 - np.square(x))


@functools.cache
def make_gaussian_kernel(dim: int) -> Kernel:
    """
    Return the `dim`-dimensional Gaussian kernel.

    `(2*pi)**(-dim/2) * exp(-|x|**2 / 2)`.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1")
    norm = (2 * np.pi) ** (-dim / 2)

    def fn(x: np.ndarray) -> np.ndarray:
        return norm * np.exp(-0.5 * np.square(x).sum(-1))

    return numba.njit(fn)


@functools.cache
def make_epanechnikov_kernel(dim: int) -> Kernel:
    """
    Return the `dim`-dimensional Epanechnikov kernel.

    `gamma(dim/2 + 2) / pi**(dim/2) * max(0, 1 - |x|**2)`.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1")
    norm = scipy.special.gamma(dim / 2 + 2) * np.pi ** (-dim / 2)

    def fn(x: np.ndarray) -> np.ndarray:
        return norm * np.maximum(0, 1 - np.square(x).sum(-1))

    return numba.njit(fn)
