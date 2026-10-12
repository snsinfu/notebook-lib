"""
Numerical kernel density, regression, and flux estimation.

Plotting-oriented density helpers live in `lib.plot.density`.
"""

from __future__ import annotations

from .estimate import enclosed_thresholds, estimate_density
from .kernels import (
    make_epanechnikov_kernel,
    make_gaussian_kernel,
    scalar_epanechnikov_kernel,
    scalar_gaussian_kernel,
)
from .regression import (
    kernel_flux,
    kernel_regression,
    kernel_regression_at,
    mesh_kernel_flux,
    mesh_kernel_regression,
)

__all__ = [
    "enclosed_thresholds",
    "estimate_density",
    "kernel_flux",
    "kernel_regression",
    "kernel_regression_at",
    "make_epanechnikov_kernel",
    "make_gaussian_kernel",
    "mesh_kernel_flux",
    "mesh_kernel_regression",
    "scalar_epanechnikov_kernel",
    "scalar_gaussian_kernel",
]
