"""
Numerical kernel density, regression, and flux estimation.

Plotting-oriented density helpers live in `lib.plot.density`.
"""

from __future__ import annotations

from .bandwidth import BandwidthMethod, bw_method, resolve_bandwidth
from .estimate import (
    DensityMeshAux,
    enclosed_thresholds,
    estimate_conditional_density_mesh,
    estimate_density_at,
    estimate_density_mesh,
)
from .kernels import (
    make_epanechnikov_kernel,
    make_gaussian_kernel,
    resolve_kernel,
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
    "BandwidthMethod",
    "DensityMeshAux",
    "bw_method",
    "enclosed_thresholds",
    "estimate_conditional_density_mesh",
    "estimate_density_at",
    "estimate_density_mesh",
    "kernel_flux",
    "kernel_regression",
    "kernel_regression_at",
    "make_epanechnikov_kernel",
    "make_gaussian_kernel",
    "mesh_kernel_flux",
    "mesh_kernel_regression",
    "resolve_bandwidth",
    "resolve_kernel",
    "scalar_epanechnikov_kernel",
    "scalar_gaussian_kernel",
]
