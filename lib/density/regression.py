"""
Nadaraya-Watson kernel regression and flux estimation.
"""

from __future__ import annotations

import numba
import numpy as np

from .kernels import Kernel


@numba.njit(error_model="numpy")
def kernel_regression_at(
    point: np.ndarray,
    sample_points: np.ndarray,
    sample_values: np.ndarray,
    *,
    kernel_fn: Kernel,
    scale: float | np.ndarray,
    average: bool,
) -> float:
    """
    Evaluate a kernel-weighted average (or raw sum) at a single point.
    """
    weights = kernel_fn((sample_points - point) * scale)
    w_sum = weights.sum(0)
    v_sum = (sample_values * weights).sum(0)
    result = v_sum / w_sum if average else v_sum
    return float(result)


def kernel_regression(
    points: np.ndarray,
    sample_points: np.ndarray,
    sample_values: np.ndarray,
    *,
    kernel_fn: Kernel,
    scale: float | np.ndarray = 1,
    average: bool = True,
) -> np.ndarray:
    """
    Evaluate `kernel_regression_at` for each row of `points`.
    """
    n = points.shape[0]
    values = np.empty(n)
    for i in range(n):
        values[i] = kernel_regression_at(
            points[i],
            sample_points,
            sample_values,
            kernel_fn=kernel_fn,
            scale=scale,
            average=average,
        )
    return values


def _mesh_points(
    mesh_spans: tuple[tuple[float, float], ...],
    n_div: int,
    kernel_coverage: float,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Build a regular grid over `mesh_spans`.

    Returns `(grid_coords, points, scales)`: the `meshgrid` coordinates, the
    flattened `(n_div**d, d)` points, and the per-axis scales that normalize
    distances to ~1 (used as the kernel bandwidth).
    """
    if kernel_coverage <= 0:
        raise ValueError("kernel_coverage must be positive")
    if n_div < 2:
        raise ValueError("n_div must be at least 2")
    scales = np.array(
        [1 / (kernel_coverage * (span[1] - span[0])) for span in mesh_spans]
    )
    coord_points = [np.linspace(*span, num=n_div) for span in mesh_spans]
    grid_coords = np.meshgrid(*coord_points, indexing="xy")
    points = np.reshape(grid_coords, (len(grid_coords), -1)).T
    return grid_coords, points, scales


def mesh_kernel_regression(
    mesh_spans: tuple[tuple[float, float], ...],
    sample_points: np.ndarray,
    sample_values: np.ndarray,
    *,
    kernel_fn: Kernel,
    kernel_coverage: float = 0.05,
    n_div: int = 21,
    average: bool = True,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """
    Kernel regression evaluated on a regular grid over `mesh_spans`.
    """
    grid_coords, points, scales = _mesh_points(mesh_spans, n_div, kernel_coverage)

    values = kernel_regression(
        points,
        sample_points,
        sample_values,
        kernel_fn=kernel_fn,
        scale=scales,
        average=average,
    )

    grid_values = values.reshape((n_div,) * len(mesh_spans))

    return grid_coords, grid_values


def kernel_flux(
    points: np.ndarray,
    sequence: np.ndarray,
    *,
    kernel_fn: Kernel,
    scale: float | np.ndarray = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate flux and density from a sequence of points.

    Parameters:
    - `points` (N, d): points at which to estimate flux.
    - `sequence` (T, d): sample points describing a trajectory.
    - `kernel_fn`: d-dimensional kernel function to use.
    - `scale`: per-axis inverse bandwidth that normalizes distances to ~1.

    Returns `(flux, density)` where `flux` is `(N, d)` and `density` is `(N,)`.
    """
    points = np.asarray(points, dtype=float)
    sequence = np.asarray(sequence, dtype=float)
    if sequence.ndim != 2 or sequence.shape[0] < 2:
        raise ValueError("sequence must be a (T, d) array with T >= 2")

    prev_deltas = sequence[None, :-1] - points[:, None]
    next_deltas = sequence[None, 1:] - points[:, None]
    weights = kernel_fn(prev_deltas * scale) * np.prod(scale)
    flux = (weights[:, :, None] * next_deltas).mean(1)
    density = weights.mean(1)
    return flux, density


def mesh_kernel_flux(
    mesh_spans: tuple[tuple[float, float], ...],
    sequence: np.ndarray,
    *,
    kernel_fn: Kernel,
    kernel_coverage: float = 0.05,
    n_div: int = 21,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Flux and density estimated on a regular grid over `mesh_spans`.
    """
    grid_coords, points, scales = _mesh_points(mesh_spans, n_div, kernel_coverage)

    flux, density = kernel_flux(points, sequence, kernel_fn=kernel_fn, scale=scales)

    shape = (n_div,) * len(mesh_spans)
    grid_flux = flux.reshape(shape + (points.shape[1],))
    grid_density = density.reshape(shape)

    return grid_coords, grid_flux, grid_density
