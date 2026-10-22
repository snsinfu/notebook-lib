"""
Matrix-shaped data helpers for heat-map display.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
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


def transform_triu_coords(
    i: npt.ArrayLike, j: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform matrix `(i, j)` coordinates to rotated upper-triangle `(x, y)`.

    The transform is `x = (i + j) / 2`, `y = j - i`. It is a 45-degree rotation
    that preserves cell area and maps the upper triangle (`j >= i`) to `x >= 0`,
    `y >= 0`. Inputs are continuous grid coordinates (edge or cell-center
    positions, not necessarily integer); for the center of a cell in a mesh
    built from integer edges, pass `i + 0.5, j + 0.5`.
    """
    i_arr = np.asarray(i)
    j_arr = np.asarray(j)
    return (i_arr + j_arr) / 2, j_arr - i_arr


def make_triu_mesh(
    matrix: npt.ArrayLike,
    edges: npt.ArrayLike | None = None,
    *,
    mask_diagonals: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """
    Build a pcolormesh-ready rotated mesh of the upper triangle of `matrix`.

    Parameters:
    - `matrix`: square 2-D array of cell values.
    - `edges`: optional `n + 1` finite bin edges in matrix index units;
      defaults to `numpy.arange(n + 1)`, i.e. unit cells.
    - `mask_diagonals`: number of diagonals to mask downward from the main
      diagonal. `0` keeps the diagonal (the strict lower triangle is always
      masked), `1` also masks the main diagonal, `2` also masks the first
      subdiagonal, and so on.

    Returns `(X, Y, C)` for `ax.pcolormesh(X, Y, C, shading="flat")`. `C` is
    `matrix` with the lower triangle masked; `X` and `Y` are `(n + 1, n + 1)`
    corner arrays. Use `transform_triu_coords` to place annotations.
    """
    matrix_array = np.asanyarray(matrix)
    if matrix_array.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError("matrix must be square")
    n = matrix_array.shape[0]

    if edges is None:
        edges_array = np.arange(n + 1, dtype=float)
    else:
        edges_array = np.asarray(edges, dtype=float)
        if edges_array.ndim != 1:
            raise ValueError("edges must be 1-D")
        if edges_array.size != n + 1:
            raise ValueError("edges must have n + 1 entries for an n x n matrix")
        if not np.all(np.isfinite(edges_array)):
            raise ValueError("edges must be finite")

    if isinstance(mask_diagonals, bool) or not isinstance(
        mask_diagonals, (int, np.integer)
    ):
        raise TypeError("mask_diagonals must be an integer")
    if mask_diagonals < 0:
        raise ValueError("mask_diagonals must be nonnegative")

    indices = np.arange(n + 1)
    mesh_i, mesh_j = np.meshgrid(indices, indices, indexing="ij")
    X, Y = transform_triu_coords(edges_array[mesh_i], edges_array[mesh_j])

    mask = np.tril(np.ones((n, n), dtype=bool), k=int(mask_diagonals) - 1)
    C = np.ma.masked_where(mask, matrix_array)

    return X, Y, C
