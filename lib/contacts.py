"""
Contact-matrix construction from point clouds.
"""

from __future__ import annotations

import numba
import numpy as np
import scipy.spatial


def compute_contact_matrix(
    points: np.ndarray,
    contact_distance: float,
    box_size: float | None = None,
) -> np.ndarray:
    """
    Count point pairs within `contact_distance` into a symmetric matrix.

    Entry `(i, j)` is the number of contacts of point `i` with point `j` (so
    the matrix is symmetric with a zero diagonal). The result is an integer
    matrix of shape `(n, n)`, which costs O(n**2) memory.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2-D array")
    if box_size is not None and box_size <= 0:
        raise ValueError("box_size must be positive")

    n = points.shape[0]
    matrix = np.zeros((n, n), dtype=np.intp)
    pairs = _compute_contact_pairs(points, contact_distance, box_size)
    _collect_pairs(matrix, pairs)
    return matrix


def _compute_contact_pairs(
    points: np.ndarray,
    contact_distance: float,
    box_size: float | None = None,
) -> np.ndarray:
    """
    Return index pairs closer than `contact_distance`.

    If `box_size` is given, positions are folded into a periodic box before the
    neighbor query.
    """
    if box_size is not None:
        points = np.mod(points, box_size)
    kdtree = scipy.spatial.KDTree(points, boxsize=box_size)
    return kdtree.query_pairs(contact_distance, output_type="ndarray")


@numba.njit
def _collect_pairs(matrix: np.ndarray, pairs: np.ndarray) -> None:
    """
    Accumulate symmetric pair counts into `matrix` in place.
    """
    for pair_index in range(pairs.shape[0]):
        i, j = pairs[pair_index]
        matrix[i, j] += 1
        matrix[j, i] += 1
