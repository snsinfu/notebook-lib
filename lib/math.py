"""
Linear algebra: masked PCA and matrix balancing.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def bnewt(
    A: np.ndarray,
    tol: float = 1e-6,
    x0: np.ndarray | None = None,
    delta_close: float = 0.1,
    delta_far: float = 3,
    max_iter: int = 1000,
) -> np.ndarray:
    """
    Balance a nonnegative matrix with rows/columns summing to one.

    Ported from the MATLAB code [Knight and Ruiz, 2012]. Returns the scaling
    vector `x` such that `x * A @ (x)` has unit row (and, for symmetric `A`,
    column) sums.

    Parameters:
    - `A`: nonnegative matrix to balance.
    - `tol`: convergence tolerance.
    - `x0`: initial scaling vector; defaults to ones. It is copied, not
      modified.
    - `delta_close`, `delta_far`: boundary controls for the inner cone test.
    - `max_iter`: maximum number of outer iterations.

    Raises `ValueError` if `A` is not a finite, nonnegative square matrix
    without an all-zero row or column, and `RuntimeError` if the outer
    iteration does not converge within `max_iter` iterations.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.all(np.isfinite(A)) or np.any(A < 0):
        raise ValueError("A must be finite and nonnegative")
    if np.any(A.sum(axis=1) == 0) or np.any(A.sum(axis=0) == 0):
        raise ValueError("A must not contain an all-zero row or column")

    # Parameters used in inner stopping criterion.
    g = 0.9
    etamax = 0.1
    eta = etamax
    stop_tol = tol * 0.5

    n = A.shape[0]
    x = np.ones(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    rt = tol**2
    v = x * A.dot(x)
    rk = 1 - v
    rho_km1 = np.vdot(rk, rk)
    rho_km2 = 0.0
    rout = rho_km1
    rold = rout

    # Outer iteration count.
    i = 0

    while rout > rt:
        i += 1
        if i > max_iter:
            raise RuntimeError(
                f"bnewt did not converge within {max_iter} outer iterations"
            )
        k = 0
        y = np.ones(n)
        innertol = max(eta**2 * rout, rt)
        Z = None
        p = None

        while rho_km1 > innertol:
            k += 1
            if k == 1:
                Z = rk / v
                p = Z
                rho_km1 = np.vdot(rk, Z)
            else:
                beta = rho_km1 / rho_km2
                p = Z + beta * p

            # Update search direction efficiently.
            w = x * A.dot(x * p) + v * p
            alpha = rho_km1 / np.vdot(p, w)
            ap = alpha * p

            # Test distance to boundary of cone.
            ynew = y + ap

            if ynew.min() <= delta_close:
                if delta_close == 0:
                    break
                ind = ap < 0
                gamma = ((delta_close - y[ind]) / ap[ind]).min()
                y += gamma * ap
                break

            if ynew.max() >= delta_far:
                ind = ynew > delta_far
                gamma = ((delta_far - y[ind]) / ap[ind]).min()
                y += gamma * ap
                break

            y = ynew
            rk -= alpha * w
            rho_km2 = rho_km1
            Z = rk / v
            rho_km1 = np.vdot(rk, Z)

        x *= y
        v = x * A.dot(x)
        rk = 1 - v
        rho_km1 = np.vdot(rk, rk)
        rout = rho_km1

        # Update inner iteration stopping criterion.
        rat = rout / rold
        rold = rout
        res_norm = np.sqrt(rout)
        eta_o = eta
        eta = g * rat

        if g * eta_o**2 > 0.1:
            eta = max(eta, g * eta_o**2)

        eta = max(min(eta, etamax), stop_tol / res_norm)

    return x


def masked_pca(
    data: npt.ArrayLike,
    row_mask: npt.ArrayLike | None = None,
    col_mask: npt.ArrayLike | None = None,
    k: int = 6,
    correlation: bool = False,
    nan_tolerance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA of a data matrix, tolerating NaN/invalid rows and columns.

    Rows are observations and columns are variables. The covariance (or
    correlation) matrix of the retained variables is diagonalized. Returns the
    full covariance matrix and the principal vectors scaled by explained
    variance ratios, both padded back to the original column count with NaNs.

    Parameters:
    - `data`: data matrix, rows are observations.
    - `row_mask`, `col_mask`: additional rows/columns to drop (`True` means
      masked out).
    - `k`: number of principal components to compute (clamped to the number of
      retained columns).
    - `correlation`: diagonalize the correlation instead of the covariance
      matrix. Constant (zero-variance) columns are dropped.
    - `nan_tolerance`: maximum fraction of NaNs allowed per column before it is
      dropped.

    Returns:
    - `ext_C`: covariance/correlation matrix over the original columns (NaN
      where a column was dropped).
    - `ext_vecs`: principal vectors with variance information embedded, shape
      `(n_columns, k_eff)`. Recovering explained variance ratios with
      `nanvar(ext_vecs, axis=0)` needs at least two retained variables.
    """
    # Below we filter out invalid rows and columns.
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be 2-D (rows are observations, columns variables)")
    valid_data = data
    col_mask_arr = None if col_mask is None else np.asarray(col_mask, dtype=bool)
    row_mask_arr = None if row_mask is None else np.asarray(row_mask, dtype=bool)

    # Mask out bad features (columns). We require enough valid observations for
    # each feature so that covariance estimation is robust.
    bad_cols = np.mean(np.isnan(valid_data), axis=0) > nan_tolerance
    if correlation:
        # A constant feature has zero variance, so it cannot be correlated.
        bad_cols = bad_cols | (np.nanvar(valid_data, axis=0) == 0)
    if col_mask_arr is None:
        valid_cols = ~bad_cols
    else:
        valid_cols = ~(bad_cols | col_mask_arr)
    valid_data = valid_data[:, valid_cols]

    if valid_data.shape[1] == 0:
        raise ValueError("no valid columns available for PCA")

    # Mask out bad samples (rows). Any NaN in the data matrix will produce NaN
    # covariance, so we mask observations having one or more NaNs. Note that
    # the order of column/row masking matters: features are already removed, so
    # fewer observations are expected to be bad here.
    bad_rows = np.any(np.isnan(valid_data), axis=1)
    if row_mask_arr is None:
        valid_rows = ~bad_rows
    else:
        valid_rows = ~(bad_rows | row_mask_arr)
    valid_data = valid_data[valid_rows, :]

    if valid_data.shape[0] == 0:
        raise ValueError("no valid rows available for PCA")

    # Covariance (correlation) between valid features (columns are random
    # variables, rows are observations). The matrix is dense, so diagonalize it
    # directly; `eigsh` is avoided because its eigenvalue order is unspecified
    # and it rejects `k >= n`.
    C = np.atleast_2d(np.cov(valid_data, ddof=0, rowvar=False))

    if correlation:
        normalizer = 1 / np.sqrt(np.diag(C))
        C *= normalizer[:, None]
        C *= normalizer[None, :]

    trace = np.trace(C)
    if trace == 0:
        raise ValueError("data matrix has zero variance; PCA is undefined")

    k_eff = min(k, C.shape[0])

    # Eigenvalues of a symmetric matrix, ascending; keep the largest k.
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1][:k_eff]
    vals = evals[order]
    vecs = evecs[:, order]

    # Embed variance information in the principal vectors, so that
    # `nanvar(vecs, axis=0)` gives the explained variance ratios.
    std = np.std(vecs, axis=0)
    std[std == 0] = 1.0
    vecs = vecs / std[None, :]
    vecs = vecs * np.sqrt(vals)[None, :]
    vecs = vecs / np.sqrt(trace)

    # Extend back to the original shape, filling missing entries with NaNs.
    valid_indices = np.arange(data.shape[1])[valid_cols]

    ext_vecs = np.empty((data.shape[1], k_eff))
    ext_vecs[:, :] = np.nan

    ext_C = np.empty((data.shape[1], data.shape[1]), dtype=C.dtype)
    ext_C[:, :] = np.nan

    for i in range(valid_data.shape[1]):
        ext_vecs[valid_indices[i]] = vecs[i]
        ext_C[valid_indices[i], valid_cols] = C[i, :]

    return ext_C, ext_vecs
