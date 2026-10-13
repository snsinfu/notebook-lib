"""
Regression, ranking, and distribution-fitting helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.stats


def fit_power_law(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[float, float]:
    """
    Fit `y = beta * x**alpha` by ordinary least squares on log-log data.

    Non-positive pairs are ignored. Returns `(alpha, beta)`.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (x > 0) & (y > 0)
    if not np.any(mask):
        raise ValueError("fit_power_law needs at least one positive (x, y) pair")
    x = x[mask]
    y = y[mask]
    linreg: Any = scipy.stats.linregress(np.log(x), np.log(y))
    alpha = linreg.slope
    beta = np.exp(linreg.intercept)
    return float(alpha), float(beta)


class StatisticalLinearRegression:
    """
    Scalar-valued OLS with statistical analyses.

    Attributes:
    - `coef`: regression coefficients.
    - `coef_cov`: covariance matrix of the coefficients.
    - `coef_stderr`: standard error estimates for the coefficients.
    - `coef_dof`: degrees of freedom (`n_samples - n_features`).
    - `coef_pvalue`: two-sided t-test p-values for coefficients being zero.
    - `sse`: residual sum of squares.
    - `mse`: unbiased residual variance `sse / coef_dof`.
    - `R2`: coefficient of determination.

    No intercept column is added automatically; include one in `X` if wanted.
    """

    def __init__(self, X: npt.ArrayLike, y: npt.ArrayLike) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]

        n_samples, n_features = X.shape

        XX_inv = np.linalg.pinv(X.T @ X)
        coef = XX_inv @ (X.T @ y)

        resid = y - X @ coef
        sse = float(resid @ resid)
        sst = float(np.sum(np.square(y - y.mean())))
        if sst == 0:
            raise ValueError("y has zero variance; R2 is undefined")
        R2 = 1 - sse / sst

        coef_dof = n_samples - n_features
        # Unbiased residual variance; the coefficients' covariance and the
        # t-test both use this, so the two are consistent.
        mse = sse / coef_dof
        coef_cov = mse * XX_inv
        coef_stderr = np.sqrt(np.diag(coef_cov))
        coef_tvalue = coef / coef_stderr
        coef_pvalue = 2 * scipy.stats.t.sf(np.abs(coef_tvalue), coef_dof)

        self.coef: np.ndarray = coef
        self.coef_cov: np.ndarray = coef_cov
        self.coef_stderr: np.ndarray = coef_stderr
        self.coef_dof: int = coef_dof
        self.coef_pvalue: np.ndarray = coef_pvalue
        self.sse: float = sse
        self.mse: float = mse
        self.R2: float = R2

    def predict(self, X: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Return predicted responses and their standard errors for `X`.

        The standard error is that of the fitted mean, `sqrt(diag(X C X^T))`
        with `C = coef_cov`; it does not include residual noise.
        """
        X = np.asarray(X, dtype=float)
        y = X @ self.coef
        y_var = np.einsum("ij,jk,ik->i", X, self.coef_cov, X)
        y_stderr = np.sqrt(np.clip(y_var, 0, None))
        return y, y_stderr


def rank_values(xs: npt.ArrayLike, fraction: bool = False) -> np.ndarray:
    """
    Return the (ordinal) ranks of `xs`.

    Integral ranks in `[0, n)` are returned if `fraction` is false, and
    fractional ranks in `[0, 1)` if `fraction` is true. Ties are broken by
    position (no averaging).
    """
    xs = np.asarray(xs)
    n = len(xs)
    rank = np.empty(n, dtype=np.intp)
    rank[np.argsort(xs, kind="stable")] = np.arange(n)
    if fraction:
        rank = rank / n
    return rank
