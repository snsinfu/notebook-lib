"""
Bandwidth selection for kernel density and regression estimates.

A bandwidth is either an absolute per-axis width in data units, or a
`BandwidthMethod` that estimates one from the samples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BandwidthMethod(ABC):
    """
    Estimate an absolute per-axis bandwidth from samples.

    `samples` has shape `(dim, n)` and the result has shape `(dim,)`.
    Relative, scipy-convention methods override `scipy_bw_method` so they can
    also be forwarded to `scipy.stats.gaussian_kde`.
    """

    @abstractmethod
    def estimate_bandwidth(
        self, samples: npt.ArrayLike, *, weights: npt.ArrayLike | None = None
    ) -> np.ndarray:
        """
        Return absolute per-axis bandwidths for `samples`.
        """

    def scipy_bw_method(self) -> str | float | None:
        """
        Relative bandwidth in `scipy.stats.gaussian_kde`'s convention.

        Raises `TypeError` for methods without a relative form.
        """
        raise TypeError("custom bandwidth methods cannot be used with kernel='scipy'")


class _FactorRule(BandwidthMethod):
    """
    Scott's or Silverman's rule of thumb.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def estimate_bandwidth(
        self, samples: npt.ArrayLike, *, weights: npt.ArrayLike | None = None
    ) -> np.ndarray:
        samples = np.asarray(samples, dtype=float)
        std, n = _std_and_size(samples, weights)
        if self.name == "scott":
            factor = scott_factor(n, samples.shape[0])
        else:
            factor = silverman_factor(n, samples.shape[0])
        return factor * std

    def scipy_bw_method(self) -> str:
        return self.name


class _RelativeStd(BandwidthMethod):
    """
    A fixed multiple of the per-axis sample standard deviation.
    """

    def __init__(self, factor: float) -> None:
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("relative bandwidth must be finite and positive")
        self.factor = float(factor)

    def estimate_bandwidth(
        self, samples: npt.ArrayLike, *, weights: npt.ArrayLike | None = None
    ) -> np.ndarray:
        samples = np.asarray(samples, dtype=float)
        std, _ = _std_and_size(samples, weights)
        return self.factor * std

    def scipy_bw_method(self) -> float:
        return self.factor


def bw_method(value: str | float) -> BandwidthMethod:
    """
    Build a bandwidth method from a name or a relative factor.

    Parameters:
    - `value`: `"scott"` or `"silverman"` selects a heuristic; a positive
      number is a factor relative to the per-axis sample standard deviation.
      Both follow `scipy.stats.gaussian_kde`'s `bw_method` convention.

    Pass the result as the `bandwidth` of `kde_at`, `kde_contour`, or
    `estimate_density`.
    """
    if isinstance(value, str):
        if value in {"scott", "silverman"}:
            return _FactorRule(value)
        raise ValueError(
            f"unknown bandwidth method {value!r}; expected 'scott' or 'silverman'"
        )
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("bw_method expects 'scott', 'silverman', or a positive number")
    return _RelativeStd(float(value))


def scott_factor(n: float, dim: int) -> float:
    """
    Scott's rule factor `n ** (-1 / (dim + 4))`.
    """
    return n ** (-1.0 / (dim + 4))


def silverman_factor(n: float, dim: int) -> float:
    """
    Silverman's rule factor `(n * (dim + 2) / 4) ** (-1 / (dim + 4))`.
    """
    return (n * (dim + 2) / 4) ** (-1.0 / (dim + 4))


def _std_and_size(
    samples: np.ndarray, weights: npt.ArrayLike | None
) -> tuple[np.ndarray, float]:
    """
    Return `(per-axis std, effective sample size)` for `samples` of shape
    `(dim, n)`.
    """
    if weights is None:
        center = samples.mean(axis=1, keepdims=True)
        std = np.sqrt(np.mean(np.square(samples - center), axis=1))
        return std, float(samples.shape[1])
    weight_values = np.asarray(weights, dtype=float).ravel()
    w = weight_values / weight_values.sum()
    center = np.sum(samples * w, axis=1, keepdims=True)
    var = np.sum(w * np.square(samples - center), axis=1)
    n_eff = float(weight_values.sum() ** 2 / np.square(weight_values).sum())
    return np.sqrt(var), n_eff


def resolve_bandwidth(
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None,
    samples: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> np.ndarray:
    """
    Return an absolute per-axis bandwidth of shape `(dim,)`.

    `samples` has shape `(dim, n)`. `bandwidth` may be `None` (Scott's rule),
    a positive absolute width (scalar or per-axis array), or a
    `BandwidthMethod`.
    """
    samples = np.asarray(samples, dtype=float)
    dim = samples.shape[0]

    if bandwidth is None:
        method: BandwidthMethod = _FactorRule("scott")
    elif isinstance(bandwidth, BandwidthMethod):
        method = bandwidth
    elif isinstance(bandwidth, str):
        raise TypeError(
            f"bandwidth={bandwidth!r} is not supported; wrap it in "
            f"bw_method({bandwidth!r})"
        )
    else:
        widths = np.broadcast_to(np.asarray(bandwidth, dtype=float), (dim,))
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
            raise ValueError("bandwidth must be finite and positive")
        return np.array(widths, dtype=float)

    weight_values = (
        None if weights is None else np.asarray(weights, dtype=float).ravel()
    )
    widths = np.asarray(
        method.estimate_bandwidth(samples, weights=weight_values), dtype=float
    ).ravel()
    if widths.shape != (dim,) or not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("bandwidth method produced invalid bandwidths")
    return widths


def scipy_bandwidth(
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None,
) -> str | float | None:
    """
    Map `bandwidth` to `scipy.stats.gaussian_kde`'s `bw_method`.

    `None` keeps scipy's default. A `BandwidthMethod` must define a
    scipy-convention relative value; absolute widths have no scipy equivalent.
    """
    if bandwidth is None:
        return None
    if isinstance(bandwidth, BandwidthMethod):
        return bandwidth.scipy_bw_method()
    if isinstance(bandwidth, str):
        raise TypeError(
            f"bandwidth={bandwidth!r} is not supported; use bw_method({bandwidth!r})"
        )
    raise ValueError(
        "kernel='scipy' needs a relative bandwidth; use bandwidth=None, "
        "bw_method('scott'), bw_method('silverman'), or bw_method(factor)"
    )
