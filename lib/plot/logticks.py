"""
Log-spaced ticks on a linear axis carrying `fwd(value)`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import matplotlib.axis
import matplotlib.ticker
import numpy as np
import numpy.typing as npt

Transform = Callable[[npt.ArrayLike], np.ndarray]


class _UnloggedAxis:
    """
    Proxy axis presenting *value-space* intervals to the stock tickers.

    `inv` is the inverse of the log applied to the data (`exp` for natural log,
    `10**v` for log10).
    """

    def __init__(self, axis: matplotlib.axis.Axis, inv: Transform) -> None:
        self._axis = axis
        self._inv = inv

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (get_tick_space, get_minpos, ...) to the
        # real axis.
        return getattr(self._axis, name)

    def get_view_interval(self) -> np.ndarray:
        """
        Return the view interval in the inverse-transformed space.
        """
        return self._inv(np.asarray(self._axis.get_view_interval(), dtype=float))

    def get_data_interval(self) -> np.ndarray:
        """
        Return the data interval in the inverse-transformed space.
        """
        return self._inv(np.asarray(self._axis.get_data_interval(), dtype=float))


class LogTicksLocator(matplotlib.ticker.Locator):
    """
    Log-spaced tick positions for a linear axis carrying `fwd(value)`.
    """

    def __init__(
        self,
        fwd: Transform,
        inv: Transform,
        base: float = 10.0,
        subs: Sequence[float] | Literal["auto", "all"] = (1.0,),
        numticks: int | None = None,
    ) -> None:
        self._fwd = fwd
        self._inv = inv
        self._inner = matplotlib.ticker.LogLocator(
            base=base, subs=subs, numticks=numticks
        )

    def set_axis(self, axis: Any) -> None:
        super().set_axis(axis)
        self._inner.set_axis(cast(matplotlib.axis.Axis, _UnloggedAxis(axis, self._inv)))

    def set_params(self, **kwargs: Any) -> None:
        self._inner.set_params(**kwargs)

    def __call__(self) -> Sequence[float]:
        assert self.axis is not None
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(float(vmin), float(vmax))

    def tick_values(self, vmin: float, vmax: float) -> Sequence[float]:
        inv_vmin = float(self._inv(vmin))
        inv_vmax = float(self._inv(vmax))
        return list(self._fwd(self._inner.tick_values(inv_vmin, inv_vmax)))


class LogTicksFormatter(matplotlib.ticker.Formatter):
    """
    Power-of-base labels for positions that are `fwd(value)`.
    """

    def __init__(
        self,
        fwd: Transform,
        inv: Transform,
        base: float = 10.0,
        labelOnlyBase: bool = False,
        minor_thresholds: Any = None,
    ) -> None:
        self._fwd = fwd
        self._inv = inv
        self._inner = matplotlib.ticker.LogFormatterSciNotation(
            base=base,
            labelOnlyBase=labelOnlyBase,
            minor_thresholds=minor_thresholds,
        )

    def set_axis(self, axis: Any) -> None:
        super().set_axis(axis)
        self._inner.set_axis(cast(matplotlib.axis.Axis, _UnloggedAxis(axis, self._inv)))

    def set_locs(self, locs: Any = None) -> None:
        super().set_locs(locs)
        self._inner.set_locs(
            None if locs is None else self._inv(np.asarray(locs, dtype=float))
        )

    def __call__(self, x: float, pos: int | None = None) -> str:
        return self._inner(float(self._inv(x)), pos)


def _log_transforms(base: float) -> tuple[Transform, Transform]:
    """
    Return the forward/inverse transforms for a base-`base` log axis.
    """
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and not equal to 1")
    log_base = float(np.log(base))

    def log(v: npt.ArrayLike) -> np.ndarray:
        return np.log(np.asarray(v, dtype=float)) / log_base

    def exp(v: npt.ArrayLike) -> np.ndarray:
        return np.power(base, np.asarray(v, dtype=float))

    return log, exp


def use_log_ticks(axis: matplotlib.axis.Axis, base: float = 10.0) -> None:
    """
    Give a linear `axis` native log ticks and labels.

    The axis carries `log_base(value)`, so this is a linear axis whose ticks
    sit at `base**n`; both the tick positions and labels are derived from
    `base` alone.

    Parameters:
    - `axis`: the axis to configure (usually `ax.xaxis` or `ax.yaxis`).
    - `base`: base of the log axis, i.e. major ticks sit at `base**n`
      (default 10; use `numpy.e` for a natural-log axis).
    """
    log, exp = _log_transforms(base)
    axis.set_major_locator(LogTicksLocator(log, exp, base=base))
    axis.set_major_formatter(LogTicksFormatter(log, exp, base=base))
    axis.set_minor_locator(LogTicksLocator(log, exp, base=base, subs="auto"))
    axis.set_minor_formatter(LogTicksFormatter(log, exp, base=base))
