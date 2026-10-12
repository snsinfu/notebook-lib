import numpy as np
import pytest

from lib.stats import StatisticalLinearRegression, fit_power_law, rank_values


def test_fit_power_law_recovers_parameters():
    x = np.linspace(0.1, 10, 200)
    alpha, beta = fit_power_law(x, 2.0 * x**-1.5)
    assert np.isclose(alpha, -1.5, atol=1e-6)
    assert np.isclose(beta, 2.0, atol=1e-6)


def test_fit_power_law_empty_raises():
    with pytest.raises(ValueError):
        fit_power_law([-1.0, -2.0], [1.0, 2.0])


def test_linear_regression_recovers_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 1))
    X = np.hstack([np.ones((300, 1)), x])
    y = 1.0 + 2.0 * x[:, 0] + rng.normal(0, 0.01, 300)
    reg = StatisticalLinearRegression(X, y)
    assert np.allclose(reg.coef, [1.0, 2.0], atol=0.05)
    assert reg.R2 > 0.99


def test_rank_values_monotone_and_fractional():
    xs = np.array([3.0, 1.0, 2.0])
    integral = rank_values(xs)
    assert list(integral[np.argsort(xs)]) == [0, 1, 2]
    fractional = rank_values(xs, fraction=True)
    assert np.allclose(fractional, integral / 3)


def test_rank_values_breaks_ties_by_position():
    # Regression: np.argsort is not stable, so position tie-breaking (as
    # documented) was not guaranteed for large inputs.
    xs = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    assert list(rank_values(xs)) == [2, 3, 0, 4, 1]


def test_linear_regression_stderr_uses_unbiased_variance():
    # Regression: coef_cov used the ddof=0 MSE instead of SSR/(n - p), so the
    # standard errors were too small for finite samples.
    rng = np.random.default_rng(3)
    n = 8
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y = 1.0 + 2.0 * x + rng.normal(0, 1.0, n)
    reg = StatisticalLinearRegression(X, y)

    resid = y - X @ reg.coef
    s2 = np.sum(resid**2) / (n - X.shape[1])
    expected = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    assert np.allclose(reg.coef_stderr, expected, rtol=1e-9)


def test_linear_regression_pvalue_stays_positive_for_large_t():
    # Regression: `1 - t.cdf` underflows to 0 for large |t|; `t.sf` does not.
    n = 12
    x = np.linspace(0, 1, n)
    X = np.column_stack([np.ones(n), x])
    y = 1.0 + 5.0 * x
    y[-1] += 1e-9
    reg = StatisticalLinearRegression(X, y)
    assert np.all(reg.coef_pvalue > 0)
