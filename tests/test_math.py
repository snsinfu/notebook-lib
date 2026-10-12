import numpy as np
import pytest

from lib.math import bnewt, masked_pca


def test_bnewt_balances_small_symmetric_matrix():
    # The symmetric scaling x_i A_ij x_j reaches unit row/column sums for
    # symmetric A.
    rng = np.random.default_rng(0)
    A = rng.random((6, 6))
    A = A + A.T + 0.1
    x = bnewt(A)
    B = x[:, None] * A * x[None, :]
    assert np.allclose(B.sum(1), 1, atol=1e-4)
    assert np.allclose(B.sum(0), 1, atol=1e-4)


def test_bnewt_normalizes_rows_of_general_matrix():
    rng = np.random.default_rng(0)
    A = rng.random((6, 6)) + 0.1
    x = bnewt(A)
    B = x[:, None] * A * x[None, :]
    assert np.allclose(B.sum(1), 1, atol=1e-4)


def test_bnewt_accepts_array_x0():
    # Regression: `x0 if x0 else ...` raised on numpy arrays.
    A = np.ones((3, 3)) + np.eye(3)
    x = bnewt(A, x0=np.ones(3))
    assert x.shape == (3,)
    assert np.all(x > 0)


def test_bnewt_does_not_mutate_x0():
    A = np.ones((3, 3)) + np.eye(3)
    x0 = np.ones(3)
    bnewt(A, x0=x0)
    assert np.array_equal(x0, np.ones(3))


def test_masked_pca_orders_components_by_variance():
    rng = np.random.default_rng(0)
    data = np.column_stack(
        [
            rng.normal(0, 3.0, 300),
            rng.normal(0, 1.0, 300),
            rng.normal(0, 0.2, 300),
        ]
    )
    _, vecs = masked_pca(data, k=3)
    explained = np.nanvar(vecs, axis=0)
    assert np.all(np.diff(explained) <= 1e-9)


def test_masked_pca_clamps_k_to_available_columns():
    # Regression: k >= n produced a shape error via scipy.sparse.linalg.eigsh.
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 2))
    _, vecs = masked_pca(data, k=5)
    assert vecs.shape == (2, 2)


def test_masked_pca_masks_bad_columns():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 4))
    data[:80, 3] = np.nan  # 80% NaN > default 50% tolerance
    ext_c, ext_vecs = masked_pca(data, k=3)
    assert ext_c.shape == (4, 4)
    assert np.all(np.isnan(ext_c[3, :]))
    assert np.all(np.isnan(ext_vecs[3, :]))


def test_masked_pca_rejects_1d_input():
    # Regression: a 1-D array collapsed the column mask to a scalar and raised
    # a cryptic IndexError.
    with pytest.raises(ValueError):
        masked_pca(np.array([1.0, 2.0, 3.0, 4.0]), k=1)


def test_masked_pca_correlation_drops_constant_columns():
    # Regression: a zero-variance column made the correlation normalizer inf,
    # then eigh raised "Eigenvalues did not converge".
    rng = np.random.default_rng(0)
    data = np.column_stack([rng.normal(size=100), np.zeros(100), rng.normal(size=100)])
    ext_c, ext_vecs = masked_pca(data, correlation=True, k=2)
    assert np.all(np.isnan(ext_c[1, :]))
    assert np.all(np.isnan(ext_vecs[1, :]))
    assert np.isfinite(ext_c[0, 0])
    assert np.isfinite(ext_vecs[0, 0])


def test_bnewt_rejects_zero_row():
    with pytest.raises(ValueError):
        bnewt(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_bnewt_rejects_negative_entry():
    with pytest.raises(ValueError):
        bnewt(np.array([[1.0, -1.0], [1.0, 1.0]]))


def test_bnewt_rejects_non_square():
    with pytest.raises(ValueError):
        bnewt(np.ones((2, 3)))


def test_bnewt_raises_on_non_convergence():
    rng = np.random.default_rng(0)
    A = rng.random((6, 6)) + 0.1
    with pytest.raises(RuntimeError):
        bnewt(A, max_iter=1)
