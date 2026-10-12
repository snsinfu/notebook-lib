import numpy as np
import pytest

from lib.contacts import compute_contact_matrix


def test_contact_matrix_is_symmetric_with_zero_diagonal():
    points = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
    matrix = compute_contact_matrix(points, 0.5)
    assert matrix.shape == (3, 3)
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0)
    assert matrix[0, 1] == 1


def test_contact_matrix_box_wraps_points():
    points = np.array([[0.0, 0.0], [9.9, 0.0]])
    matrix = compute_contact_matrix(points, 0.5, box_size=10.0)
    assert matrix[0, 1] == 1


def test_contact_matrix_returns_integer_counts():
    # Regression: counts were accumulated into a float matrix.
    points = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
    matrix = compute_contact_matrix(points, 0.5)
    assert np.issubdtype(matrix.dtype, np.integer)


def test_contact_matrix_folds_exact_box_boundary():
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    matrix = compute_contact_matrix(points, 0.5, box_size=10.0)
    assert matrix[0, 1] == 1


def test_contact_matrix_rejects_1d_points():
    with pytest.raises(ValueError):
        compute_contact_matrix(np.array([0.0, 1.0]), 0.5)
