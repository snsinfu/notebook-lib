import pytest

from lib.iterutils import require_contiguous, scan_spans


def test_require_contiguous_returns_bounds():
    assert require_contiguous([3, 4, 5]) == (3, 6)


def test_require_contiguous_rejects_gaps():
    with pytest.raises(ValueError):
        require_contiguous([1, 3])


def test_require_contiguous_empty():
    assert require_contiguous([]) == (None, None)


def test_scan_spans_splits_runs():
    assert list(scan_spans([1, 2, 3, 7, 8])) == [(1, 4), (7, 9)]


def test_scan_spans_empty():
    assert list(scan_spans([])) == []


def test_scan_spans_rejects_unsorted_input():
    # Regression: unsorted input silently produced bogus spans.
    with pytest.raises(ValueError):
        list(scan_spans([1, 3, 2]))


def test_scan_spans_rejects_duplicates():
    with pytest.raises(ValueError):
        list(scan_spans([1, 1]))
