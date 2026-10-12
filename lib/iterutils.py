"""
Helpers for scanning contiguous integer sequences.

Both functions require the input to be strictly increasing; unsorted or
duplicate values raise `ValueError`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def scan_spans(xs: Iterable[int]) -> Iterator[tuple[int, int]]:
    """
    Yield `(start, stop)` for each maximal contiguous run in `xs`.

    `stop` is one past the last element. The input must be strictly increasing;
    unsorted or duplicate values raise `ValueError`.
    """
    previous: int | None = None
    start: int | None = None
    stop: int | None = None
    for x in xs:
        if previous is not None and x <= previous:
            raise ValueError("input sequence must be strictly increasing")
        if start is None:
            start = x
        elif x != stop:
            assert stop is not None
            yield start, stop
            start = x
        stop = x + 1
        previous = x
    if start is not None:
        assert stop is not None
        yield start, stop


def require_contiguous(xs: Iterable[int]) -> tuple[int | None, int | None]:
    """
    Return `(start, stop)` for a contiguous integer sequence.

    `stop` is one past the last element. Returns `(None, None)` for an empty
    input and raises `ValueError` if the values are not consecutive integers in
    increasing order.
    """
    spans = list(scan_spans(xs))
    if not spans:
        return None, None
    if len(spans) > 1:
        raise ValueError("input sequence is not contiguous")
    return spans[0]
