"""Shared pytest setup."""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

# Make the repository's ``lib`` importable when running pytest from the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture
def rng():
    return np.random.default_rng(0)
