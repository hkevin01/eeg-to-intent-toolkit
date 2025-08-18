"""Dataset loaders for MOABB/MNE/BIDS.

These are stubs with minimal interfaces to enable tests and incremental build-out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray  # shape: (n_samples, n_channels, n_times)
    y: np.ndarray  # shape: (n_samples,)
    sfreq: float
    ch_names: List[str]


def load_dummy_dataset(n_samples: int = 32, n_channels: int = 8, n_times: int = 256) -> Dataset:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_channels, n_times)).astype(np.float32)
    y = rng.integers(0, 2, size=(n_samples,)).astype(np.int64)
    return Dataset(X=X, y=y, sfreq=256.0, ch_names=[f"C{i}" for i in range(n_channels)])


def load_from_config(config: Dict) -> Dataset:
    """Load dataset based on a config dict.

    Currently returns a dummy dataset; to be extended with MOABB/MNE loaders.
    """
    _ = config
    return load_dummy_dataset()
