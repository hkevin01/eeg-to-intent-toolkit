from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .loaders import Dataset


def concat_pool(datasets: Sequence[Dataset]) -> Dataset:
    """Concatenate multiple Dataset objects if channel sets match.

    Assumes same sfreq and channel order; raises if mismatch.
    """
    if not datasets:
        raise ValueError("No datasets provided")
    sfreq = datasets[0].sfreq
    chs = datasets[0].ch_names
    for ds in datasets:
        if ds.sfreq != sfreq or ds.ch_names != chs:
            raise ValueError("Datasets must have identical sfreq and channels to concat")
    x = np.concatenate([d.X for d in datasets], axis=0)
    y = np.concatenate([d.y for d in datasets], axis=0)
    return Dataset(X=x, y=y, sfreq=sfreq, ch_names=chs)
