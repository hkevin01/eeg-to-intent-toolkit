import numpy as np

from eegintent.data.loaders import load_dummy_dataset
from eegintent.data.preprocess import zscore


def test_dummy_dataset_shapes():
    ds = load_dummy_dataset(n_samples=10, n_channels=4, n_times=100)
    assert ds.X.shape == (10, 4, 100)
    assert ds.y.shape == (10,)
    assert len(ds.ch_names) == 4


def test_zscore_normalization():
    x = np.random.randn(2, 3, 5).astype(np.float32)
    xz = zscore(x, axis=-1)
    m = xz.mean(axis=-1)
    s = xz.std(axis=-1)
    assert np.allclose(m, 0, atol=1e-5)
    assert np.allclose(s, 1, atol=1e-4)
