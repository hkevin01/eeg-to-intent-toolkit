import numpy as np

from eegintent.realtime.dsp import sliding_mean


def test_sliding_mean():
    x = np.ones((2, 3, 5), dtype=np.float32)
    x[..., 2] = 3.0
    y = sliding_mean(x, k=3)
    # Center value should be averaged with neighbors: (1 + 3 + 1)/3 = 1.666...
    assert np.allclose(y[..., 2], 5.0 / 3.0)
