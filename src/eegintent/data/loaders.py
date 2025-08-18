"""Dataset loaders for MOABB/MNE/BIDS.

Minimal interfaces to enable tests and incremental build-out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Optional MOABB imports at module level for linter friendliness
try:  # pragma: no cover - optional dependency
    from moabb.datasets import BNCI2014_001, BNCI2014_002  # type: ignore
    from moabb.paradigms import MotorImagery  # type: ignore
except Exception:  # pragma: no cover - env dependent
    BNCI2014_001 = None  # type: ignore[assignment]
    BNCI2014_002 = None  # type: ignore[assignment]
    MotorImagery = None  # type: ignore[assignment]


@dataclass
class Dataset:
    X: np.ndarray  # shape: (n_samples, n_channels, n_times)
    y: np.ndarray  # shape: (n_samples,)
    sfreq: float
    ch_names: list[str]


def load_dummy_dataset(n_samples: int = 32, n_channels: int = 8, n_times: int = 256) -> Dataset:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n_samples, n_channels, n_times)).astype(np.float32)
    y = rng.integers(0, 2, size=(n_samples,)).astype(np.int64)
    return Dataset(
        X=x,
        y=y,
        sfreq=256.0,
        ch_names=[f"C{i}" for i in range(n_channels)],
    )


def load_from_config(config: dict) -> Dataset:
    """Load dataset based on a config dict.

    Currently returns a dummy dataset; to be extended with MOABB/MNE loaders.
    """
    backend = config.get("backend", "dummy")
    if backend == "moabb":
        name = config.get("name", "BNCI2014_001")
        paradigm = config.get("paradigm", "mi")
        subjects: list[int] | None = config.get("subjects")
        return load_moabb_dataset(name=name, paradigm=paradigm, subjects=subjects)
    return load_dummy_dataset()


def load_moabb_dataset(
    name: str = "BNCI2014_001",
    paradigm: str = "mi",
    subjects: list[int] | None = None,
) -> Dataset:
    """Load a MOABB dataset minimally, returning numpy arrays.

    Notes:
        - Downloads data to MOABB cache as needed.
        - This is a minimal convenience wrapper for quick experiments.
    """
    if BNCI2014_001 is None or BNCI2014_002 is None or MotorImagery is None:
        raise RuntimeError("moabb is required for MOABB dataset loading")

    ds_map = {
        "BNCI2014_001": BNCI2014_001,
        "BNCI2014_002": BNCI2014_002,
    }
    if name not in ds_map:
        raise ValueError(f"Unsupported MOABB dataset: {name}")
    dataset = ds_map[name]()

    if paradigm != "mi":
        raise ValueError("Only 'mi' paradigm supported in this minimal loader")
    paradigm_obj = MotorImagery()

    x_data, labels, meta = paradigm_obj.get_data(dataset=dataset, subjects=subjects)
    # x_data: dict[subj][sess][run] -> ndarray; flatten to one array
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    for subj_dict in x_data.values():
        for sess_dict in subj_dict.values():
            for run_x in sess_dict.values():
                # run_x shape: (n_trials, n_channels, n_times)
                x_list.append(run_x.astype(np.float32))
    # labels is a parallel structure of lists -> flatten

    def _flatten(lbls):
        out: list[int] = []
        for a in lbls.values():
            for b in a.values():
                out.extend(b)
        return out

    y_list = [int(v) for v in _flatten(labels)]
    x_arr = np.concatenate(x_list, axis=0)
    y_arr = np.asarray(y_list, dtype=np.int64)

    # basic metadata
    sfreq = float(dataset.sfreq) if hasattr(dataset, "sfreq") else 250.0
    ch_names = (
        [str(ch) for ch in dataset.channels]
        if hasattr(dataset, "channels")
        else [f"C{i}" for i in range(x_arr.shape[1])]
    )
    return Dataset(X=x_arr, y=y_arr, sfreq=sfreq, ch_names=ch_names)
