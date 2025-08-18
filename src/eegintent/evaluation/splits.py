from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator

import numpy as np

try:  # pragma: no cover
    from sklearn.model_selection import GroupKFold
except Exception:  # pragma: no cover
    GroupKFold = None  # type: ignore[assignment]


def within_subject_holdout(
    subject_ids: np.ndarray,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each subject into train/val/test and concat across subjects."""
    rng = np.random.default_rng(seed)
    subj_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(subject_ids.tolist()):
        subj_to_idx[int(s)].append(i)

    train, val, test = [], [], []
    for idxs in subj_to_idx.values():
        arr = np.array(idxs)
        rng.shuffle(arr)
        n = len(arr)
    n_test = round(test_frac * n)
    n_val = round(val_frac * (n - n_test))
    test_idx = arr[:n_test]
    val_idx = arr[n_test : n_test + n_val]
    train_idx = arr[n_test + n_val :]
    train.extend(train_idx.tolist())
    val.extend(val_idx.tolist())
    test.extend(test_idx.tolist())

    return np.array(train), np.array(val), np.array(test)


def cross_subject_kfold(
    subject_ids: np.ndarray, n_splits: int = 5
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield (train_idx, test_idx) with no subject leakage via GroupKFold."""
    n_samples = len(subject_ids)
    x = np.zeros((n_samples, 1))
    if GroupKFold is None:
        raise RuntimeError("scikit-learn required for cross_subject_kfold")
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(x, groups=subject_ids)


def leave_one_subject_out(
    subject_ids: np.ndarray,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield (train_idx, test_idx) where test is one subject at a time."""
    unique_subj = np.unique(subject_ids)
    for s in unique_subj:
        test = np.where(subject_ids == s)[0]
        train = np.where(subject_ids != s)[0]
        yield train, test
