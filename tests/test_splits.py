import numpy as np

from eegintent.evaluation.splits import (
    cross_subject_kfold,
    leave_one_subject_out,
    within_subject_holdout,
)


def test_within_subject_holdout_shapes():
    subject_ids = np.array([0] * 10 + [1] * 12)
    train, val, test = within_subject_holdout(subject_ids, 0.2, 0.2, 0)
    assert set(train).isdisjoint(set(val))
    assert set(train).isdisjoint(set(test))
    assert set(val).isdisjoint(set(test))
    assert len(train) + len(val) + len(test) == len(subject_ids)


def test_cross_subject_kfold_group_separation():
    subject_ids = np.array([0] * 5 + [1] * 5 + [2] * 5)
    n_splits = 3
    folds = list(cross_subject_kfold(subject_ids, n_splits=n_splits))
    assert len(folds) == n_splits
    for train, test in folds:
        assert set(subject_ids[train]).isdisjoint(set(subject_ids[test]))


def test_leave_one_subject_out():
    subject_ids = np.array([0] * 3 + [1] * 4)
    n_subjects = 2
    folds = list(leave_one_subject_out(subject_ids))
    assert len(folds) == n_subjects
    for train, test in folds:
        assert set(subject_ids[train]).isdisjoint(set(subject_ids[test]))
        assert len(train) + len(test) == len(subject_ids)
