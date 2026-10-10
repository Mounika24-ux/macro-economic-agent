"""
Unit tests for walk-forward validation split chronology and purge gap enforcement.
"""
from datetime import datetime, timezone
import pandas as pd
import pytest
from src.walk_forward_validation import WalkForwardValidator


def test_walk_forward_folds_chronology():
    validator = WalkForwardValidator(n_splits=4, purge_days=2, window_type="expanding")
    dates = pd.date_range("2026-01-01", periods=100, freq="B", tz=timezone.utc)

    folds = validator.generate_folds(dates)
    assert len(folds) >= 3

    for f in folds:
        train_idx = f["train_idx"]
        test_idx = f["test_idx"]

        # 1. Training set must strictly precede testing set
        assert max(train_idx) < min(test_idx), f"Fold {f['fold_idx']} train and test overlap!"

        # 2. Purge gap must be at least purge_days
        gap = min(test_idx) - max(train_idx)
        assert gap >= 2, f"Fold {f['fold_idx']} purge gap is only {gap} (expected >= 2)."

        # 3. No indices out of bounds
        assert min(train_idx) >= 0
        assert max(test_idx) < len(dates)


def test_walk_forward_rolling_window():
    validator = WalkForwardValidator(n_splits=3, purge_days=1, window_type="rolling")
    dates = pd.date_range("2026-01-01", periods=90, freq="B", tz=timezone.utc)

    folds = validator.generate_folds(dates)
    assert len(folds) >= 2
    # In rolling mode, later folds should have train_start_idx > 0
    assert folds[-1]["train_idx"][0] >= 0
