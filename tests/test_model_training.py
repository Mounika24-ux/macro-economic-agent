"""
Unit tests for ML model training, scaling, comparison, and artifact serialization.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.model_training import ModelTrainer


def test_train_regression_models(tmp_path):
    trainer = ModelTrainer(artifacts_dir=tmp_path)

    # Synthetic training & test matrices
    np.random.seed(42)
    n_train = 60
    n_test = 20
    n_features = 8

    X_train = pd.DataFrame(np.random.randn(n_train, n_features), columns=[f"f_{i}" for i in range(n_features)])
    y_train = pd.Series(X_train["f_0"] * 0.5 + np.random.randn(n_train) * 0.1)

    X_test = pd.DataFrame(np.random.randn(n_test, n_features), columns=[f"f_{i}" for i in range(n_features)])
    y_test = pd.Series(X_test["f_0"] * 0.5 + np.random.randn(n_test) * 0.1)

    results = trainer.train_regression_models(X_train, y_train, X_test, y_test)

    assert "Ridge_Baseline" in results
    assert "RandomForest" in results
    assert "ElasticNet" in results
    assert "GradientBoosting" in results

    ridge_res = results["Ridge_Baseline"]
    assert "mse_test" in ridge_res
    assert "directional_accuracy" in ridge_res
    assert len(ridge_res["predictions_test"]) == n_test


def test_save_model_artifact(tmp_path):
    trainer = ModelTrainer(artifacts_dir=tmp_path)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    model = Ridge().fit([[1, 2], [3, 4]], [1, 2])
    scaler = StandardScaler().fit([[1, 2], [3, 4]])

    path = trainer.save_model_artifact(
        model_name="test_ridge",
        model_obj=model,
        scaler_obj=scaler,
        feature_names=["f1", "f2"],
        metadata={"accuracy": 0.85}
    )

    assert path.exists()
    assert (tmp_path / "test_ridge_metadata.json").exists()
