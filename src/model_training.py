"""
Machine Learning Model Training, Selection, and Persistence Engine.

Trains interpretable baseline and ensemble models to predict strategy P&L and win rate.
Computes comprehensive validation metrics, extracts feature importances, and
persists artifacts to the `artifacts/` directory.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains regression and classification models for quantitative strategy forecasting."""

    def __init__(self, artifacts_dir: Optional[Path] = None):
        settings = get_settings()
        self.artifacts_dir = artifacts_dir or settings.artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(
        self,
        feature_df: pd.DataFrame,
        target_col: str = "target_pnl_1d",
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Extract clean feature matrix X and target vector y, dropping rows with unobserved future targets.
        """
        clean_df = feature_df.dropna(subset=[target_col]).copy()

        if feature_cols:
            selected_features = [c for c in feature_cols if c in clean_df.columns]
        else:
            # Exclude targets, identifiers, text metadata, and synthetic flags from X
            exclude_cols = {
                "id", "symbol", "source", "provider", "is_synthetic",
                "open", "high", "low", "close", "volume",
                "target_pnl_1d", "target_pnl_5d", "target_win_1d", "target_win_5d"
            }
            numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
            selected_features = [c for c in numeric_cols if c not in exclude_cols]

        X = clean_df[selected_features].astype(float).fillna(0.0)
        y = clean_df[target_col].astype(float)

        logger.info(f"Prepared dataset: {len(X)} samples, {len(selected_features)} features for target '{target_col}'.")
        return X, y, selected_features

    def train_regression_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        Train and evaluate Ridge, ElasticNet, and Random Forest regressors.
        Scaler is fit strictly on X_train to prevent data leakage.
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Ridge_Baseline": Ridge(alpha=1.0, random_state=42),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, min_samples_leaf=3),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42),
        }

        results = {}
        for name, model in models.items():
            # Linear models use scaled inputs, trees can use scaled or unscaled
            model.fit(X_train_scaled, y_train)
            preds_train = model.predict(X_train_scaled)
            preds_test = model.predict(X_test_scaled)

            # Compute regression metrics
            mse_train = mean_squared_error(y_train, preds_train)
            mse_test = mean_squared_error(y_test, preds_test)
            mae_test = mean_absolute_error(y_test, preds_test)
            r2_test = r2_score(y_test, preds_test) if len(y_test) > 2 else 0.0

            # Directional accuracy (sign correctness)
            dir_acc_train = np.mean(np.sign(preds_train) == np.sign(y_train)) if len(y_train) > 0 else 0.0
            dir_acc_test = np.mean(np.sign(preds_test) == np.sign(y_test)) if len(y_test) > 0 else 0.0

            # Extract feature importances or coefficients
            if hasattr(model, "feature_importances_"):
                importances = dict(zip(X_train.columns, [round(float(v), 4) for v in model.feature_importances_]))
            elif hasattr(model, "coef_"):
                importances = dict(zip(X_train.columns, [round(float(v), 4) for v in model.coef_]))
            else:
                importances = {}

            results[name] = {
                "model": model,
                "scaler": scaler,
                "mse_train": float(mse_train),
                "mse_test": float(mse_test),
                "mae_test": float(mae_test),
                "r2_test": float(r2_test),
                "directional_accuracy_train": float(dir_acc_train),
                "directional_accuracy_test": float(dir_acc_test),
                "directional_accuracy": float(dir_acc_test),
                "predictions_train": preds_train.tolist(),
                "predictions_test": preds_test.tolist(),
                "feature_importances": importances,
            }

        return results

    def save_model_artifact(
        self,
        model_name: str,
        model_obj: Any,
        scaler_obj: Any,
        feature_names: List[str],
        metadata: Dict[str, Any]
    ) -> Path:
        """Persist model bundle and metadata JSON to artifacts directory."""
        bundle = {
            "model_name": model_name,
            "model": model_obj,
            "scaler": scaler_obj,
            "features": feature_names,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        artifact_path = self.artifacts_dir / f"{model_name}.joblib"
        meta_path = self.artifacts_dir / f"{model_name}_metadata.json"

        joblib.dump(bundle, artifact_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Persisted model artifact to {artifact_path}")
        return artifact_path
