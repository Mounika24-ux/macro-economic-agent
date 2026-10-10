"""
Chronological Walk-Forward Validation Engine.

Implements expanding and rolling window chronological cross-validation with
purging/embargo to prevent lookahead bias and target autocorrelation leakage.
Evaluates out-of-sample model predictions and quantitative trading strategies.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.financial_metrics import evaluate_strategy_performance
from src.model_training import ModelTrainer

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Performs rigorous out-of-sample chronological backtesting and ML validation."""

    def __init__(
        self,
        n_splits: Optional[int] = None,
        window_type: Optional[str] = None,
        train_size_days: Optional[int] = None,
        test_size_days: Optional[int] = None,
        purge_days: Optional[int] = None,
    ):
        settings = get_settings()
        self.n_splits = n_splits or settings.wf_n_splits
        self.window_type = window_type or settings.wf_window_type
        self.train_size_days = train_size_days or settings.wf_train_size_days
        self.test_size_days = test_size_days or settings.wf_test_size_days
        self.purge_days = purge_days or settings.wf_purge_days
        self.trainer = ModelTrainer()

    def generate_folds(
        self,
        dates: pd.DatetimeIndex
    ) -> List[Dict[str, Any]]:
        """
        Generate chronological (train_indices, test_indices) with purge gap.

        Returns:
            List of fold metadata dicts with 'fold_idx', 'train_idx', 'test_idx', 'train_dates', 'test_dates'.
        """
        n_samples = len(dates)
        if n_samples < 30:
            raise ValueError(f"Insufficient samples ({n_samples}) for walk-forward validation (minimum 30 required).")

        # Determine step size
        # We need n_splits test periods.
        # Total days needed approx = min_train_samples + n_splits * test_samples
        test_samples = max(5, int(n_samples / (self.n_splits + 2)))
        min_train_samples = max(15, int(n_samples * 0.4))
        purge_samples = max(1, self.purge_days)

        folds = []
        for i in range(self.n_splits):
            test_end_idx = n_samples - (self.n_splits - 1 - i) * test_samples
            test_start_idx = test_end_idx - test_samples

            # Train end is before test start minus purge gap to eliminate forward label overlap
            train_end_idx = test_start_idx - purge_samples

            if train_end_idx < min_train_samples:
                continue

            if self.window_type == "expanding":
                train_start_idx = 0
            else:
                # Rolling window
                train_start_idx = max(0, train_end_idx - min_train_samples)

            train_indices = np.arange(train_start_idx, train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)

            if len(train_indices) >= 10 and len(test_indices) >= 3:
                folds.append({
                    "fold_idx": len(folds) + 1,
                    "train_idx": train_indices,
                    "test_idx": test_indices,
                    "train_start_date": dates[train_start_idx].strftime("%Y-%m-%d"),
                    "train_end_date": dates[train_end_idx - 1].strftime("%Y-%m-%d"),
                    "test_start_date": dates[test_start_idx].strftime("%Y-%m-%d"),
                    "test_end_date": dates[test_end_idx - 1].strftime("%Y-%m-%d"),
                    "train_count": len(train_indices),
                    "test_count": len(test_indices),
                })

        logger.info(f"Generated {len(folds)} valid walk-forward folds (window: {self.window_type}, purge: {purge_samples} bars).")
        return folds

    def run_walk_forward_validation(
        self,
        feature_df: pd.DataFrame,
        target_col: str = "target_pnl_1d"
    ) -> Dict[str, Any]:
        """
        Execute full chronological walk-forward ML evaluation.

        Returns:
            Dictionary with fold-level metrics, aggregated out-of-sample metrics,
            feature importances, and predicted return streams.
        """
        df = feature_df.dropna(subset=[target_col]).copy().sort_index()
        dates = df.index

        X, y, feature_names = self.trainer.prepare_dataset(df, target_col=target_col)
        folds = self.generate_folds(dates)

        if not folds:
            raise ValueError("Could not construct valid walk-forward folds with the available sample size.")

        fold_results = []
        all_oos_predictions: Dict[str, List[float]] = {
            "Ridge_Baseline": [],
            "ElasticNet": [],
            "RandomForest": [],
            "GradientBoosting": []
        }
        all_oos_actuals: List[float] = []
        all_oos_dates: List[pd.Timestamp] = []

        last_models = {}

        for fold in folds:
            f_idx = fold["fold_idx"]
            tr_idx = fold["train_idx"]
            te_idx = fold["test_idx"]

            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
            fold_test_dates = dates[te_idx]

            # Train all model variants strictly on training fold
            res = self.trainer.train_regression_models(X_tr, y_tr, X_te, y_te)
            last_models = res

            fold_metric_entry = {
                "fold_idx": f_idx,
                "train_start": fold["train_start_date"],
                "train_end": fold["train_end_date"],
                "test_start": fold["test_start_date"],
                "test_end": fold["test_end_date"],
                "train_samples": len(X_tr),
                "test_samples": len(X_te),
                "models": {}
            }

            for model_name, m_res in res.items():
                fold_metric_entry["models"][model_name] = {
                    "mse_test": m_res["mse_test"],
                    "mae_test": m_res["mae_test"],
                    "r2_test": m_res["r2_test"],
                    "directional_accuracy": m_res["directional_accuracy"],
                }
                all_oos_predictions[model_name].extend(m_res["predictions_test"])

            all_oos_actuals.extend(y_te.tolist())
            all_oos_dates.extend(fold_test_dates.tolist())
            fold_results.append(fold_metric_entry)

        # Aggregate Out-of-Sample Overall Metrics
        aggregate_metrics = {}
        oos_actuals_arr = np.array(all_oos_actuals)

        for model_name, preds_list in all_oos_predictions.items():
            preds_arr = np.array(preds_list)
            if len(preds_arr) > 0 and len(oos_actuals_arr) > 0:
                mse_oos = float(np.mean((oos_actuals_arr - preds_arr) ** 2))
                mae_oos = float(np.mean(np.abs(oos_actuals_arr - preds_arr)))
                dir_acc_oos = float(np.mean(np.sign(preds_arr) == np.sign(oos_actuals_arr)))
                total_variance = float(np.var(oos_actuals_arr))
                r2_oos = float(1.0 - (mse_oos / total_variance)) if total_variance > 1e-9 else 0.0

                # Construct ML-driven trading return series: Signal = sign(predicted return)
                ml_signals = np.sign(preds_arr)
                ml_returns = ml_signals * oos_actuals_arr
                ml_return_series = pd.Series(ml_returns, index=all_oos_dates)
                ml_perf = evaluate_strategy_performance(ml_return_series, [], initial_capital=100_000.0)

                aggregate_metrics[model_name] = {
                    "oos_mse": round(mse_oos, 6),
                    "oos_mae": round(mae_oos, 6),
                    "oos_r2": round(r2_oos, 4),
                    "oos_directional_accuracy": round(dir_acc_oos * 100.0, 2),
                    "strategy_sharpe": ml_perf["sharpe_ratio"],
                    "strategy_sortino": ml_perf["sortino_ratio"],
                    "strategy_total_return_pct": ml_perf["total_return_pct"],
                    "strategy_max_drawdown_pct": ml_perf["max_drawdown_pct"],
                }

        # Baseline Benchmark: Buy-and-Hold on OOS period
        oos_asset_returns = pd.Series(all_oos_actuals, index=all_oos_dates)
        benchmark_perf = evaluate_strategy_performance(oos_asset_returns, [], initial_capital=100_000.0)
        aggregate_metrics["Benchmark_BuyAndHold"] = {
            "strategy_sharpe": benchmark_perf["sharpe_ratio"],
            "strategy_sortino": benchmark_perf["sortino_ratio"],
            "strategy_total_return_pct": benchmark_perf["total_return_pct"],
            "strategy_max_drawdown_pct": benchmark_perf["max_drawdown_pct"],
        }

        # Save Best Model Artifact
        best_model_name = max(
            [m for m in aggregate_metrics.keys() if m != "Benchmark_BuyAndHold"],
            key=lambda k: aggregate_metrics[k]["oos_directional_accuracy"]
        )
        if best_model_name in last_models:
            best_model_bundle = last_models[best_model_name]
            self.trainer.save_model_artifact(
                model_name=best_model_name,
                model_obj=best_model_bundle["model"],
                scaler_obj=best_model_bundle["scaler"],
                feature_names=feature_names,
                metadata={
                    "best_model_name": best_model_name,
                    "target": target_col,
                    "n_folds": len(folds),
                    "aggregate_metrics": aggregate_metrics[best_model_name],
                    "feature_importances": best_model_bundle.get("feature_importances", {}),
                }
            )

        return {
            "folds": fold_results,
            "aggregate_metrics": aggregate_metrics,
            "best_model_name": best_model_name,
            "feature_names": feature_names,
            "oos_dates": [d.strftime("%Y-%m-%d") for d in all_oos_dates],
            "oos_actuals": all_oos_actuals,
            "oos_predictions": all_oos_predictions,
        }
