"""
Quantitative Visualization Suite for Macroeconomic Intelligence.

Generates static publication-ready figures (Matplotlib / Seaborn saved to reports/figures/)
and interactive figures (Plotly) for Streamlit and HTML reports:
1. Macroeconomic Sentiment Timeline & News Volume
2. Economic Event Surprises Over Time
3. Strategy Equity Curves & Drawdown Underwater Chart
4. Strategy Parameter Permutation Heatmap
5. Model Out-of-Sample Performance Comparison Matrix
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Apply clean quantitative theme
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class Visualizer:
    """Generates charts for reports and dashboards."""

    def __init__(self, figures_dir: Optional[Path] = None):
        settings = get_settings()
        self.figures_dir = figures_dir or settings.figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_sentiment_timeline(
        self,
        feature_df: pd.DataFrame,
        save_filename: str = "sentiment_timeline.png"
    ) -> Path:
        """Plot rolling sentiment and news volume over time."""
        fig, ax1 = plt.subplots(figsize=(11, 5), dpi=150)

        dates = feature_df.index
        sent_daily = feature_df.get("news_sentiment_daily", pd.Series(0, index=dates))
        sent_7d = feature_df.get("news_sentiment_7d", pd.Series(0, index=dates))
        volume_1d = feature_df.get("news_volume_1d", pd.Series(0, index=dates))

        # Bar plot for volume
        ax2 = ax1.twinx()
        ax2.bar(dates, volume_1d, color="#d0e1fd", alpha=0.4, width=0.8, label="Daily News Volume")
        ax2.set_ylabel("Articles Published / Day", color="#5c768d", fontsize=10)
        ax2.grid(False)

        # Line plot for sentiment
        ax1.plot(dates, sent_daily, color="#94a3b8", alpha=0.5, linestyle=":", label="Daily Raw Sentiment")
        ax1.plot(dates, sent_7d, color="#1e40af", linewidth=2.0, label="7-Day Rolling Sentiment")
        ax1.axhline(0, color="#64748b", linestyle="--", linewidth=0.8)
        ax1.set_ylabel("Macro Sentiment Score [-1.0 to +1.0]", color="#1e293b", fontsize=10)
        ax1.set_ylim(-1.05, 1.05)

        plt.title("Macroeconomic News Sentiment & Volume Timeline", fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

        out_path = self.figures_dir / save_filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved sentiment timeline chart to {out_path}")
        return out_path

    def plot_event_surprises(
        self,
        events: List[Dict[str, Any]],
        save_filename: str = "event_surprises.png"
    ) -> Path:
        """Scatter/Stem plot of macroeconomic indicator surprises."""
        fig, ax = plt.subplots(figsize=(11, 5), dpi=150)

        ev_with_surprise = [e for e in events if e.get("surprise") is not None]
        if not ev_with_surprise:
            ax.text(0.5, 0.5, "No Economic Event Surprises Available", horizontalalignment="center", verticalalignment="center")
            out_path = self.figures_dir / save_filename
            plt.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return out_path

        times = [e["timestamp_utc"] for e in ev_with_surprise]
        surprises = [e["surprise"] for e in ev_with_surprise]
        impacts = [e.get("impact_level", "Medium") for e in ev_with_surprise]

        colors = ["#ef4444" if s < 0 else "#22c55e" for s in surprises]
        sizes = [90 if imp == "High" else 40 for imp in impacts]

        ax.scatter(times, surprises, c=colors, s=sizes, alpha=0.85, edgecolors="#1e293b", linewidth=0.5, zorder=3)
        ax.axhline(0, color="#64748b", linestyle="-", linewidth=1.0, zorder=2)

        ax.set_ylabel("Surprise Delta (Actual - Forecast)", fontsize=10)
        ax.set_title("Macroeconomic Indicator Surprises Over Time", fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()

        out_path = self.figures_dir / save_filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved event surprises chart to {out_path}")
        return out_path

    def plot_equity_and_drawdown(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_label: str = "ML Macro Strategy",
        benchmark_label: str = "Benchmark (Buy & Hold)",
        save_filename: str = "equity_and_drawdown.png"
    ) -> Path:
        """Two-panel chart: Cumulative Equity Curves and Underwater Drawdowns."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}, dpi=150)

        strat_cum = (1.0 + strategy_returns.fillna(0.0)).cumprod() * 100_000.0
        bench_cum = (1.0 + benchmark_returns.fillna(0.0)).cumprod() * 100_000.0

        # Panel 1: Equity Curve
        ax1.plot(strat_cum.index, strat_cum.values, label=strategy_label, color="#2563eb", linewidth=2.0)
        ax1.plot(bench_cum.index, bench_cum.values, label=benchmark_label, color="#94a3b8", linewidth=1.5, linestyle="--")
        ax1.set_ylabel("Portfolio Value (USD, $100k Base)", fontsize=10)
        ax1.set_title("Out-of-Sample Strategy Cumulative Equity & Underwater Drawdown", fontsize=13, fontweight="bold", pad=12)
        ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

        # Panel 2: Underwater Drawdown
        strat_max = strat_cum.cummax()
        strat_dd = (strat_cum - strat_max) / strat_max * 100.0

        bench_max = bench_cum.cummax()
        bench_dd = (bench_cum - bench_max) / bench_max * 100.0

        ax2.fill_between(strat_dd.index, strat_dd.values, 0, color="#ef4444", alpha=0.35, label=f"{strategy_label} Drawdown")
        ax2.plot(bench_dd.index, bench_dd.values, color="#64748b", linewidth=1.0, linestyle=":", label=f"{benchmark_label} Drawdown")
        ax2.set_ylabel("Drawdown %", fontsize=10)
        ax2.set_ylim(min(-20.0, strat_dd.min() * 1.15), 1.0)
        ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

        fig.tight_layout()
        out_path = self.figures_dir / save_filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved equity and drawdown chart to {out_path}")
        return out_path

    def plot_strategy_heatmap(
        self,
        strategy_grid_df: pd.DataFrame,
        save_filename: str = "strategy_heatmap.png"
    ) -> Path:
        """Heatmap of Sharpe ratios across Stop-Loss vs Take-Profit permutations."""
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

        if strategy_grid_df.empty:
            ax.text(0.5, 0.5, "No Strategy Grid Results", ha="center", va="center")
            out_path = self.figures_dir / save_filename
            plt.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return out_path

        # Pivot by Stop Loss and Take Profit
        pivot_data = strategy_grid_df.pivot_table(
            index="stop_loss_pct",
            columns="take_profit_pct",
            values="sharpe_ratio",
            aggfunc="mean"
        )

        # Format axes labels as %
        pivot_data.index = [f"{int(x*100)}%" for x in pivot_data.index]
        pivot_data.columns = [f"{int(x*100)}%" for x in pivot_data.columns]

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0.0,
            cbar_kws={"label": "Sharpe Ratio (Annualized)"},
            ax=ax
        )
        ax.set_title("Strategy Parameter Grid: Sharpe Ratio by Stop-Loss & Take-Profit", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Take-Profit Threshold", fontsize=10)
        ax.set_ylabel("Stop-Loss Threshold", fontsize=10)

        fig.tight_layout()
        out_path = self.figures_dir / save_filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved strategy heatmap to {out_path}")
        return out_path

    def plot_model_comparison(
        self,
        aggregate_metrics: Dict[str, Dict[str, Any]],
        save_filename: str = "model_comparison.png"
    ) -> Path:
        """Bar chart comparing out-of-sample directional accuracy and Sharpe ratio across ML models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

        models = [m for m in aggregate_metrics.keys() if "Benchmark" not in m]
        if not models:
            out_path = self.figures_dir / save_filename
            plt.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return out_path

        dir_accs = [aggregate_metrics[m].get("oos_directional_accuracy", 0.0) for m in models]
        sharpes = [aggregate_metrics[m].get("strategy_sharpe", 0.0) for m in models]

        short_names = [m.replace("Baseline", "Base").replace("GradientBoosting", "GradBoost") for m in models]

        # Panel 1: Directional Accuracy
        bars1 = ax1.bar(short_names, dir_accs, color="#3b82f6", alpha=0.85, edgecolor="#1e3a8a")
        ax1.axhline(50.0, color="#ef4444", linestyle="--", linewidth=1.0, label="Random Guess (50%)")
        ax1.set_ylabel("Directional Accuracy %", fontsize=10)
        ax1.set_ylim(35, 75)
        ax1.set_title("Out-of-Sample Directional Accuracy", fontsize=11, fontweight="bold")
        ax1.legend(loc="lower right", fontsize=8)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

        # Panel 2: Strategy Sharpe Ratio
        bars2 = ax2.bar(short_names, sharpes, color="#10b981", alpha=0.85, edgecolor="#065f46")
        ax2.axhline(0, color="#64748b", linestyle="-", linewidth=0.8)
        ax2.set_ylabel("Annualized Sharpe Ratio", fontsize=10)
        ax2.set_title("ML Strategy Risk-Adjusted Sharpe Ratio", fontsize=11, fontweight="bold")
        for bar in bars2:
            yval = bar.get_height()
            offset = 0.05 if yval >= 0 else -0.15
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        out_path = self.figures_dir / save_filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved model comparison chart to {out_path}")
        return out_path
