"""
Unit tests for strategy simulation, execution accounting, and parameter grid.
"""
import pandas as pd
import pytest
from src.feature_engineering import FeatureEngineer
from src.strategy_engine import StrategyBacktester


def test_strategy_backtester_accounting(sample_market_df, sample_events, sample_news):
    fe = FeatureEngineer()
    feature_df = fe.build_feature_matrix(sample_market_df, sample_events, sample_news)

    backtester = StrategyBacktester(commission_per_share=0.001, slippage_bps=2.0)
    returns, trades, metrics = backtester.run_backtest(
        feature_df=feature_df,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        lookback_window=5,
        sentiment_threshold=0.02
    )

    assert isinstance(returns, pd.Series)
    assert len(returns) == len(feature_df)

    if trades:
        # Check net P&L is less than gross P&L due to transaction fees and slippage
        total_gross = sum(t["pnl"] for t in trades)
        total_net = sum(t["net_pnl"] for t in trades)
        assert total_net <= total_gross


def test_strategy_grid_sweep(sample_market_df, sample_events, sample_news):
    fe = FeatureEngineer()
    feature_df = fe.build_feature_matrix(sample_market_df, sample_events, sample_news)

    backtester = StrategyBacktester()
    grid_df = backtester.run_strategy_grid(
        feature_df=feature_df,
        stop_losses=[0.02, 0.03],
        take_profits=[0.04, 0.06],
        lookback_windows=[5],
        sentiment_thresholds=[0.05]
    )

    assert len(grid_df) == 4  # 2 x 2 x 1 x 1 = 4 combinations
    assert "sharpe_ratio" in grid_df.columns
    assert "net_pnl" in grid_df.columns
    assert grid_df["sharpe_ratio"].iloc[0] >= grid_df["sharpe_ratio"].iloc[-1]
