"""
Unit tests for financial performance metrics, Sharpe, Sortino, Drawdown, and edge cases.
"""
import numpy as np
import pandas as pd
import pytest
from src.financial_metrics import (
    calculate_cumulative_returns,
    calculate_drawdown_series,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_trade_metrics,
    evaluate_strategy_performance,
)


def test_sharpe_ratio():
    # Positive constant returns with minor noise
    r = pd.Series([0.01, 0.008, 0.012, 0.009, 0.011, 0.01])
    sharpe = calculate_sharpe_ratio(r, annual_risk_free_rate=0.0)
    assert sharpe > 1.0

    # Zero volatility edge case
    r_zero_vol = pd.Series([0.005, 0.005, 0.005, 0.005])
    assert calculate_sharpe_ratio(r_zero_vol) == 0.0

    # Insufficient samples
    assert calculate_sharpe_ratio(pd.Series([0.01])) == 0.0


def test_sortino_ratio():
    # Mixture of positive and negative returns
    r = pd.Series([0.02, -0.01, 0.015, -0.005, 0.03, 0.01])
    sortino = calculate_sortino_ratio(r, annual_risk_free_rate=0.0)
    assert sortino > 0.0

    # Only positive returns (zero downside deviation)
    r_all_pos = pd.Series([0.01, 0.02, 0.015, 0.03])
    sortino_pos = calculate_sortino_ratio(r_all_pos, annual_risk_free_rate=0.0)
    assert sortino_pos == 10.0  # Safe cap for zero downside deviation


def test_drawdown_calculation():
    # Sequence with a known 20% drawdown: 100 -> 120 -> 96 (drop from 120 is 20%) -> 130
    wealth = pd.Series([100.0, 120.0, 96.0, 130.0])
    dd_series, max_dd_pct, max_dur = calculate_drawdown_series(wealth)

    assert np.isclose(max_dd_pct, 20.0, atol=1e-2)
    assert max_dur >= 1


def test_trade_metrics():
    trades = [
        {"pnl": 500.0, "net_pnl": 480.0},
        {"pnl": -200.0, "net_pnl": -220.0},
        {"pnl": 300.0, "net_pnl": 280.0},
    ]
    res = calculate_trade_metrics(trades)

    assert res["total_trades"] == 3
    assert np.isclose(res["win_rate_pct"], 66.67, atol=0.1)
    assert res["gross_pnl"] == 600.0
    assert res["net_pnl"] == 540.0
    assert res["profit_factor"] > 1.0


def test_trade_metrics_empty():
    res = calculate_trade_metrics([])
    assert res["total_trades"] == 0
    assert res["win_rate_pct"] == 0.0
    assert res["net_pnl"] == 0.0
