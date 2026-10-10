"""
Financial Performance and Risk Metrics Calculation Engine.

Calculates standard quantitative finance metrics:
- Cumulative Total Return, CAGR, Annualized Volatility
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, Peak-to-Trough Drawdown series
- Win Rate, Profit Factor, Average Trade Return, Gross vs Net P&L
- Edge case handling: zero volatility, zero trades, no negative returns.
"""
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config.settings import get_settings


def calculate_cumulative_returns(returns: Union[pd.Series, np.ndarray]) -> pd.Series:
    """Calculate compounding cumulative wealth index starting at 1.0."""
    r_series = pd.Series(returns).fillna(0.0)
    return (1.0 + r_series).cumprod()


def calculate_drawdown_series(cumulative_wealth: Union[pd.Series, np.ndarray]) -> Tuple[pd.Series, float, int]:
    """
    Calculate underwater drawdown series, maximum drawdown percentage, and max duration.

    Returns:
        (drawdown_series, max_drawdown_pct, max_drawdown_duration_bars)
    """
    wealth = pd.Series(cumulative_wealth).fillna(1.0)
    if wealth.empty:
        return pd.Series([0.0]), 0.0, 0

    running_max = wealth.cummax()
    drawdowns = (wealth - running_max) / running_max
    max_dd = float(abs(drawdowns.min())) if not drawdowns.empty else 0.0

    # Drawdown duration calculation
    is_underwater = drawdowns < 0
    duration = 0
    max_duration = 0
    for underwater in is_underwater:
        if underwater:
            duration += 1
            if duration > max_duration:
                max_duration = duration
        else:
            duration = 0

    return drawdowns, max_dd * 100.0, max_duration


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    annual_risk_free_rate: Optional[float] = None,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Args:
        returns: Period returns series (e.g. daily).
        annual_risk_free_rate: Annualized risk-free rate (e.g. 0.045 for 4.5%).
        periods_per_year: 252 for daily trading bars.

    Returns:
        Annualized Sharpe Ratio or 0.0 if zero volatility / insufficient samples.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0

    settings = get_settings()
    rf = annual_risk_free_rate if annual_risk_free_rate is not None else settings.annual_risk_free_rate
    daily_rf = (1.0 + rf) ** (1.0 / periods_per_year) - 1.0

    excess_returns = r - daily_rf
    std = excess_returns.std(ddof=1)

    if std <= 1e-9 or math.isnan(std):
        return 0.0

    mean_excess = excess_returns.mean()
    sharpe = (mean_excess / std) * np.sqrt(periods_per_year)
    return float(round(sharpe, 4)) if not math.isnan(sharpe) else 0.0


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    annual_risk_free_rate: Optional[float] = None,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino Ratio using semi-variance of downside returns.

    Args:
        returns: Period returns series.
        annual_risk_free_rate: Annualized risk-free rate.
        periods_per_year: 252 for daily bars.

    Returns:
        Annualized Sortino Ratio. Returns large positive or 0.0 if no downside volatility exists.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0

    settings = get_settings()
    rf = annual_risk_free_rate if annual_risk_free_rate is not None else settings.annual_risk_free_rate
    daily_rf = (1.0 + rf) ** (1.0 / periods_per_year) - 1.0

    excess_returns = r - daily_rf
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        # No negative excess returns: Sortino is exceptionally high or capped
        mean_excess = excess_returns.mean()
        return 10.0 if mean_excess > 0 else 0.0

    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    if downside_std <= 1e-9 or math.isnan(downside_std):
        return 0.0

    mean_excess = excess_returns.mean()
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    return float(round(sortino, 4)) if not math.isnan(sortino) else 0.0


def calculate_trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate granular execution and trade statistics.

    Args:
        trades: List of trade dictionaries with keys: 'pnl', 'net_pnl', 'return_pct'.

    Returns:
        Dict with total_trades, win_rate_pct, profit_factor, avg_trade_pnl, gross_pnl, net_pnl.
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
        }

    pnl_list = [t.get("pnl", 0.0) for t in trades]
    net_pnl_list = [t.get("net_pnl", 0.0) for t in trades]

    wins = [p for p in net_pnl_list if p > 0]
    losses = [p for p in net_pnl_list if p <= 0]

    total_trades = len(trades)
    win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0
    gross_pnl = sum(pnl_list)
    net_pnl = sum(net_pnl_list)
    avg_trade_pnl = net_pnl / total_trades if total_trades > 0 else 0.0

    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    profit_factor = (sum_wins / sum_losses) if sum_losses > 1e-6 else (10.0 if sum_wins > 0 else 0.0)

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    win_loss_ratio = (avg_win / abs(avg_loss)) if abs(avg_loss) > 1e-6 else 0.0

    return {
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(float(profit_factor), 2),
        "avg_trade_pnl": round(float(avg_trade_pnl), 2),
        "gross_pnl": round(float(gross_pnl), 2),
        "net_pnl": round(float(net_pnl), 2),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "win_loss_ratio": round(float(win_loss_ratio), 2),
    }


def evaluate_strategy_performance(
    returns_series: pd.Series,
    trades: List[Dict[str, Any]],
    initial_capital: float = 100_000.0,
    annual_risk_free_rate: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute full quantitative performance summary for a strategy returns stream.
    """
    r = returns_series.fillna(0.0)
    cum_wealth = calculate_cumulative_returns(r)
    final_equity = initial_capital * cum_wealth.iloc[-1] if not cum_wealth.empty else initial_capital
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0

    drawdowns, max_dd_pct, max_dd_duration = calculate_drawdown_series(cum_wealth)
    sharpe = calculate_sharpe_ratio(r, annual_risk_free_rate)
    sortino = calculate_sortino_ratio(r, annual_risk_free_rate)

    # Annualized Volatility & CAGR
    n_days = len(r)
    ann_factor = 252.0 / max(n_days, 1)
    cagr = ((cum_wealth.iloc[-1] ** (252.0 / max(n_days, 1))) - 1.0) * 100.0 if not cum_wealth.empty and cum_wealth.iloc[-1] > 0 else 0.0
    ann_vol = (r.std(ddof=1) * np.sqrt(252.0)) * 100.0 if len(r) > 1 else 0.0
    calmar = (cagr / max_dd_pct) if max_dd_pct > 1e-4 else 0.0

    trade_stats = calculate_trade_metrics(trades)

    return {
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(float(cagr), 2),
        "annualized_vol_pct": round(float(ann_vol), 2),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": round(float(calmar), 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_drawdown_duration_bars": max_dd_duration,
        "gross_pnl": trade_stats["gross_pnl"],
        "net_pnl": trade_stats["net_pnl"],
        "win_rate_pct": trade_stats["win_rate_pct"],
        "profit_factor": trade_stats["profit_factor"],
        "total_trades": trade_stats["total_trades"],
        "avg_trade_pnl": trade_stats["avg_trade_pnl"],
    }
