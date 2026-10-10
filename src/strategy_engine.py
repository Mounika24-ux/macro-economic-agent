"""
Strategy Permutation Grid and Backtesting Engine.

Simulates quantitative trading strategies driven by macroeconomic sentiment,
event surprise signals, and technical indicators across a configurable parameter grid.
Models realistic slippage and transaction commissions without lookahead bias.
"""
import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.financial_metrics import evaluate_strategy_performance

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """Simulates trading strategies on historical OHLCV and feature datasets."""

    def __init__(
        self,
        commission_per_share: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        initial_capital: float = 100_000.0,
    ):
        settings = get_settings()
        self.commission_per_share = commission_per_share if commission_per_share is not None else settings.commission_per_share
        self.slippage_bps = slippage_bps if slippage_bps is not None else settings.slippage_bps
        self.initial_capital = initial_capital

    def run_backtest(
        self,
        feature_df: pd.DataFrame,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        lookback_window: int = 5,
        sentiment_threshold: float = 0.05,
        holding_period_bars: int = 5,
        position_size_dollars: float = 10_000.0,
    ) -> Tuple[pd.Series, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute backtest for a specific parameter combination.

        Strategy Logic:
        - Enter LONG when rolling sentiment > threshold AND 5d return is positive (momentum alignment)
          OR when positive macro event surprise occurs.
        - Enter SHORT when rolling sentiment < -threshold AND 5d return is negative.
        - Intra-bar Stop-Loss and Take-Profit execution checking High and Low.
        - Exit at market close after holding_period_bars if neither stop nor target is reached.

        Returns:
            (daily_strategy_returns, trade_log, performance_metrics)
        """
        df = feature_df.copy().sort_index()
        n = len(df)
        if n < 10:
            return pd.Series(0.0, index=df.index), [], {}

        close_prices = df["close"].values
        open_prices = df["open"].values
        high_prices = df["high"].values
        low_prices = df["low"].values
        dates = df.index

        # Sentiment and momentum features
        sent_col = "news_sentiment_3d" if "news_sentiment_3d" in df.columns else "news_sentiment_daily"
        sent_vals = df[sent_col].values if sent_col in df.columns else np.zeros(n)
        surprise_vals = df["event_surprise_mean_today"].values if "event_surprise_mean_today" in df.columns else np.zeros(n)
        ret_lag = df["ret_5d_lag1"].values if "ret_5d_lag1" in df.columns else np.zeros(n)

        # Slippage multiplier: e.g. 1.5 bps = 0.00015
        slip_frac = (self.slippage_bps * 1e-4)

        trades: List[Dict[str, Any]] = []
        daily_returns = np.zeros(n)

        in_position = False
        pos_direction = 0  # +1 Long, -1 Short
        entry_price = 0.0
        entry_bar = 0
        shares = 0.0

        for t in range(lookback_window, n - 1):
            # 1. Check open position exit / intra-bar SL & TP
            if in_position:
                bars_held = t - entry_bar
                current_high = high_prices[t]
                current_low = low_prices[t]
                current_close = close_prices[t]

                exit_price = None
                exit_reason = None

                if pos_direction == 1:  # Long
                    sl_level = entry_price * (1.0 - stop_loss_pct)
                    tp_level = entry_price * (1.0 + take_profit_pct)

                    if current_low <= sl_level:
                        exit_price = sl_level * (1.0 - slip_frac)
                        exit_reason = "stop_loss"
                    elif current_high >= tp_level:
                        exit_price = tp_level * (1.0 - slip_frac)
                        exit_reason = "take_profit"
                    elif bars_held >= holding_period_bars:
                        exit_price = current_close * (1.0 - slip_frac)
                        exit_reason = "time_exit"

                elif pos_direction == -1:  # Short
                    sl_level = entry_price * (1.0 + stop_loss_pct)
                    tp_level = entry_price * (1.0 - take_profit_pct)

                    if current_high >= sl_level:
                        exit_price = sl_level * (1.0 + slip_frac)
                        exit_reason = "stop_loss"
                    elif current_low <= tp_level:
                        exit_price = tp_level * (1.0 + slip_frac)
                        exit_reason = "take_profit"
                    elif bars_held >= holding_period_bars:
                        exit_price = current_close * (1.0 + slip_frac)
                        exit_reason = "time_exit"

                if exit_price is not None:
                    # Calculate gross and net P&L
                    if pos_direction == 1:
                        gross_trade_pnl = (exit_price - entry_price) * shares
                        pct_ret = (exit_price - entry_price) / entry_price
                    else:
                        gross_trade_pnl = (entry_price - exit_price) * shares
                        pct_ret = (entry_price - exit_price) / entry_price

                    total_commission = 2.0 * (shares * self.commission_per_share)
                    net_trade_pnl = gross_trade_pnl - total_commission

                    trades.append({
                        "entry_date": dates[entry_bar],
                        "exit_date": dates[t],
                        "direction": "LONG" if pos_direction == 1 else "SHORT",
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(exit_price, 2),
                        "shares": round(shares, 2),
                        "pnl": round(gross_trade_pnl, 2),
                        "net_pnl": round(net_trade_pnl, 2),
                        "return_pct": round(pct_ret * 100.0, 3),
                        "exit_reason": exit_reason,
                        "bars_held": bars_held,
                    })

                    # Record return on portfolio
                    daily_returns[t] += net_trade_pnl / self.initial_capital

                    in_position = False
                    pos_direction = 0

            # 2. Check entry signal at bar close for next open/close execution
            if not in_position and t < n - 1:
                signal = 0
                # Long rule: positive sentiment or positive surprise with positive trend
                if (sent_vals[t] > sentiment_threshold and ret_lag[t] >= 0) or (surprise_vals[t] > 0.1):
                    signal = 1
                # Short rule: negative sentiment or negative surprise with negative trend
                elif (sent_vals[t] < -sentiment_threshold and ret_lag[t] < 0) or (surprise_vals[t] < -0.1):
                    signal = -1

                if signal != 0:
                    in_position = True
                    pos_direction = signal
                    entry_bar = t
                    raw_entry = close_prices[t]
                    entry_price = raw_entry * (1.0 + slip_frac) if signal == 1 else raw_entry * (1.0 - slip_frac)
                    shares = max(1.0, position_size_dollars / entry_price)

        returns_series = pd.Series(daily_returns, index=dates)
        metrics = evaluate_strategy_performance(returns_series, trades, self.initial_capital)
        return returns_series, trades, metrics

    def run_strategy_grid(
        self,
        feature_df: pd.DataFrame,
        stop_losses: Optional[List[float]] = None,
        take_profits: Optional[List[float]] = None,
        lookback_windows: Optional[List[int]] = None,
        sentiment_thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate full grid of strategy permutations across parameter ranges.

        Returns:
            pd.DataFrame ranking each parameter combination by Sharpe ratio and Net P&L.
        """
        sl_grid = stop_losses or [0.015, 0.025, 0.035]
        tp_grid = take_profits or [0.03, 0.05, 0.08]
        lb_grid = lookback_windows or [3, 5, 10]
        sent_grid = sentiment_thresholds or [0.03, 0.08]

        results = []
        combos = list(itertools.product(sl_grid, tp_grid, lb_grid, sent_grid))
        logger.info(f"Simulating strategy parameter grid: {len(combos)} permutations...")

        for sl, tp, lb, st in combos:
            params = {
                "stop_loss_pct": sl,
                "take_profit_pct": tp,
                "lookback_window": lb,
                "sentiment_threshold": st,
            }
            _, trades, metrics = self.run_backtest(
                feature_df=feature_df,
                stop_loss_pct=sl,
                take_profit_pct=tp,
                lookback_window=lb,
                sentiment_threshold=st
            )

            results.append({
                "strategy_name": f"MacroSent_SL{int(sl*100)}TP{int(tp*100)}_LB{lb}",
                "stop_loss_pct": sl,
                "take_profit_pct": tp,
                "lookback_window": lb,
                "sentiment_threshold": st,
                "gross_pnl": metrics.get("gross_pnl", 0.0),
                "net_pnl": metrics.get("net_pnl", 0.0),
                "total_return_pct": metrics.get("total_return_pct", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "sortino_ratio": metrics.get("sortino_ratio", 0.0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
                "win_rate_pct": metrics.get("win_rate_pct", 0.0),
                "profit_factor": metrics.get("profit_factor", 0.0),
                "total_trades": metrics.get("total_trades", 0),
            })

        results_df = pd.DataFrame(results)
        results_df.sort_values(by=["sharpe_ratio", "net_pnl"], ascending=False, inplace=True)
        return results_df
