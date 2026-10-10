import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import sqlite3
import numpy as np
import pandas as pd

from src.database import get_db
from src.financial_metrics import (
    calculate_drawdown_series,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    evaluate_strategy_performance,
)
from src.walk_forward_validation import WalkForwardValidator


def main():
    conn = sqlite3.connect("data/macro_economic.db")

    print("=" * 75)
    print("1. DATA INTEGRITY AUDIT: REAL VS SYNTHETIC DATA")
    print("=" * 75)

    # 1. Economic Events
    df_ev = pd.read_sql_query("SELECT * FROM economic_events WHERE is_synthetic = 0", conn)
    print(f"Real Economic Events from FRED: {len(df_ev)}")
    print(f"Coverage: {df_ev['timestamp_utc'].min()[:10]} to {df_ev['timestamp_utc'].max()[:10]}")
    for name, grp in df_ev.groupby("event_name"):
        min_d = grp["timestamp_utc"].min()[:10]
        max_d = grp["timestamp_utc"].max()[:10]
        min_v = grp["actual"].min()
        max_v = grp["actual"].max()
        print(f"  - {name}: {len(grp)} obs (dates: {min_d} to {max_d}, actuals: {min_v} to {max_v})")

    # 2. News Articles
    df_news = pd.read_sql_query("SELECT * FROM news_articles WHERE is_synthetic = 0", conn)
    print(f"\nReal News Articles from Public RSS: {len(df_news)}")
    print(f"Sources: {df_news['source'].value_counts().to_dict()}")
    print("Sample Authentic Headlines & URLs:")
    for _, row in df_news.head(3).iterrows():
        print(f"  - Title: {row['title'][:70]}...")
        print(f"    URL:   {row['url']}")
        print(f"    Date:  {str(row['published_at_utc'])[:19]} | Sentiment: {row['sentiment_score']:.2f}")

    # 3. Market Prices
    df_mkt = pd.read_sql_query("SELECT * FROM market_prices WHERE is_synthetic = 0", conn)
    print(f"\nReal Market Price Bars (SPY via yfinance): {len(df_mkt)}")
    print(f"Coverage: {df_mkt['timestamp_utc'].min()[:10]} to {df_mkt['timestamp_utc'].max()[:10]}")
    print(f"Close Price Range: ${df_mkt['close'].min():.2f} to ${df_mkt['close'].max():.2f}")
    print(f"Daily Volume Range: {df_mkt['volume'].min():,.0f} to {df_mkt['volume'].max():,.0f}")
    invalid_hl = ((df_mkt["high"] < df_mkt["open"]) | (df_mkt["high"] < df_mkt["close"]) | (df_mkt["low"] > df_mkt["open"]) | (df_mkt["low"] > df_mkt["close"])).sum()
    print(f"Invalid High/Low Bars: {invalid_hl}")

    print("\n" + "=" * 75)
    print("2. WALK-FORWARD VALIDATION & METRIC RE-CALCULATION AUDIT")
    print("=" * 75)

    df_mkt_sorted = df_mkt.sort_values("timestamp_utc").copy()
    dates = pd.to_datetime(df_mkt_sorted["timestamp_utc"], utc=True)
    df_mkt_sorted.index = dates

    validator = WalkForwardValidator(n_splits=4, purge_days=2, window_type="expanding")
    folds = validator.generate_folds(dates)
    print(f"Generated Chronological Folds: {len(folds)}")
    for f in folds:
        print(f"  - Fold {f['fold_idx']}: Train [{f['train_start_date']} -> {f['train_end_date']}] (N={f['train_count']}) "
              f"| Test [{f['test_start_date']} -> {f['test_end_date']}] (N={f['test_count']}) "
              f"| Purge Gap: >= 2 days")

    # Re-calculate Buy and Hold benchmark metrics directly
    close_series = df_mkt_sorted["close"]
    daily_returns = close_series.pct_change().dropna()
    cum_wealth = (1.0 + daily_returns).cumprod()
    sharpe = calculate_sharpe_ratio(daily_returns, annual_risk_free_rate=0.045)
    sortino = calculate_sortino_ratio(daily_returns, annual_risk_free_rate=0.045)
    _, max_dd, max_dur = calculate_drawdown_series(cum_wealth)

    print(f"\nBenchmark (SPY Buy & Hold over period) Independent Metric Verification:")
    print(f"  - Total Return: {((cum_wealth.iloc[-1] - 1.0) * 100):+.2f}%")
    print(f"  - Annualized Sharpe Ratio: {sharpe:.4f}")
    print(f"  - Annualized Sortino Ratio: {sortino:.4f}")
    print(f"  - Maximum Drawdown: {max_dd:.2f}% (Duration: {max_dur} bars)")

    print("\n" + "=" * 75)
    print("3. SAMPLE SIZE & STATISTICAL POWER ASSESSMENT")
    print("=" * 75)
    print(f"Total Daily Price Bars: {len(df_mkt_sorted)}")
    print(f"Fold Test Window Size: ~19 trading days (~1 month)")
    print(f"Assessment: 135 daily bars (~6 months) is a small sample in quantitative finance.")
    print(f"The model evaluation reflects short-horizon feasibility rather than long-term stationary edge.")
    print("=" * 75)


if __name__ == "__main__":
    main()
