"""
Macroeconomic and Financial Feature Engineering Pipeline.

Calculates time-aligned, strictly causal (no-lookahead) quantitative features:
- Event surprise metrics & previous value delta
- Time-to-next-event countdown (hours/days)
- Calendar & cyclical time features (day of week, month, day of month)
- Aggregated rolling news sentiment (1d, 3d, 7d, 14d) and news volume
- Market technical & volatility features (lagged returns, realized vol, high-low range)
- Forward target construction (strictly out-of-sample forward P&L and win rate)
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.preprocessing import parse_timestamp_to_utc

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Constructs time-series feature matrix from events, news, and market bars."""

    def __init__(self):
        pass

    def build_feature_matrix(
        self,
        market_df: pd.DataFrame,
        events: List[Dict],
        news_articles: List[Dict],
    ) -> pd.DataFrame:
        """
        Synthesize unified daily feature matrix.

        Args:
            market_df: OHLCV price DataFrame indexed by UTC datetime.
            events: List of economic event dicts.
            news_articles: List of news article dicts with sentiment_score.

        Returns:
            pd.DataFrame indexed by UTC datetime containing engineered features and targets.
        """
        if market_df.empty:
            raise ValueError("Cannot construct feature matrix with empty market data.")

        df = market_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            if df.index.tz is None:
                df.index = df.index.tz_localize(timezone.utc)
            else:
                df.index = df.index.tz_convert(timezone.utc)

        df.sort_index(inplace=True)

        # 1. Market Technicals (Strictly Lagged)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Lagged returns
        df["ret_1d_lag1"] = close.pct_change(1)
        df["ret_3d_lag1"] = close.pct_change(3)
        df["ret_5d_lag1"] = close.pct_change(5)
        df["ret_20d_lag1"] = close.pct_change(20)

        # Realized Volatility (annualized rolling 10d and 20d)
        df["realized_vol_10d"] = df["ret_1d_lag1"].rolling(10, min_periods=3).std() * np.sqrt(252)
        df["realized_vol_20d"] = df["ret_1d_lag1"].rolling(20, min_periods=5).std() * np.sqrt(252)

        # High-Low normalized range
        df["normalized_hl_range"] = (high - low) / close

        # Moving Average Distance
        df["dist_sma_20"] = (close - close.rolling(20, min_periods=5).mean()) / close.rolling(20, min_periods=5).mean()

        # 2. Calendar and Cyclical Features
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        # Cyclical sin/cos encodings
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5.0)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

        # 3. News Sentiment and Volume Aggregation
        news_df = self._process_news_timeline(news_articles, df.index)
        for col in news_df.columns:
            df[col] = news_df[col]

        # 4. Economic Event Surprises and Time-to-Event
        event_features = self._process_events_timeline(events, df.index)
        for col in event_features.columns:
            df[col] = event_features[col]

        # 5. Forward Target Generation (for Supervised Learning)
        # Note: Targets represent future outcomes; strictly separated during training/validation.
        # Forward 1-day return: (Close_{t+1} - Close_t) / Close_t
        df["target_pnl_1d"] = close.shift(-1) / close - 1.0
        # Forward 5-day return: (Close_{t+5} - Close_t) / Close_t
        df["target_pnl_5d"] = close.shift(-5) / close - 1.0

        # Binary targets (win = 1, loss = 0)
        df["target_win_1d"] = (df["target_pnl_1d"] > 0).astype(int)
        df["target_win_5d"] = (df["target_pnl_5d"] > 0).astype(int)

        # Forward targets for the final rows will be NaN naturally (future is unknown)
        # Fill missing technicals from warmup with backward fill or 0
        feature_cols = [c for c in df.columns if not c.startswith("target_") and c not in ["symbol", "is_synthetic"]]
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)

        logger.info(f"Built feature matrix with {len(df)} rows and {len(df.columns)} total columns.")
        return df

    def _process_news_timeline(self, news_articles: List[Dict], market_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Aggregate news sentiment scores and counts onto the daily market timeline without lookahead."""
        out = pd.DataFrame(index=market_index)

        if not news_articles:
            out["news_sentiment_daily"] = 0.0
            out["news_sentiment_3d"] = 0.0
            out["news_sentiment_7d"] = 0.0
            out["news_sentiment_14d"] = 0.0
            out["news_volume_1d"] = 0.0
            out["news_volume_7d"] = 0.0
            out["news_sentiment_momentum"] = 0.0
            return out

        # Parse news items into DataFrame
        records = []
        for art in news_articles:
            ts = art.get("published_at_utc")
            if not isinstance(ts, datetime):
                ts = parse_timestamp_to_utc(ts)
            if ts:
                score = art.get("sentiment_score", 0.0) or 0.0
                records.append({
                    "timestamp_utc": pd.Timestamp(ts),
                    "sentiment_score": float(score),
                    "count": 1
                })

        if not records:
            out["news_sentiment_daily"] = 0.0
            out["news_sentiment_3d"] = 0.0
            out["news_sentiment_7d"] = 0.0
            out["news_sentiment_14d"] = 0.0
            out["news_volume_1d"] = 0.0
            out["news_volume_7d"] = 0.0
            out["news_sentiment_momentum"] = 0.0
            return out

        news_raw = pd.DataFrame(records).set_index("timestamp_utc").sort_index()

        # Resample by calendar date (UTC)
        daily_sentiment = news_raw["sentiment_score"].resample("D").mean()
        daily_volume = news_raw["count"].resample("D").sum()

        combined_daily = pd.DataFrame({
            "daily_sentiment": daily_sentiment,
            "daily_volume": daily_volume
        }).reindex(pd.date_range(market_index.min().floor("D"), market_index.max().ceil("D"), freq="D", tz=timezone.utc))

        combined_daily["daily_sentiment"] = combined_daily["daily_sentiment"].fillna(0.0)
        combined_daily["daily_volume"] = combined_daily["daily_volume"].fillna(0.0)

        # Rolling sentiment metrics
        combined_daily["sentiment_3d"] = combined_daily["daily_sentiment"].rolling(3, min_periods=1).mean()
        combined_daily["sentiment_7d"] = combined_daily["daily_sentiment"].rolling(7, min_periods=1).mean()
        combined_daily["sentiment_14d"] = combined_daily["daily_sentiment"].rolling(14, min_periods=1).mean()
        combined_daily["volume_7d"] = combined_daily["daily_volume"].rolling(7, min_periods=1).sum()
        combined_daily["sentiment_momentum"] = combined_daily["sentiment_3d"] - combined_daily["sentiment_14d"]

        # Map to market trading dates by nearest preceding date (floor to day)
        date_keys = market_index.floor("D")
        out["news_sentiment_daily"] = combined_daily.loc[date_keys, "daily_sentiment"].values
        out["news_sentiment_3d"] = combined_daily.loc[date_keys, "sentiment_3d"].values
        out["news_sentiment_7d"] = combined_daily.loc[date_keys, "sentiment_7d"].values
        out["news_sentiment_14d"] = combined_daily.loc[date_keys, "sentiment_14d"].values
        out["news_volume_1d"] = combined_daily.loc[date_keys, "daily_volume"].values
        out["news_volume_7d"] = combined_daily.loc[date_keys, "volume_7d"].values
        out["news_sentiment_momentum"] = combined_daily.loc[date_keys, "sentiment_momentum"].values

        return out

    def _process_events_timeline(self, events: List[Dict], market_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Process economic event surprises, delta from previous, and time-to-next-event."""
        out = pd.DataFrame(index=market_index)

        # Default fallback series
        out["event_count_today"] = 0.0
        out["event_surprise_mean_today"] = 0.0
        out["event_surprise_sum_7d"] = 0.0
        out["event_previous_change_today"] = 0.0
        out["high_impact_event_today"] = 0.0
        out["hours_to_next_high_impact_event"] = 72.0  # default 3 days
        out["days_to_next_event"] = 3.0

        if not events:
            return out

        event_records = []
        for ev in events:
            ts = ev.get("timestamp_utc")
            if not isinstance(ts, datetime):
                ts = parse_timestamp_to_utc(ts)
            if ts:
                actual = ev.get("actual")
                forecast = ev.get("forecast")
                prev = ev.get("previous")

                surprise = ev.get("surprise")
                if surprise is None and actual is not None and forecast is not None:
                    surprise = actual - forecast

                prev_change = None
                if actual is not None and prev is not None:
                    prev_change = actual - prev

                is_high_impact = 1 if str(ev.get("impact_level", "")).lower() == "high" else 0

                event_records.append({
                    "timestamp_utc": pd.Timestamp(ts),
                    "surprise": float(surprise) if surprise is not None else 0.0,
                    "prev_change": float(prev_change) if prev_change is not None else 0.0,
                    "is_high_impact": is_high_impact,
                    "count": 1
                })

        if not event_records:
            return out

        ev_df = pd.DataFrame(event_records).sort_values("timestamp_utc")

        # Map daily event features
        all_event_times = ev_df["timestamp_utc"].tolist()
        high_impact_times = ev_df[ev_df["is_high_impact"] == 1]["timestamp_utc"].tolist()

        counts_today = []
        surprise_mean_today = []
        prev_change_today = []
        high_impact_today = []
        hours_to_next_hi = []
        days_to_next = []

        for bar_time in market_index:
            # Events occurred on the current bar day up to market bar time (strict causality)
            day_start = bar_time.floor("D")
            day_events = ev_df[(ev_df["timestamp_utc"] >= day_start) & (ev_df["timestamp_utc"] <= bar_time)]

            if not day_events.empty:
                counts_today.append(len(day_events))
                surprise_mean_today.append(day_events["surprise"].mean())
                prev_change_today.append(day_events["prev_change"].mean())
                high_impact_today.append(day_events["is_high_impact"].max())
            else:
                counts_today.append(0.0)
                surprise_mean_today.append(0.0)
                prev_change_today.append(0.0)
                high_impact_today.append(0.0)

            # Countdown to future events known on calendar
            future_events = [t for t in all_event_times if t > bar_time]
            if future_events:
                next_ev_time = future_events[0]
                delta_days = max(0.0, (next_ev_time - bar_time).total_seconds() / 86400.0)
                days_to_next.append(round(delta_days, 2))
            else:
                days_to_next.append(7.0)

            future_hi_events = [t for t in high_impact_times if t > bar_time]
            if future_hi_events:
                next_hi_time = future_hi_events[0]
                delta_hours = max(0.0, (next_hi_time - bar_time).total_seconds() / 3600.0)
                hours_to_next_hi.append(round(delta_hours, 1))
            else:
                hours_to_next_hi.append(168.0)  # 1 week

        out["event_count_today"] = counts_today
        out["event_surprise_mean_today"] = surprise_mean_today
        out["event_surprise_sum_7d"] = out["event_surprise_mean_today"].rolling(7, min_periods=1).sum()
        out["event_previous_change_today"] = prev_change_today
        out["high_impact_event_today"] = high_impact_today
        out["hours_to_next_high_impact_event"] = hours_to_next_hi
        out["days_to_next_event"] = days_to_next

        return out
