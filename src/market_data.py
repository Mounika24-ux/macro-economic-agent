"""
Market Data Ingestion, CSV Loader, and Synthetic Price Engine.

Provides daily OHLCV bars via:
1. Public Yahoo Finance API (`yfinance`).
2. User-provided OHLCV CSV file importer.
3. Geometric Brownian Motion synthetic price generator (clearly labeled).
"""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.preprocessing import parse_numeric_value, parse_timestamp_to_utc

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """Loads and formats financial OHLCV price histories."""

    def __init__(self):
        self.settings = get_settings()

    def get_market_data(
        self,
        symbol: Optional[str] = None,
        lookback_days: int = 180,
        source: Optional[str] = None,
        csv_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV market data dataframe based on source configuration.

        Args:
            symbol: Ticker symbol (e.g. 'SPY', 'QQQ', 'TLT').
            lookback_days: Lookback duration in calendar days.
            source: 'yfinance', 'csv', or 'synthetic'.
            csv_path: Path to user-provided CSV file if source is 'csv'.

        Returns:
            pd.DataFrame indexed by `timestamp_utc` with columns [open, high, low, close, volume, is_synthetic].
        """
        sym = symbol or self.settings.default_symbol
        selected_source = source or self.settings.market_data_source

        if selected_source == "csv" and csv_path:
            return self.load_from_csv(csv_path, sym)

        if selected_source == "yfinance":
            try:
                df = self.fetch_yfinance(sym, lookback_days)
                if not df.empty and len(df) >= 10:
                    return df
                logger.warning(f"yfinance returned insufficient data for {sym}. Falling back to synthetic market generator.")
            except Exception as e:
                logger.warning(f"Failed to fetch market data from yfinance for {sym}: {e}. Falling back to synthetic generator.")

        # Synthetic Demonstration Fallback
        return self.generate_synthetic_ohlcv(symbol=sym, lookback_days=lookback_days)

    def fetch_yfinance(self, symbol: str, lookback_days: int = 180) -> pd.DataFrame:
        """Fetch daily OHLCV from Yahoo Finance."""
        import yfinance as yf

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days + 15)  # slight buffer for holidays

        logger.info(f"Fetching public market data for {symbol} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}) via yfinance...")
        ticker = yf.Ticker(symbol)
        raw_df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d")

        if raw_df.empty:
            return pd.DataFrame()

        # Format dataframe
        df = pd.DataFrame()
        raw_df.columns = [str(c).lower().replace(" ", "_") for c in raw_df.columns]

        # Convert index to UTC Timestamp
        idx_utc = []
        for dt in raw_df.index:
            parsed = parse_timestamp_to_utc(dt)
            idx_utc.append(parsed)

        df["open"] = raw_df["open"].astype(float)
        df["high"] = raw_df["high"].astype(float)
        df["low"] = raw_df["low"].astype(float)
        df["close"] = raw_df["close"].astype(float)
        df["volume"] = raw_df["volume"].astype(float)
        df["symbol"] = symbol
        df["is_synthetic"] = False
        df.index = pd.DatetimeIndex(idx_utc, name="timestamp_utc")
        df.sort_index(inplace=True)

        logger.info(f"Loaded {len(df)} bars for {symbol} via yfinance.")
        return df

    def load_from_csv(self, csv_path: Union[str, Path], symbol: str = "CUSTOM") -> pd.DataFrame:
        """
        Import user-provided OHLCV CSV file.
        Accepts standard column variations (Date, Timestamp, Open, High, Low, Close, Volume).
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"User CSV not found: {path}")

        raw_df = pd.read_csv(path)
        col_map = {c.lower().strip(): c for c in raw_df.columns}

        # Identify date/time column
        date_col = next((col_map[k] for k in ["timestamp_utc", "timestamp", "datetime", "date", "time"] if k in col_map), None)
        if not date_col:
            raise ValueError(f"CSV must contain a date/timestamp column. Available: {list(raw_df.columns)}")

        # Map OHLCV
        open_col = next((col_map[k] for k in ["open", "o"] if k in col_map), None)
        high_col = next((col_map[k] for k in ["high", "h"] if k in col_map), None)
        low_col = next((col_map[k] for k in ["low", "l"] if k in col_map), None)
        close_col = next((col_map[k] for k in ["close", "adj close", "adj_close", "c"] if k in col_map), None)
        vol_col = next((col_map[k] for k in ["volume", "vol", "v"] if k in col_map), None)

        if not all([open_col, high_col, low_col, close_col]):
            raise ValueError(f"CSV must contain Open, High, Low, and Close columns.")

        parsed_times = [parse_timestamp_to_utc(t) for t in raw_df[date_col]]
        df = pd.DataFrame(index=pd.DatetimeIndex(parsed_times, name="timestamp_utc"))

        df["open"] = [parse_numeric_value(x) for x in raw_df[open_col]]
        df["high"] = [parse_numeric_value(x) for x in raw_df[high_col]]
        df["low"] = [parse_numeric_value(x) for x in raw_df[low_col]]
        df["close"] = [parse_numeric_value(x) for x in raw_df[close_col]]
        df["volume"] = [parse_numeric_value(x) for x in raw_df[vol_col]] if vol_col else 1_000_000.0
        df["symbol"] = symbol
        df["is_synthetic"] = False

        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        df.sort_index(inplace=True)
        logger.info(f"Loaded {len(df)} bars from user CSV {path.name}")
        return df

    def generate_synthetic_ohlcv(
        self,
        symbol: str = "SPY",
        lookback_days: int = 180,
        initial_price: float = 500.0,
        annual_volatility: float = 0.16,
        annual_drift: float = 0.08,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate geometric Brownian motion daily OHLCV bars.
        Clearly tagged as `is_synthetic=True`.
        """
        np.random.seed(seed)
        end_dt = datetime.now(timezone.utc).replace(hour=20, minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=lookback_days)

        # Generate trading calendar business days
        date_range = pd.date_range(start=start_dt, end=end_dt, freq="B", tz=timezone.utc)
        n_days = len(date_range)
        dt = 1.0 / 252.0

        # Daily returns
        daily_drift = (annual_drift - 0.5 * (annual_volatility ** 2)) * dt
        daily_vol = annual_volatility * np.sqrt(dt)
        shocks = np.random.normal(daily_drift, daily_vol, n_days)

        # Price trajectory
        log_prices = np.log(initial_price) + np.cumsum(shocks)
        close_prices = np.exp(log_prices)

        opens, highs, lows, closes, volumes = [], [], [], [], []
        prev_close = initial_price

        for i, close in enumerate(close_prices):
            # Open near previous close with small overnight gap
            overnight_shock = float(np.random.normal(0, daily_vol * 0.3))
            bar_open = float(prev_close * (1 + overnight_shock))
            # Intraday excursion
            intraday_vol = float(np.random.uniform(0.005, 0.018))
            high = float(max(bar_open, close) * (1 + abs(np.random.normal(0, intraday_vol))))
            low = float(min(bar_open, close) * (1 - abs(np.random.normal(0, intraday_vol))))
            volume = float(np.random.lognormal(mean=17.5, sigma=0.3))  # ~40M-60M shares for SPY

            opens.append(round(bar_open, 2))
            highs.append(round(high, 2))
            lows.append(round(low, 2))
            closes.append(round(float(close), 2))
            volumes.append(round(volume, 0))
            prev_close = close

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "symbol": symbol,
            "is_synthetic": True,
        }, index=pd.DatetimeIndex(date_range, name="timestamp_utc"))

        logger.info(f"Generated {len(df)} synthetic OHLCV price bars for {symbol} (Demo Mode).")
        return df
