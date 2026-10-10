"""
Shared Pytest Fixtures and Mock Data.
"""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from src.database import DatabaseManager


@pytest.fixture
def mock_db(tmp_path):
    """Provide an isolated SQLite database in a temporary directory."""
    db_file = tmp_path / "test_macro.db"
    db_url = f"sqlite:///{db_file}"
    manager = DatabaseManager(db_url=db_url)
    return manager


@pytest.fixture
def sample_events():
    """Sample macroeconomic events with various formats and surprises."""
    return [
        {
            "event_name": "US CPI YoY",
            "country": "US",
            "currency": "USD",
            "timestamp_utc": datetime(2026, 1, 15, 13, 30, tzinfo=timezone.utc),
            "original_timestamp": "2026-01-15 08:30:00 EST",
            "source_timezone": "US/Eastern",
            "actual": 3.4,
            "forecast": 3.2,
            "previous": 3.1,
            "surprise": 0.2,
            "impact_level": "High",
            "raw_payload": {"source": "test"},
            "is_synthetic": False,
        },
        {
            "event_name": "US Nonfarm Payrolls",
            "country": "US",
            "currency": "USD",
            "timestamp_utc": datetime(2026, 2, 6, 13, 30, tzinfo=timezone.utc),
            "original_timestamp": "2026-02-06 08:30:00 EST",
            "source_timezone": "US/Eastern",
            "actual": 215.0,
            "forecast": 180.0,
            "previous": 175.0,
            "surprise": 35.0,
            "impact_level": "High",
            "raw_payload": {"source": "test"},
            "is_synthetic": False,
        },
    ]


@pytest.fixture
def sample_news():
    """Sample news articles with sentiment."""
    return [
        {
            "url_hash": "hash_1111",
            "title": "Federal Reserve signals steady rates amid cooling inflation",
            "normalized_title": "federal reserve signals steady rates amid cooling inflation",
            "url": "https://example.com/fed-cooling",
            "source": "Reuters",
            "query": "federal reserve",
            "published_at_utc": datetime(2026, 1, 14, 18, 0, tzinfo=timezone.utc),
            "snippet": "Inflation is moderating and policymakers are considering rate cuts.",
            "sentiment_score": 0.45,
            "sentiment_label": "positive",
            "provider": "tavily",
            "is_synthetic": False,
        },
        {
            "url_hash": "hash_2222",
            "title": "Manufacturing downturn deepens as orders plummet",
            "normalized_title": "manufacturing downturn deepens as orders plummet",
            "url": "https://example.com/pmi-drop",
            "source": "Bloomberg",
            "query": "manufacturing",
            "published_at_utc": datetime(2026, 1, 20, 10, 0, tzinfo=timezone.utc),
            "snippet": "Severe slump in industrial demand sparks recession concerns.",
            "sentiment_score": -0.65,
            "sentiment_label": "negative",
            "provider": "tavily",
            "is_synthetic": False,
        },
    ]


@pytest.fixture
def sample_market_df():
    """Generate 60 days of synthetic daily OHLCV bars for testing."""
    dates = pd.date_range(start="2026-01-01", periods=60, freq="B", tz=timezone.utc)
    np.random.seed(42)
    prices = 450.0 + np.cumsum(np.random.normal(0.2, 2.0, len(dates)))

    df = pd.DataFrame({
        "open": prices * 0.998,
        "high": prices * 1.012,
        "low": prices * 0.988,
        "close": prices,
        "volume": np.random.uniform(40_000_000, 60_000_000, len(dates)),
        "symbol": "SPY",
        "is_synthetic": False
    }, index=pd.DatetimeIndex(dates, name="timestamp_utc"))

    return df
