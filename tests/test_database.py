"""
Unit tests for database management, schemas, idempotency, and CSV operations.
"""
from datetime import datetime, timezone
import pandas as pd
import pytest
from src.database import EconomicEvent, MarketPrice, NewsArticle


def test_insert_economic_events_idempotency(mock_db, sample_events):
    # First insert
    count1 = mock_db.insert_economic_events(sample_events)
    assert count1 == 2

    # Second insert with identical events should not duplicate
    count2 = mock_db.insert_economic_events(sample_events)
    assert count2 == 0

    # Verify total rows in table
    with mock_db.session_scope() as session:
        rows = session.query(EconomicEvent).all()
        assert len(rows) == 2


def test_insert_news_articles_idempotency(mock_db, sample_news):
    # First insert
    count1 = mock_db.insert_news_articles(sample_news)
    assert count1 == 2

    # Second insert should not duplicate
    count2 = mock_db.insert_news_articles(sample_news)
    assert count2 == 0

    with mock_db.session_scope() as session:
        rows = session.query(NewsArticle).all()
        assert len(rows) == 2


def test_insert_market_prices(mock_db, sample_market_df):
    count1 = mock_db.insert_market_prices("SPY", sample_market_df)
    assert count1 == len(sample_market_df)

    # Re-inserting updates existing rows rather than creating duplicates
    count2 = mock_db.insert_market_prices("SPY", sample_market_df)
    assert count2 == 0

    with mock_db.session_scope() as session:
        rows = session.query(MarketPrice).filter(MarketPrice.symbol == "SPY").all()
        assert len(rows) == len(sample_market_df)


def test_csv_export_and_import(mock_db, sample_events, tmp_path):
    mock_db.insert_economic_events(sample_events)
    csv_file = tmp_path / "exported_events.csv"

    mock_db.export_table_to_csv("economic_events", csv_file)
    assert csv_file.exists()

    df = pd.read_csv(csv_file)
    assert len(df) == 2
    assert "US CPI YoY" in df["event_name"].values
