"""
Unit tests for data preprocessing, timestamp parsing, and numeric conversion.
"""
from datetime import datetime, timezone
import pytest
from src.preprocessing import (
    canonicalize_url,
    compute_url_hash,
    normalize_title,
    parse_numeric_value,
    parse_timestamp_to_utc,
)


def test_parse_timestamp_to_utc():
    # ISO 8601 UTC string
    dt1 = parse_timestamp_to_utc("2026-03-15T14:30:00Z")
    assert dt1 is not None
    assert dt1.year == 2026 and dt1.month == 3 and dt1.day == 15
    assert dt1.tzinfo is not None

    # Epoch timestamp
    dt2 = parse_timestamp_to_utc(1700000000)
    assert dt2 is not None
    assert dt2.tzinfo == timezone.utc

    # Source timezone conversion
    dt3 = parse_timestamp_to_utc("2026-01-15 08:30:00", source_tz="US/Eastern")
    assert dt3 is not None
    assert dt3.hour == 13 and dt3.minute == 30  # EST is UTC-5

    # None and empty
    assert parse_timestamp_to_utc(None) is None
    assert parse_timestamp_to_utc("") is None
    assert parse_timestamp_to_utc("invalid-date-format-xyz") is None


def test_parse_numeric_value():
    # Simple numbers
    assert parse_numeric_value(12.5) == 12.5
    assert parse_numeric_value("-4.2") == -4.2

    # Percentage
    assert parse_numeric_value("3.2%") == 3.2
    assert parse_numeric_value("+0.4%") == 0.4

    # Currency and commas
    assert parse_numeric_value("$1,250.50") == 1250.50
    assert parse_numeric_value("€ 500") == 500.0

    # Multipliers K, M, B
    assert parse_numeric_value("250K") == 250_000.0
    assert parse_numeric_value("1.5M") == 1_500_000.0
    assert parse_numeric_value("2.1B") == 2_100_000_000.0

    # Parenthetical negatives
    assert parse_numeric_value("(1.25)") == -1.25

    # Basis points
    assert parse_numeric_value("25bps") == 0.25

    # Missing / Malformed
    assert parse_numeric_value(None) is None
    assert parse_numeric_value("N/A") is None
    assert parse_numeric_value("--") is None
    assert parse_numeric_value("nil") is None


def test_normalize_title():
    t1 = "  Federal Reserve: Rate   Hike Decisions! 2026  "
    assert normalize_title(t1) == "federal reserve rate hike decisions 2026"
    assert normalize_title("") == ""
    assert normalize_title(None) == ""


def test_canonicalize_url_and_hash():
    url1 = "https://www.reuters.com/markets/macro-news?utm_source=twitter&utm_medium=social"
    url2 = "https://www.reuters.com/markets/macro-news"

    canon1 = canonicalize_url(url1)
    canon2 = canonicalize_url(url2)

    assert canon1 == canon2
    assert compute_url_hash(url1) == compute_url_hash(url2)
