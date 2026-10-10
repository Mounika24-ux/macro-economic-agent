"""
Unit tests for Apify, Tavily, Exa ingestion clients and schema mappings with mocks.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from src.apify_ingestion import ApifyCalendarIngestion
from src.economic_calendar import EconomicCalendarCoordinator
from src.exa_ingestion import ExaNewsIngestion
from src.tavily_ingestion import TavilyNewsIngestion


def test_apify_schema_mapping():
    client = ApifyCalendarIngestion(api_token="test_token", actor_id="test/actor")

    raw_items = [
        {
            "event": "US Core CPI MoM",
            "country": "US",
            "currency": "USD",
            "date": "2026-03-12T13:30:00Z",
            "actual": "0.3%",
            "forecast": "0.2%",
            "previous": "0.4%",
            "importance": "High"
        },
        {
            "title": "ECB Interest Rate Decision",
            "region": "EU",
            "timestamp": "2026-03-15T12:45:00Z",
            "value": "3.75%",
            "estimate": "3.75%",
            "prior": "4.00%",
            "impact": "High"
        }
    ]

    canonical = client.map_actor_items_to_canonical(raw_items)
    assert len(canonical) == 2
    assert canonical[0]["event_name"] == "US Core CPI MoM"
    assert canonical[0]["actual"] == 0.3
    assert canonical[0]["forecast"] == 0.2
    assert canonical[0]["surprise"] == 0.1
    assert canonical[1]["event_name"] == "ECB Interest Rate Decision"
    assert canonical[1]["currency"] == "EUR"


def test_unconfigured_clients_handle_gracefully():
    # Empty token clients should return empty list and report not configured
    apify = ApifyCalendarIngestion(api_token=None)
    assert not apify.is_configured
    assert apify.fetch_calendar_events() == []

    tavily = TavilyNewsIngestion(api_key=None)
    assert not tavily.is_configured
    assert tavily.fetch_macro_news() == []

    exa = ExaNewsIngestion(api_key=None)
    assert not exa.is_configured
    assert exa.fetch_macro_news() == []


def test_economic_calendar_synthetic_generator():
    coord = EconomicCalendarCoordinator()
    events = coord.generate_synthetic_events(lookback_days=90, seed=42)

    assert len(events) > 10
    assert all(e["is_synthetic"] is True for e in events)
    assert all("timestamp_utc" in e for e in events)
    assert any("CPI" in e["event_name"] or "Payrolls" in e["event_name"] for e in events)
