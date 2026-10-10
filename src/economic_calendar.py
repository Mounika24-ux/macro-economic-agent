"""
Economic Calendar Coordinator and Fallback Ingestion.

Coordinates fetching economic events from Apify, and provides a clearly labeled
synthetic macroeconomic calendar generator for demonstration and testing when
external APIs are unconfigured.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from config.settings import get_settings
from src.apify_ingestion import ApifyCalendarIngestion

logger = logging.getLogger(__name__)

# Standard major economic indicators
CALENDAR_INDICATORS = [
    {"name": "US CPI YoY", "country": "US", "currency": "USD", "mean": 3.1, "std": 0.4, "impact": "High"},
    {"name": "US Core CPI MoM", "country": "US", "currency": "USD", "mean": 0.3, "std": 0.1, "impact": "High"},
    {"name": "US Nonfarm Payrolls", "country": "US", "currency": "USD", "mean": 180.0, "std": 50.0, "impact": "High"}, # in Thousands
    {"name": "US Unemployment Rate", "country": "US", "currency": "USD", "mean": 3.9, "std": 0.2, "impact": "High"},
    {"name": "FOMC Interest Rate Decision", "country": "US", "currency": "USD", "mean": 5.25, "std": 0.25, "impact": "High"},
    {"name": "US GDP QoQ Annualized", "country": "US", "currency": "USD", "mean": 2.5, "std": 0.8, "impact": "High"},
    {"name": "US Retail Sales MoM", "country": "US", "currency": "USD", "mean": 0.4, "std": 0.3, "impact": "Medium"},
    {"name": "US ISM Manufacturing PMI", "country": "US", "currency": "USD", "mean": 49.5, "std": 1.5, "impact": "Medium"},
    {"name": "US Initial Jobless Claims", "country": "US", "currency": "USD", "mean": 220.0, "std": 15.0, "impact": "Medium"}, # in Thousands
    {"name": "US Consumer Confidence", "country": "US", "currency": "USD", "mean": 102.0, "std": 4.0, "impact": "Medium"},
]


class EconomicCalendarCoordinator:
    """Coordinates economic event retrieval across live, public-free, and synthetic modes."""

    def __init__(self, apify_client: Optional[ApifyCalendarIngestion] = None):
        self.apify_client = apify_client or ApifyCalendarIngestion()
        self.settings = get_settings()

    def get_events(
        self,
        lookback_days: int = 180,
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve economic calendar events based on configuration mode.

        Args:
            lookback_days: Number of days of historical events.
            mode: 'live', 'fallback' (free public FRED data), or 'demo' (synthetic data).

        Returns:
            List of economic event dictionaries.
        """
        selected_mode = mode or self.settings.pipeline_mode

        if selected_mode == "live" and self.apify_client.is_configured:
            logger.info("Executing live Apify economic calendar scrape...")
            events = self.apify_client.fetch_calendar_events(lookback_days=lookback_days)
            if events:
                return events
            logger.warning("Live Apify scraping returned 0 events. Switching to public FRED fallback.")

        if selected_mode in ["fallback", "public_free", "live"]:
            try:
                from src.public_data_ingestion import PublicMacroIngestion
                public_ingest = PublicMacroIngestion()
                events = public_ingest.fetch_free_macro_events(lookback_days=lookback_days)
                if events:
                    return events
            except Exception as e:
                logger.warning(f"Could not load public FRED events: {e}. Using synthetic demo generator.")

        # Demonstration mode with explicitly labeled synthetic data
        logger.info(f"Generating explicitly labeled synthetic economic events for {lookback_days} days (Demo Mode)...")
        return self.generate_synthetic_events(lookback_days=lookback_days)

    def generate_synthetic_events(self, lookback_days: int = 180, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Generate realistic, reproducible macroeconomic indicator releases.
        All records are explicitly labeled with `is_synthetic=True`.
        """
        np.random.seed(seed)
        end_dt = datetime.now(timezone.utc).replace(hour=13, minute=30, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=lookback_days)

        events: List[Dict[str, Any]] = []

        # Iterate day by day
        curr_dt = start_dt
        while curr_dt <= end_dt:
            # Skip weekends for releases
            if curr_dt.weekday() < 5:
                # 35% probability of an economic release on a given weekday
                if np.random.rand() < 0.45:
                    indicator = np.random.choice(CALENDAR_INDICATORS)
                    mean_val = indicator["mean"]
                    std_val = indicator["std"]

                    forecast = round(float(np.random.normal(mean_val, std_val)), 2)
                    # Surprise component
                    surprise_delta = float(np.random.normal(0, std_val * 0.4))
                    actual = round(forecast + surprise_delta, 2)
                    previous = round(float(np.random.normal(mean_val, std_val)), 2)
                    surprise = round(actual - forecast, 4)

                    # Event release time (typically 8:30 AM ET -> 13:30 UTC or 14:00 UTC)
                    event_hour = 13 if np.random.rand() > 0.3 else 14
                    event_min = 30 if event_hour == 13 else 0
                    release_dt = curr_dt.replace(hour=event_hour, minute=event_min, second=0)

                    events.append({
                        "event_name": indicator["name"],
                        "country": indicator["country"],
                        "currency": indicator["currency"],
                        "timestamp_utc": release_dt,
                        "original_timestamp": release_dt.strftime("%Y-%m-%d %H:%M:%S EST"),
                        "source_timezone": "US/Eastern",
                        "actual": actual,
                        "forecast": forecast,
                        "previous": previous,
                        "surprise": surprise,
                        "impact_level": indicator["impact"],
                        "raw_payload": {
                            "source": "synthetic_generator",
                            "note": "EXPLICITLY LABELED DEMO DATA",
                            "indicator": indicator["name"]
                        },
                        "is_synthetic": True
                    })

            curr_dt += timedelta(days=1)

        # Sort chronologically
        events.sort(key=lambda x: x["timestamp_utc"])
        logger.info(f"Generated {len(events)} synthetic calendar events.")
        return events
