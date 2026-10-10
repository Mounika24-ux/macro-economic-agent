"""
Apify Economic Event Ingestion Client.

Fetches macroeconomic calendar events via the Apify platform, supports
custom actor output schema mapping, saves raw JSON payloads for reproducibility,
and standardizes outputs for database storage.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from config.settings import get_settings
from src.preprocessing import parse_numeric_value, parse_timestamp_to_utc

logger = logging.getLogger(__name__)


class ApifyCalendarIngestion:
    """Client for executing and fetching Apify economic calendar actors."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        actor_id: Optional[str] = None,
        raw_output_dir: Optional[Path] = None
    ):
        settings = get_settings()
        self.api_token = api_token or settings.apify_api_token
        self.actor_id = actor_id or settings.apify_actor_id
        self.raw_output_dir = raw_output_dir or settings.raw_data_dir
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.apify.com/v2"

    @property
    def is_configured(self) -> bool:
        """Check if Apify credentials and actor ID are present."""
        return bool(self.api_token and self.api_token.strip() and self.actor_id and self.actor_id.strip())

    def fetch_calendar_events(
        self,
        lookback_days: int = 180,
        country_filter: Optional[List[str]] = None,
        custom_input_payload: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute the configured Apify actor, retrieve items, and map to canonical schema.

        Args:
            lookback_days: Number of days back from today to fetch.
            country_filter: List of ISO countries or currencies (e.g. ['USD', 'US', 'EUR']).
            custom_input_payload: Optional overrides for the actor's run input.

        Returns:
            List of standardized economic event dictionaries.
        """
        if not self.is_configured:
            logger.warning(
                "Apify is not configured. Set APIFY_API_TOKEN and APIFY_ACTOR_ID in .env to enable live scraping. "
                "See README for actor configuration guide."
            )
            return []

        start_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Default run input structure compatible with standard calendar scrapers
        run_input = {
            "startDate": start_date,
            "endDate": end_date,
            "countries": country_filter or ["USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD"],
            "importance": ["High", "Medium"],
        }
        if custom_input_payload:
            run_input.update(custom_input_payload)

        logger.info(f"Triggering Apify actor '{self.actor_id}' from {start_date} to {end_date}...")

        try:
            # Try importing official Apify client if available
            try:
                from apify_client import ApifyClient
                client = ApifyClient(self.api_token)
                run = client.actor(self.actor_id).call(run_input=run_input, timeout_secs=120)
                if not run:
                    logger.error("Apify actor run returned empty response.")
                    return []
                dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
            except ImportError:
                # Fallback to direct HTTP API requests
                headers = {"Authorization": f"Bearer {self.api_token}"}
                actor_slug = self.actor_id.replace("/", "~")
                run_url = f"{self.base_url}/acts/{actor_slug}/runs?waitForFinish=120"
                resp = requests.post(run_url, json=run_input, headers=headers, timeout=130)
                resp.raise_for_status()
                run_data = resp.json().get("data", {})
                dataset_id = run_data.get("defaultDatasetId")
                if not dataset_id:
                    logger.error("No dataset ID found in Apify run response.")
                    return []
                dataset_url = f"{self.base_url}/datasets/{dataset_id}/items?format=json&clean=true"
                ds_resp = requests.get(dataset_url, headers=headers, timeout=60)
                ds_resp.raise_for_status()
                dataset_items = ds_resp.json()

            # Save raw payload for reproducibility
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            raw_file = self.raw_output_dir / f"apify_raw_events_{timestamp_str}.json"
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(dataset_items, f, indent=2, default=str)
            logger.info(f"Persisted {len(dataset_items)} raw items to {raw_file}")

            # Normalize and map output schema
            canonical_events = self.map_actor_items_to_canonical(dataset_items)
            logger.info(f"Successfully parsed and normalized {len(canonical_events)} economic events.")
            return canonical_events

        except Exception as e:
            logger.error(f"Failed to fetch Apify economic events: {str(e)}", exc_info=True)
            return []

    def map_actor_items_to_canonical(self, raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map diverse Apify actor JSON output schemas to the standardized event dictionary.
        Supports common keys: date, time, title, event, name, country, currency, actual,
        forecast, previous, consensus, estimate, impact, etc.
        """
        canonical_list = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            # Event Name
            event_name = (
                item.get("event") or
                item.get("event_name") or
                item.get("title") or
                item.get("name") or
                item.get("indicator") or
                ""
            ).strip()

            if not event_name:
                continue

            # Country / Currency
            country = item.get("country") or item.get("country_code") or item.get("region") or ""
            currency = item.get("currency") or item.get("currency_code") or ""
            if not currency and country:
                country_to_curr = {"US": "USD", "USA": "USD", "EA": "EUR", "EU": "EUR", "UK": "GBP", "GB": "GBP", "JP": "JPY"}
                currency = country_to_curr.get(str(country).upper(), "")

            # Timestamp & Timezone
            raw_ts = item.get("date") or item.get("timestamp") or item.get("time") or item.get("datetime")
            tz_str = item.get("timezone") or item.get("tz") or "UTC"
            ts_utc = parse_timestamp_to_utc(raw_ts, source_tz=tz_str)

            if not ts_utc:
                continue

            # Numeric values
            actual = parse_numeric_value(item.get("actual") or item.get("value") or item.get("current"))
            forecast = parse_numeric_value(item.get("forecast") or item.get("consensus") or item.get("estimate") or item.get("expected"))
            previous = parse_numeric_value(item.get("previous") or item.get("prior"))

            # Calculate surprise when actual and forecast are both available
            surprise = None
            if actual is not None and forecast is not None:
                surprise = round(actual - forecast, 6)

            impact = item.get("impact") or item.get("importance") or item.get("volatility") or "Medium"

            canonical_list.append({
                "event_name": event_name,
                "country": str(country).upper() if country else None,
                "currency": str(currency).upper() if currency else None,
                "timestamp_utc": ts_utc,
                "original_timestamp": str(raw_ts),
                "source_timezone": str(tz_str),
                "actual": actual,
                "forecast": forecast,
                "previous": previous,
                "surprise": surprise,
                "impact_level": str(impact).capitalize(),
                "raw_payload": item,
                "is_synthetic": False,
            })

        return canonical_list
