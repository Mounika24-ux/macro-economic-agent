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
        self.api_token = api_token if api_token is not None else settings.apify_api_token
        self.actor_id = actor_id if actor_id is not None else settings.apify_actor_id
        self.raw_output_dir = raw_output_dir or settings.raw_data_dir
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.apify.com/v2"

    @property
    def is_configured(self) -> bool:
        """Check if Apify credentials and actor ID are valid and configured."""
        if not self.api_token or not self.api_token.strip():
            return False
        if self.api_token.startswith("your_") or "token_here" in self.api_token:
            return False
        if not self.actor_id or not self.actor_id.strip():
            return False
        return True

    def test_connection(self) -> Dict[str, Any]:
        """
        Safely test API token authentication against Apify without triggering costly actor runs.

        Returns:
            Dict with 'status': 'OK' | 'ERROR', 'user_id', 'username', or 'error_message'.
        """
        if not self.is_configured:
            return {
                "status": "UNCONFIGURED",
                "error_message": "APIFY_API_TOKEN or APIFY_ACTOR_ID is not configured in .env"
            }

        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            resp = requests.get(f"{self.base_url}/users/me", headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                actor_info = self.inspect_actor()
                return {
                    "status": "OK",
                    "user_id": data.get("id"),
                    "username": data.get("username"),
                    "email": data.get("email"),
                    "actor_id": self.actor_id,
                    "actor_details": actor_info
                }
            elif resp.status_code == 401:
                return {
                    "status": "AUTH_FAILED",
                    "error_message": "Invalid Apify API token (HTTP 401 Unauthorized)."
                }
            else:
                return {
                    "status": "ERROR",
                    "error_message": f"Apify API returned HTTP {resp.status_code}: {resp.text}"
                }
        except Exception as e:
            return {
                "status": "CONNECTION_FAILED",
                "error_message": f"Failed to connect to Apify API: {str(e)}"
            }

    def inspect_actor(self) -> Dict[str, Any]:
        """
        Safely fetch metadata, description, and input schema for the configured actor (non-billing).
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            actor_slug = self.actor_id.replace("/", "~")
            resp = requests.get(f"{self.base_url}/acts/{actor_slug}", headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                return {
                    "id": data.get("id"),
                    "name": data.get("name"),
                    "title": data.get("title"),
                    "username": data.get("username"),
                    "description": data.get("description"),
                    "pricing_model": data.get("pricingModel", "FREE"),
                    "is_public": data.get("isPublic", True),
                    "stats": data.get("stats", {})
                }
            elif resp.status_code == 404:
                return {"error": f"Actor '{self.actor_id}' not found on Apify Store (HTTP 404)."}
            else:
                return {"error": f"Failed to inspect actor: HTTP {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def fetch_dataset_items(self, dataset_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch items directly from an existing Apify dataset by dataset ID."""
        if not self.api_token:
            return []
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            url = f"{self.base_url}/datasets/{dataset_id}/items?format=json&clean=true&limit={limit}"
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch Apify dataset {dataset_id}: {e}")
            return []

    def fetch_calendar_events(
        self,
        lookback_days: int = 180,
        country_filter: Optional[List[str]] = None,
        custom_input_payload: Optional[Dict[str, Any]] = None,
        timeout_secs: int = 120
    ) -> List[Dict[str, Any]]:
        """
        Execute the configured Apify actor, retrieve dataset items, and map to canonical schema.

        Args:
            lookback_days: Number of days back from today to fetch.
            country_filter: List of ISO countries or currencies (e.g. ['USD', 'US', 'EUR']).
            custom_input_payload: Optional overrides for the actor's run input.
            timeout_secs: Maximum seconds to wait for actor completion.

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

        # Multi-actor compatible input payload
        run_input = {
            "startDate": start_date,
            "endDate": end_date,
            "start_date": start_date,
            "end_date": end_date,
            "countries": country_filter or ["USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD"],
            "importance": ["High", "Medium", "1", "2", "3"],
            "timeZone": "UTC",
            "maxItems": 1000,
        }
        if custom_input_payload:
            run_input.update(custom_input_payload)

        logger.info(f"Triggering Apify actor '{self.actor_id}' for range {start_date} to {end_date}...")

        try:
            # Try importing official Apify client if available
            dataset_items = None
            try:
                from apify_client import ApifyClient
                client = ApifyClient(self.api_token)
                run = client.actor(self.actor_id).call(run_input=run_input, timeout_secs=timeout_secs)
                if run and "defaultDatasetId" in run:
                    dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
            except (ImportError, Exception) as client_err:
                logger.debug(f"SDK call failed or not installed, using direct REST API: {client_err}")

            # Fallback to direct REST API
            if dataset_items is None:
                headers = {
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json"
                }
                actor_slug = self.actor_id.replace("/", "~")
                run_url = f"{self.base_url}/acts/{actor_slug}/runs?waitForFinish={timeout_secs}"
                resp = requests.post(run_url, json=run_input, headers=headers, timeout=timeout_secs + 10)
                resp.raise_for_status()
                run_data = resp.json().get("data", {})
                dataset_id = run_data.get("defaultDatasetId")
                if not dataset_id:
                    logger.error("No defaultDatasetId found in Apify run response.")
                    return []

                dataset_url = f"{self.base_url}/datasets/{dataset_id}/items?format=json&clean=true&limit=1500"
                ds_resp = requests.get(dataset_url, headers=headers, timeout=60)
                ds_resp.raise_for_status()
                dataset_items = ds_resp.json()

            if not dataset_items:
                logger.warning("Apify actor run completed but returned 0 dataset items.")
                return []

            # Save raw payload for reproducibility & auditability
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            raw_file = self.raw_output_dir / f"apify_raw_events_{timestamp_str}.json"
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(dataset_items, f, indent=2, default=str)
            logger.info(f"Persisted {len(dataset_items)} raw items to {raw_file}")

            # Normalize and map output schema
            canonical_events = self.map_actor_items_to_canonical(dataset_items)
            logger.info(f"Successfully parsed and normalized {len(canonical_events)} economic events from Apify.")
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
                item.get("headline") or
                ""
            ).strip()

            if not event_name:
                continue

            # Country / Currency
            country = item.get("country") or item.get("country_code") or item.get("region") or ""
            currency = item.get("currency") or item.get("currency_code") or ""
            if not currency and country:
                country_to_curr = {
                    "US": "USD", "USA": "USD", "UNITED STATES": "USD",
                    "EA": "EUR", "EU": "EUR", "EURO ZONE": "EUR", "GERMANY": "EUR", "FRANCE": "EUR",
                    "UK": "GBP", "GB": "GBP", "UNITED KINGDOM": "GBP",
                    "JP": "JPY", "JAPAN": "JPY",
                    "CN": "CNY", "CHINA": "CNY",
                    "CA": "CAD", "CANADA": "CAD",
                    "AU": "AUD", "AUSTRALIA": "AUD"
                }
                currency = country_to_curr.get(str(country).upper(), "")

            # Timestamp & Timezone
            raw_ts = (
                item.get("date") or
                item.get("timestamp") or
                item.get("datetime") or
                item.get("time") or
                item.get("release_date")
            )
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
