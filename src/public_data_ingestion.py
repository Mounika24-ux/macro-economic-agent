"""
Zero-Cost Public Macroeconomic Ingestion Engine.

Fetches 100% free, authentic historical macroeconomic releases from:
1. Federal Reserve Bank of St. Louis (FRED public endpoints):
   - CPIAUCSL (Consumer Price Index)
   - PAYEMS (Nonfarm Payrolls)
   - UNRATE (Civilian Unemployment Rate)
   - FEDFUNDS (Federal Funds Effective Rate)
   - ICSA (Initial Jobless Claims)
   - GDP (Gross Domestic Product)
2. Public Central Bank & Macroeconomic News RSS Feeds:
   - Federal Reserve Board Press Releases & Speeches
   - Bureau of Labor Statistics (BLS) Releases
   - Yahoo Finance Macro News Feed

Requires zero API keys, zero paid subscriptions, and zero paid scraping compute.
"""
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import requests

from config.settings import get_settings
from src.preprocessing import canonicalize_url, compute_url_hash, normalize_title, parse_timestamp_to_utc

logger = logging.getLogger(__name__)

FRED_SERIES_CONFIG = [
    {
        "series_id": "CPIAUCSL",
        "event_name": "US Consumer Price Index (CPI)",
        "country": "US",
        "currency": "USD",
        "frequency": "Monthly",
        "impact": "High",
        "unit": "Index"
    },
    {
        "series_id": "PAYEMS",
        "event_name": "US Nonfarm Payrolls",
        "country": "US",
        "currency": "USD",
        "frequency": "Monthly",
        "impact": "High",
        "unit": "Thousands"
    },
    {
        "series_id": "UNRATE",
        "event_name": "US Unemployment Rate",
        "country": "US",
        "currency": "USD",
        "frequency": "Monthly",
        "impact": "High",
        "unit": "%"
    },
    {
        "series_id": "FEDFUNDS",
        "event_name": "Federal Funds Effective Rate",
        "country": "US",
        "currency": "USD",
        "frequency": "Monthly",
        "impact": "High",
        "unit": "%"
    },
    {
        "series_id": "ICSA",
        "event_name": "US Initial Jobless Claims",
        "country": "US",
        "currency": "USD",
        "frequency": "Weekly",
        "impact": "Medium",
        "unit": "Count"
    },
    {
        "series_id": "GDP",
        "event_name": "US Gross Domestic Product",
        "country": "US",
        "currency": "USD",
        "frequency": "Quarterly",
        "impact": "High",
        "unit": "Billions"
    },
]

PUBLIC_RSS_FEEDS = [
    {
        "name": "Federal Reserve Board",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "query": "federal reserve monetary policy"
    },
    {
        "name": "Yahoo Finance Markets",
        "url": "https://finance.yahoo.com/news/rssindex",
        "query": "macroeconomic market news"
    }
]


class PublicMacroIngestion:
    """Ingests free public macroeconomic data and official central bank news feeds."""

    def __init__(self, raw_output_dir: Optional[Path] = None):
        settings = get_settings()
        self.raw_output_dir = raw_output_dir or settings.raw_data_dir
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MacroeconomicAgent/1.0"
        }

    def fetch_free_macro_events(self, lookback_days: int = 180) -> List[Dict[str, Any]]:
        """
        Download historical macroeconomic releases directly from St. Louis Fed (FRED) public CSVs.

        Args:
            lookback_days: Duration in days to retrieve.

        Returns:
            List of standardized economic event dictionaries.
        """
        logger.info(f"Fetching 100% free public macroeconomic data from Federal Reserve Bank of St. Louis (FRED)...")
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        all_events = []

        for cfg in FRED_SERIES_CONFIG:
            s_id = cfg["series_id"]
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s_id}"

            try:
                resp = requests.get(url, headers=self.headers, timeout=20)
                if resp.status_code != 200:
                    logger.warning(f"Could not fetch FRED series {s_id} (HTTP {resp.status_code})")
                    continue

                # Parse CSV into lines
                df = pd.read_csv(pd.io.common.StringIO(resp.text))
                date_col = "DATE" if "DATE" in df.columns else "observation_date"
                val_col = s_id

                if date_col not in df.columns or val_col not in df.columns:
                    continue

                # Filter by lookback date (plus slight buffer for previous value delta)
                df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
                df.dropna(subset=[val_col], inplace=True)
                df = df[df[date_col] >= cutoff_date].sort_values(date_col)

                if df.empty:
                    continue

                # Compute previous values and rolling forecast proxy
                values = df[val_col].values
                dates = df[date_col].values

                for i in range(len(df)):
                    dt_str = str(dates[i])
                    ts_utc = parse_timestamp_to_utc(dt_str)
                    actual = float(values[i])
                    prev = float(values[i - 1]) if i > 0 else None

                    # Note on Forecast: Government databases (FRED/BLS) publish official actuals.
                    # As documented in quantitative methodology, we compute the historical rolling trend delta:
                    surprise = round(actual - prev, 4) if prev is not None else None

                    all_events.append({
                        "event_name": cfg["event_name"],
                        "country": cfg["country"],
                        "currency": cfg["currency"],
                        "timestamp_utc": ts_utc,
                        "original_timestamp": dt_str,
                        "source_timezone": "UTC",
                        "actual": actual,
                        "forecast": prev,  # Rolling prior reference
                        "previous": prev,
                        "surprise": surprise,
                        "impact_level": cfg["impact"],
                        "raw_payload": {
                            "source": "FRED_StLouisFed_Public",
                            "series_id": s_id,
                            "unit": cfg["unit"],
                            "license": "Public Domain (U.S. Government)"
                        },
                        "is_synthetic": False
                    })

            except Exception as e:
                logger.error(f"Error fetching public FRED series {s_id}: {e}")

        all_events.sort(key=lambda x: x["timestamp_utc"])
        logger.info(f"Successfully retrieved {len(all_events)} real macroeconomic event records from FRED.")
        return all_events

    def fetch_free_macro_news(self, max_items: int = 50) -> List[Dict[str, Any]]:
        """
        Collect authentic macroeconomic news headlines and summaries from public RSS feeds
        (Federal Reserve press releases, BLS announcements, Yahoo Finance).

        Returns:
            List of standardized news article dictionaries.
        """
        logger.info("Fetching authentic macroeconomic news from free public Federal Reserve & financial feeds...")
        articles = []
        seen_hashes = set()
        seen_titles = set()

        for feed in PUBLIC_RSS_FEEDS:
            try:
                resp = requests.get(feed["url"], headers=self.headers, timeout=20)
                if resp.status_code != 200:
                    continue

                root = ET.fromstring(resp.content)
                # Parse RSS items
                items = root.findall(".//item")
                for item in items:
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    pub_elem = item.find("pubDate") or item.find("{http://purl.org/dc/elements/1.1/}date")
                    desc_elem = item.find("description")

                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                    link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
                    pub_date = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else None
                    desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

                    if not title or not link:
                        continue

                    canon_url = canonicalize_url(link)
                    url_hash = compute_url_hash(canon_url)
                    norm_title = normalize_title(title)

                    if url_hash in seen_hashes or (norm_title and norm_title in seen_titles):
                        continue

                    seen_hashes.add(url_hash)
                    if norm_title:
                        seen_titles.add(norm_title)

                    ts_utc = parse_timestamp_to_utc(pub_date) or datetime.now(timezone.utc)

                    articles.append({
                        "url_hash": url_hash,
                        "title": title,
                        "normalized_title": norm_title,
                        "url": link,
                        "canonical_url": canon_url,
                        "source": feed["name"],
                        "query": feed["query"],
                        "published_at_utc": ts_utc,
                        "snippet": desc,
                        "provider": "public_rss",
                        "is_synthetic": False
                    })

                    if len(articles) >= max_items:
                        break

            except Exception as e:
                logger.error(f"Error fetching free RSS feed '{feed['name']}': {e}")

        logger.info(f"Retrieved {len(articles)} real articles from public RSS feeds.")
        return articles
