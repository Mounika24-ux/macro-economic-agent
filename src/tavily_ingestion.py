"""
Tavily Macroeconomic News Ingestion Client.

Searches for high-impact macroeconomic news covering CPI/Inflation,
FOMC/Federal Reserve, GDP, Labor/Employment, and Central Bank Policy.
Deduplicates results by canonical URL and normalized title.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from config.settings import get_settings
from src.preprocessing import canonicalize_url, compute_url_hash, normalize_title, parse_timestamp_to_utc

logger = logging.getLogger(__name__)

MACRO_TOPICS = [
    "US inflation CPI consumer price index data release",
    "Federal Reserve FOMC interest rate monetary policy decision",
    "US nonfarm payrolls unemployment labor market report",
    "US GDP economic growth quarterly report",
    "Treasury yields interest rates bond market inflation outlook",
    "US consumer spending retail sales economic health",
]


class TavilyNewsIngestion:
    """Ingestion client for Tavily AI Search API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results_per_query: int = 10,
        raw_output_dir: Optional[Path] = None
    ):
        settings = get_settings()
        self.api_key = api_key or settings.tavily_api_key
        self.max_results_per_query = max_results_per_query or settings.tavily_max_results_per_topic
        self.raw_output_dir = raw_output_dir or settings.raw_data_dir
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint = "https://api.tavily.com/search"

    @property
    def is_configured(self) -> bool:
        """Check if Tavily API key is provided."""
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

    def fetch_macro_news(
        self,
        topics: Optional[List[str]] = None,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Execute search across macroeconomic topics, deduplicate and format articles.

        Args:
            topics: List of topic queries (defaults to standard macro topics).
            days_back: Lookback window for search recency.

        Returns:
            List of standardized news article dictionaries.
        """
        if not self.is_configured:
            logger.warning(
                "Tavily API key is not configured. Set TAVILY_API_KEY in .env to enable live search."
            )
            return []

        search_topics = topics or MACRO_TOPICS
        all_raw_results: List[Dict[str, Any]] = []
        canonical_articles: List[Dict[str, Any]] = []
        seen_url_hashes = set()
        seen_titles = set()

        # Attempt to import official SDK if installed
        tavily_client = None
        try:
            from tavily import TavilyClient
            tavily_client = TavilyClient(api_key=self.api_key)
        except ImportError:
            pass

        for topic in search_topics:
            logger.info(f"Querying Tavily for: '{topic}'...")
            try:
                results = []
                if tavily_client is not None:
                    response = tavily_client.search(
                        query=topic,
                        search_depth="advanced",
                        include_domains=[],
                        exclude_domains=[],
                        max_results=self.max_results_per_query,
                    )
                    results = response.get("results", [])
                else:
                    payload = {
                        "api_key": self.api_key,
                        "query": topic,
                        "search_depth": "advanced",
                        "max_results": self.max_results_per_query,
                        "include_answer": False,
                    }
                    resp = requests.post(self.endpoint, json=payload, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    results = data.get("results", [])

                all_raw_results.extend(results)

                for item in results:
                    url = (item.get("url") or "").strip()
                    title = (item.get("title") or "").strip()
                    if not url or not title:
                        continue

                    canon_url = canonicalize_url(url)
                    url_hash = compute_url_hash(canon_url)
                    norm_title = normalize_title(title)

                    if url_hash in seen_url_hashes or (norm_title and norm_title in seen_titles):
                        continue

                    seen_url_hashes.add(url_hash)
                    if norm_title:
                        seen_titles.add(norm_title)

                    raw_pub = item.get("published_date") or item.get("published_at") or item.get("date")
                    pub_ts = parse_timestamp_to_utc(raw_pub) or datetime.now(timezone.utc)

                    canonical_articles.append({
                        "url_hash": url_hash,
                        "title": title,
                        "normalized_title": norm_title,
                        "url": url,
                        "canonical_url": canon_url,
                        "source": item.get("source") or item.get("domain") or "TavilySearch",
                        "query": topic,
                        "published_at_utc": pub_ts,
                        "snippet": item.get("content") or item.get("snippet") or "",
                        "provider": "tavily",
                        "is_synthetic": False,
                    })

            except Exception as e:
                logger.error(f"Error querying Tavily for topic '{topic}': {e}")

        # Save raw responses
        if all_raw_results:
            ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            raw_path = self.raw_output_dir / f"tavily_raw_news_{ts_str}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(all_raw_results, f, indent=2, default=str)

        logger.info(f"Retrieved and deduplicated {len(canonical_articles)} articles via Tavily.")
        return canonical_articles
