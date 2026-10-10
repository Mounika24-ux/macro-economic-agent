"""
Exa Search API Ingestion Client.

Performs neural and keyword search for macroeconomic research, central bank
speeches, and economic analysis.
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

EXA_TOPICS = [
    "Federal Reserve FOMC policy statement press conference summary",
    "US macroeconomic forecast inflation interest rates GDP outlook",
    "Consumer price index CPI labor market jobs report analysis",
]


class ExaNewsIngestion:
    """Ingestion client for Exa Search API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
        raw_output_dir: Optional[Path] = None
    ):
        settings = get_settings()
        self.api_key = api_key or settings.exa_api_key
        self.max_results = max_results or settings.exa_max_results
        self.raw_output_dir = raw_output_dir or settings.raw_data_dir
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint = "https://api.exa.ai/search"

    @property
    def is_configured(self) -> bool:
        """Check if Exa API key is provided."""
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

    def fetch_macro_news(
        self,
        topics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query Exa for macroeconomic articles and analysis.

        Args:
            topics: List of search queries.

        Returns:
            List of standardized news article dictionaries.
        """
        if not self.is_configured:
            logger.warning(
                "Exa API key is not configured. Set EXA_API_KEY in .env to enable live search."
            )
            return []

        search_topics = topics or EXA_TOPICS
        all_raw_results: List[Dict[str, Any]] = []
        canonical_articles: List[Dict[str, Any]] = []
        seen_url_hashes = set()
        seen_titles = set()

        exa_client = None
        try:
            from exa_py import Exa
            exa_client = Exa(api_key=self.api_key)
        except ImportError:
            pass

        for topic in search_topics:
            logger.info(f"Querying Exa for: '{topic}'...")
            try:
                results = []
                if exa_client is not None:
                    response = exa_client.search_and_contents(
                        query=topic,
                        num_results=self.max_results,
                        text={"max_characters": 1000},
                        highlights={"num_sentences": 3},
                    )
                    results = getattr(response, "results", [])
                else:
                    headers = {
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "query": topic,
                        "numResults": self.max_results,
                        "contents": {
                            "text": {"maxCharacters": 1000}
                        }
                    }
                    resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    results = data.get("results", [])

                for item in results:
                    # Support both SDK objects and raw dicts
                    if isinstance(item, dict):
                        url = item.get("url", "").strip()
                        title = item.get("title", "").strip()
                        published_date = item.get("publishedDate")
                        text_content = item.get("text") or item.get("snippet", "")
                        raw_dict = item
                    else:
                        url = getattr(item, "url", "").strip()
                        title = getattr(item, "title", "").strip()
                        published_date = getattr(item, "published_date", None)
                        text_content = getattr(item, "text", "") or ""
                        raw_dict = {
                            "url": url,
                            "title": title,
                            "publishedDate": published_date,
                            "text": text_content
                        }

                    if not url or not title:
                        continue

                    all_raw_results.append(raw_dict)

                    canon_url = canonicalize_url(url)
                    url_hash = compute_url_hash(canon_url)
                    norm_title = normalize_title(title)

                    if url_hash in seen_url_hashes or (norm_title and norm_title in seen_titles):
                        continue

                    seen_url_hashes.add(url_hash)
                    if norm_title:
                        seen_titles.add(norm_title)

                    pub_ts = parse_timestamp_to_utc(published_date) or datetime.now(timezone.utc)

                    canonical_articles.append({
                        "url_hash": url_hash,
                        "title": title,
                        "normalized_title": norm_title,
                        "url": url,
                        "canonical_url": canon_url,
                        "source": "ExaSearch",
                        "query": topic,
                        "published_at_utc": pub_ts,
                        "snippet": text_content,
                        "provider": "exa",
                        "is_synthetic": False,
                    })

            except Exception as e:
                logger.error(f"Error querying Exa for topic '{topic}': {e}")

        if all_raw_results:
            ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            raw_path = self.raw_output_dir / f"exa_raw_news_{ts_str}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(all_raw_results, f, indent=2, default=str)

        logger.info(f"Retrieved and deduplicated {len(canonical_articles)} articles via Exa.")
        return canonical_articles
