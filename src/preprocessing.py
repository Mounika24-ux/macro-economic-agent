"""
Data Preprocessing, Cleaning, Normalization, and Parsing Module.

Handles:
- Robust timestamp parsing and timezone normalization to UTC.
- Numeric parsing for financial indicators (percentages, suffixes, parentheses).
- Text and title normalization for NLP and deduplication.
- URL canonicalization.
"""
import hashlib
import math
import re
from datetime import datetime, timezone
from typing import Any, Optional, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
import numpy as np
import pandas as pd


def parse_timestamp_to_utc(
    ts_input: Union[str, int, float, datetime, pd.Timestamp, None],
    source_tz: Optional[str] = None
) -> Optional[datetime]:
    """
    Convert an arbitrary timestamp input into a UTC-aware datetime object.
    Preserves UTC accuracy and correctly handles daylight savings when timezone is known.

    Args:
        ts_input: String, timestamp integer/float, datetime, or pandas Timestamp.
        source_tz: Timezone name if known (e.g. 'US/Eastern', 'UTC', 'Europe/London').

    Returns:
        A timezone-aware datetime object in UTC, or None if parsing fails.
    """
    if ts_input is None or (isinstance(ts_input, float) and math.isnan(ts_input)):
        return None

    try:
        # Handle UNIX timestamp in seconds or milliseconds
        if isinstance(ts_input, (int, float)):
            # If timestamp is likely in milliseconds (> 1e11), scale down
            ts_val = float(ts_input)
            if ts_val > 1e11:
                ts_val /= 1000.0
            dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
            return dt

        if isinstance(ts_input, pd.Timestamp):
            if ts_input.tzinfo is None:
                if source_tz:
                    dt = ts_input.tz_localize(source_tz).tz_convert(timezone.utc).to_pydatetime()
                else:
                    dt = ts_input.tz_localize(timezone.utc).to_pydatetime()
            else:
                dt = ts_input.tz_convert(timezone.utc).to_pydatetime()
            return dt

        if isinstance(ts_input, datetime):
            if ts_input.tzinfo is None:
                if source_tz:
                    # Localize with pandas for robust tz database handling
                    dt_series = pd.Series([ts_input]).dt.tz_localize(source_tz).dt.tz_convert("UTC")
                    return dt_series.iloc[0].to_pydatetime()
                else:
                    return ts_input.replace(tzinfo=timezone.utc)
            return ts_input.astimezone(timezone.utc)

        # Parse string formats
        ts_str = str(ts_input).strip()
        if not ts_str:
            return None

        # Try pandas to_datetime which handles RFC 2822, ISO 8601, and custom formats
        parsed = pd.to_datetime(ts_str, errors="coerce")
        if pd.isna(parsed):
            return None

        if parsed.tzinfo is None:
            if source_tz:
                parsed = parsed.tz_localize(source_tz).tz_convert("UTC")
            else:
                parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")

        return parsed.to_pydatetime()

    except Exception:
        return None


def parse_numeric_value(val: Any) -> Optional[float]:
    """
    Clean and convert financial text values into standard float representation.
    Handles percentages (e.g. '3.2%'), thousands/millions/billions ('250K', '1.5M', '2.1B'),
    parenthetical negatives '(0.4)', signed values '+0.2%', currency symbols, and commas.

    Args:
        val: Any raw input value.

    Returns:
        Float value or None if unparseable / missing.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if math.isnan(val):
            return None
        return float(val)

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "-", "n/a", "--", "nil"):
        return None

    # Check for parenthetical negative, e.g. '(1.25)' -> -1.25
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()

    # Remove currency symbols ($ € £ ¥ ₹), commas, and spaces
    s = re.sub(r"[\$,€£¥₹\s]", "", s)

    # Multiplier detection
    multiplier = 1.0
    if s.endswith("%"):
        s = s[:-1].strip()
        # Note: In macroeconomic figures, we keep 3.2% as 3.2 for surprise diffs or 0.032.
        # Keeping standard numerical value (e.g. 3.2 for 3.2%) maintains scale comparability
        # with forecast.
    elif s.upper().endswith("K"):
        multiplier = 1e3
        s = s[:-1].strip()
    elif s.upper().endswith("M"):
        multiplier = 1e6
        s = s[:-1].strip()
    elif s.upper().endswith("B"):
        multiplier = 1e9
        s = s[:-1].strip()
    elif s.upper().endswith("T"):
        multiplier = 1e12
        s = s[:-1].strip()
    elif s.lower().endswith("bps") or s.lower().endswith("bp"):
        multiplier = 0.01  # In basis points
        s = re.sub(r"(?i)bps?", "", s).strip()

    try:
        num = float(s) * multiplier
        return -num if is_negative else num
    except (ValueError, TypeError):
        # Extract first valid numeric subpattern if possible
        match = re.search(r"[-+]?\d*\.?\d+", s)
        if match:
            try:
                num = float(match.group(0)) * multiplier
                return -num if is_negative else num
            except Exception:
                return None
        return None


def normalize_title(title: Optional[str]) -> str:
    """
    Normalize article or event titles for fuzzy deduplication.
    Strips punctuation, collapses whitespace, and lowercases.
    """
    if not title:
        return ""
    # Strip HTML tags if any
    clean = re.sub(r"<[^>]+>", "", title)
    # Remove non-alphanumeric chars
    clean = re.sub(r"[^\w\s]", " ", clean.lower())
    # Collapse multiple whitespaces
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def canonicalize_url(url: Optional[str]) -> str:
    """
    Standardize a URL by removing common tracking query params and trailing slashes.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        # Drop tracking parameters (utm_*, ref, fbclid, gclid, etc.)
        tracking_prefixes = ("utm_", "ref", "fbclid", "gclid", "src", "source")
        qs = parse_qs(parsed.query, keep_blank_values=False)
        filtered_qs = {
            k: v for k, v in qs.items()
            if not any(k.lower().startswith(prefix) for prefix in tracking_prefixes)
        }
        new_query = urlencode(filtered_qs, doseq=True)
        # Normalize path
        new_path = parsed.path.rstrip("/")
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            new_path,
            parsed.params,
            new_query,
            ""  # drop fragment
        ))
        return normalized
    except Exception:
        return url.strip()


def compute_url_hash(url: str) -> str:
    """Compute a deterministic SHA-256 hash for a canonical URL."""
    canon = canonicalize_url(url)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()
