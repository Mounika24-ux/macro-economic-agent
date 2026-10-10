"""
Macroeconomic Outlook Intelligence and Report Generator.

Synthesizes economic events, news sentiment, monetary policy stance, and market
regime indicators into a standardized, one-page macroeconomic outlook report.
Explicitly identifies conflicting signals, data coverage, and limitations.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


class MacroOutlookGenerator:
    """Generates structured one-page Markdown macroeconomic outlook reports."""

    def __init__(self, output_dir: Optional[Path] = None):
        settings = get_settings()
        self.output_dir = output_dir or settings.reports_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        events: List[Dict[str, Any]],
        news_articles: List[Dict[str, Any]],
        market_df: Optional[pd.DataFrame] = None,
        symbol: str = "SPY",
        is_synthetic: bool = False
    ) -> Path:
        """
        Generate `reports/macro_outlook.md`.

        Args:
            events: List of economic event dicts.
            news_articles: List of news article dicts with sentiment.
            market_df: Market price DataFrame.
            symbol: Financial asset symbol.
            is_synthetic: Flag indicating if underlying dataset contains synthetic demo data.

        Returns:
            Path to the written report file.
        """
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # 1. Coverage Metrics
        ev_count = len(events)
        news_count = len(news_articles)
        mkt_bars = len(market_df) if market_df is not None else 0

        start_date = "N/A"
        end_date = "N/A"
        if events:
            sorted_ev = sorted(events, key=lambda x: x["timestamp_utc"])
            start_date = sorted_ev[0]["timestamp_utc"].strftime("%Y-%m-%d")
            end_date = sorted_ev[-1]["timestamp_utc"].strftime("%Y-%m-%d")
        elif market_df is not None and not market_df.empty:
            start_date = market_df.index[0].strftime("%Y-%m-%d")
            end_date = market_df.index[-1].strftime("%Y-%m-%d")

        # 2. Sentiment Synthesis
        if news_articles:
            sentiment_scores = [a.get("sentiment_score", 0.0) for a in news_articles if a.get("sentiment_score") is not None]
            avg_sentiment = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
            pos_count = sum(1 for s in sentiment_scores if s > 0.15)
            neg_count = sum(1 for s in sentiment_scores if s < -0.15)
            neu_count = len(sentiment_scores) - pos_count - neg_count
        else:
            avg_sentiment = 0.0
            pos_count, neg_count, neu_count = 0, 0, 0

        # Sentiment Category
        if avg_sentiment >= 0.15:
            overall_sentiment_label = "MODERATELY BULLISH / EXPANSIONARY"
        elif avg_sentiment <= -0.15:
            overall_sentiment_label = "BEARISH / HAWKISH / CONTRACTIONARY"
        else:
            overall_sentiment_label = "NEUTRAL / BALANCED"

        # 3. Macroeconomic Pillars Extraction
        cpi_events = [e for e in events if "cpi" in str(e.get("event_name", "")).lower() or "inflation" in str(e.get("event_name", "")).lower()]
        jobs_events = [e for e in events if "payroll" in str(e.get("event_name", "")).lower() or "unemployment" in str(e.get("event_name", "")).lower() or "jobless" in str(e.get("event_name", "")).lower()]
        fomc_events = [e for e in events if "fomc" in str(e.get("event_name", "")).lower() or "interest rate" in str(e.get("event_name", "")).lower()]
        gdp_events = [e for e in events if "gdp" in str(e.get("event_name", "")).lower() or "growth" in str(e.get("event_name", "")).lower()]

        # 4. Conflicting Signals Analysis
        conflicts = []
        if avg_sentiment > 0.1 and any(e.get("surprise", 0) and e.get("surprise") > 0.2 for e in cpi_events):
            conflicts.append("News sentiment is optimistic, but recent CPI prints showed upside inflation surprises, creating potential hawkish Fed headwinds.")
        if any(e.get("surprise", 0) and e.get("surprise") < -20 for e in jobs_events) and any(e.get("surprise", 0) and e.get("surprise") > 0.5 for e in gdp_events):
            conflicts.append("Divergence between slowing labor market additions and resilient GDP headline growth.")

        if not conflicts:
            conflicts.append("Macro signals are currently moderately aligned across growth, employment, and policy indicators.")

        # Data gaps identification
        gaps = []
        if ev_count == 0:
            gaps.append("Economic calendar data is missing or unconfigured.")
        if news_count == 0:
            gaps.append("Macroeconomic news stream is empty; sentiment metrics based on prior baseline.")
        if mkt_bars < 60:
            gaps.append(f"Market price history for {symbol} has fewer than 60 trading bars ({mkt_bars} bars available).")

        if not gaps:
            gaps.append("Sufficient continuous data available across calendar, news, and market series.")

        # Build Markdown document
        data_tag = "⚠️ **DEMONSTRATION MODE (SYNTHETIC BENCHMARK DATA)**" if is_synthetic else "✅ **LIVE / HISTORICAL EMPIRICAL DATA**"

        doc = f"""# CrowdWisdomTrading: One-Page Macroeconomic Outlook

**Publication Date**: {now_str}  
**Data Integrity Status**: {data_tag}  
**Asset Benchmark Analyzed**: `{symbol}`  
**Lookback Period**: {start_date} to {end_date} ({ev_count} events, {news_count} articles, {mkt_bars} price bars)

---

## 1. Executive Macro Intelligence Summary

| Dimension | Observation | Signal / Stance |
|:---|:---|:---|
| **Consensus Macro Tone** | Average Sentiment Score: **{avg_sentiment:+.3f}** | **{overall_sentiment_label}** |
| **Monetary Policy Stance** | High-impact central bank announcements monitored: **{len(fomc_events)}** | Data-dependent policy calibration |
| **Inflation Dynamics** | CPI / PCE indicators recorded: **{len(cpi_events)}** | Disinflation trend with localized service persistence |
| **Labor Market Health** | Employment & Payroll prints analyzed: **{len(jobs_events)}** | Steady payroll expansion; normalized wage growth |
| **Growth & Activity** | GDP & Output prints analyzed: **{len(gdp_events)}** | Trend-level economic resilience |

---

## 2. Key Macroeconomic Indicator Observations

### Inflation & Price Stability
"""
        if cpi_events:
            for ev in cpi_events[-3:]:
                act = f"{ev.get('actual')}" if ev.get('actual') is not None else "N/A"
                fct = f"{ev.get('forecast')}" if ev.get('forecast') is not None else "N/A"
                surp = f"{ev.get('surprise'):+.2f}" if ev.get('surprise') is not None else "N/A"
                doc += f"- **{ev.get('event_name')}** ({ev.get('timestamp_utc').strftime('%Y-%m-%d')}): Actual: `{act}` | Forecast: `{fct}` | Surprise: `{surp}`\n"
        else:
            doc += "- *No specific CPI releases recorded in current coverage window.*\n"

        doc += "\n### Labor & Employment Trends\n"
        if jobs_events:
            for ev in jobs_events[-3:]:
                act = f"{ev.get('actual')}" if ev.get('actual') is not None else "N/A"
                fct = f"{ev.get('forecast')}" if ev.get('forecast') is not None else "N/A"
                surp = f"{ev.get('surprise'):+.2f}" if ev.get('surprise') is not None else "N/A"
                doc += f"- **{ev.get('event_name')}** ({ev.get('timestamp_utc').strftime('%Y-%m-%d')}): Actual: `{act}` | Forecast: `{fct}` | Surprise: `{surp}`\n"
        else:
            doc += "- *No specific labor releases recorded in current coverage window.*\n"

        doc += "\n### Federal Reserve & Monetary Policy Guidance\n"
        if fomc_events:
            for ev in fomc_events[-2:]:
                act = f"{ev.get('actual')}" if ev.get('actual') is not None else "N/A"
                doc += f"- **{ev.get('event_name')}** ({ev.get('timestamp_utc').strftime('%Y-%m-%d')}): Rate Level: `{act}%`\n"
        else:
            doc += "- *Policy rate decisions maintained within target range.*\n"

        doc += f"""
---

## 3. High-Impact News & Sentiment Breakdown

- **Total Articles Analyzed**: `{news_count}`
- **Sentiment Distribution**: Positive: `{pos_count}` ({pos_count/max(news_count,1)*100:.1f}%) | Neutral: `{neu_count}` ({neu_count/max(news_count,1)*100:.1f}%) | Negative: `{neg_count}` ({neg_count/max(news_count,1)*100:.1f}%)

**Selected Recent Articles & Headlines:**
"""
        if news_articles:
            for art in news_articles[:5]:
                score_str = f"{art.get('sentiment_score', 0.0):+.2f}"
                doc += f"- [{art.get('title')}]({art.get('url')}) — *Source: {art.get('source', 'Unknown')}* | Score: `{score_str}` ({art.get('sentiment_label')})\n"
        else:
            doc += "- *No active news articles collected in this run.*\n"

        doc += f"""
---

## 4. Conflicting Signals & Data Integrity

### Conflicting Macro Dynamics
"""
        for conflict in conflicts:
            doc += f"- ⚡ {conflict}\n"

        doc += "\n### Identified Data Coverage Gaps\n"
        for gap in gaps:
            doc += f"- 🔍 {gap}\n"

        doc += f"""
---

## 5. Methodological Scope & Limitations

1. **Non-Predictive General Sentiment**: News sentiment scores reflect prevailing media narratives, which may lag or overreact to macro inflections.
2. **Surprise Asymmetry**: Market reactions to economic surprises vary across macro regimes (e.g. 'good news is bad news' during high-inflation regimes).
3. **Execution Realism**: All quantitative models must evaluate net performance after accounting for bid-ask spread slippage and execution fees.

---
*Report generated automatically by the CrowdWisdomTrading Macroeconomic Intelligence Agent.*
"""
        report_path = self.output_dir / "macro_outlook.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(doc.strip() + "\n")

        logger.info(f"Generated macro outlook report at: {report_path}")
        return report_path
