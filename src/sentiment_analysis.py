"""
Macroeconomic and Financial Lexicon Sentiment Analyzer.

Implements transparent, reproducible sentiment scoring combining VADER sentiment
with financial and macroeconomic domain-specific lexicon adjustments.
Distinguishes general news tone, macroeconomic regime implications, and expected
market impact.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Domain-specific financial sentiment lexicons
MACRO_HAWKISH_TERMS = {
    "rate hike": -0.5, "hiking rates": -0.5, "tightening": -0.4, "inflation surges": -0.6,
    "hotter than expected": -0.5, "sticky inflation": -0.6, "wage pressure": -0.4,
    "hawkish": -0.5, "overheating": -0.5, "soaring prices": -0.6, "tariff shock": -0.5
}

MACRO_DOVISH_BULLISH_TERMS = {
    "rate cut": 0.6, "cutting rates": 0.6, "easing": 0.5, "cooling inflation": 0.6,
    "soft landing": 0.7, "disinflation": 0.6, "dovish": 0.5, "rebound": 0.5,
    "economic resilience": 0.6, "strong labor": 0.4, "gdp beats": 0.6, "optimism": 0.5
}

MACRO_RECESSIONARY_TERMS = {
    "recession": -0.8, "stagflation": -0.9, "layoffs": -0.6, "job losses": -0.7,
    "banking crisis": -0.9, "debt ceiling": -0.6, "default risk": -0.8, "downturn": -0.7,
    "contraction": -0.6, "weak manufacturing": -0.5, "slump": -0.6
}


class MacroSentimentAnalyzer:
    """Calculates sentiment scores and macroeconomic interpretations from financial text."""

    def __init__(self):
        self._vader_analyzer = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader_analyzer = SentimentIntensityAnalyzer()
            # Augment VADER lexicon with financial terms
            self._vader_analyzer.lexicon.update({
                "bullish": 2.5, "bearish": -2.5, "rally": 2.0, "plummet": -2.5,
                "surge": 2.0, "slump": -2.0, "hawkish": -1.5, "dovish": 1.5,
                "recession": -3.0, "disinflation": 1.8, "stagflation": -3.0
            })
        except ImportError:
            logger.info("vaderSentiment not installed. Utilizing internal financial lexicon tokenizer.")

    def analyze_text(self, text_input: Optional[str]) -> Dict[str, Any]:
        """
        Analyze financial text to produce sentiment polarity, label, and macro interpretation.

        Args:
            text_input: Title, snippet, or full article text.

        Returns:
            Dict containing:
            - sentiment_score: float in [-1.0, 1.0]
            - sentiment_label: 'positive', 'neutral', 'negative', 'mixed'
            - macro_bias: 'dovish_expansion', 'hawkish_tightening', 'recessionary', 'neutral'
            - confidence: float in [0.0, 1.0]
        """
        if not text_input or not str(text_input).strip():
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "macro_bias": "neutral",
                "confidence": 0.0,
            }

        clean_text = str(text_input).strip()
        lower_text = clean_text.lower()

        # 1. Base Score calculation
        if self._vader_analyzer:
            vader_scores = self._vader_analyzer.polarity_scores(clean_text)
            compound_score = vader_scores["compound"]  # [-1, 1]
            pos_weight = vader_scores["pos"]
            neg_weight = vader_scores["neg"]
        else:
            # Lexicon based counting fallback
            words = re.findall(r"\b\w+\b", lower_text)
            pos_matches = sum(1 for w in words if w in ["growth", "beat", "surge", "gain", "high", "rebound", "resilient"])
            neg_matches = sum(1 for w in words if w in ["fall", "miss", "drop", "recession", "loss", "slump", "crisis"])
            total = pos_matches + neg_matches
            if total > 0:
                compound_score = (pos_matches - neg_matches) / total
            else:
                compound_score = 0.0
            pos_weight = pos_matches / max(len(words), 1)
            neg_weight = neg_matches / max(len(words), 1)

        # 2. Financial Macro Adjustments
        macro_adjustment = 0.0
        hawkish_hits = 0
        dovish_hits = 0
        recession_hits = 0

        for term, weight in MACRO_HAWKISH_TERMS.items():
            if term in lower_text:
                macro_adjustment += weight
                hawkish_hits += 1

        for term, weight in MACRO_DOVISH_BULLISH_TERMS.items():
            if term in lower_text:
                macro_adjustment += weight
                dovish_hits += 1

        for term, weight in MACRO_RECESSIONARY_TERMS.items():
            if term in lower_text:
                macro_adjustment += weight
                recession_hits += 1

        # Combine base compound with macro signals
        combined_score = (compound_score * 0.6) + (macro_adjustment * 0.4)
        bounded_score = max(-1.0, min(1.0, combined_score))
        bounded_score = round(bounded_score, 4)

        # 3. Determine Categorical Label
        if (pos_weight > 0.15 and neg_weight > 0.15) or (hawkish_hits > 0 and dovish_hits > 0):
            label = "mixed"
        elif bounded_score >= 0.15:
            label = "positive"
        elif bounded_score <= -0.15:
            label = "negative"
        else:
            label = "neutral"

        # 4. Macroeconomic Regime Interpretation
        if recession_hits > 0 and bounded_score < -0.3:
            macro_bias = "recessionary"
        elif hawkish_hits > dovish_hits and bounded_score < 0:
            macro_bias = "hawkish_tightening"
        elif dovish_hits > hawkish_hits and bounded_score > 0:
            macro_bias = "dovish_expansion"
        else:
            macro_bias = "neutral"

        confidence = round(min(1.0, abs(bounded_score) * 0.8 + 0.2), 2)

        return {
            "sentiment_score": bounded_score,
            "sentiment_label": label,
            "macro_bias": macro_bias,
            "confidence": confidence,
        }

    def batch_analyze_news(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich a collection of articles with sentiment metrics."""
        for art in articles:
            text_to_eval = f"{art.get('title', '')} {art.get('snippet', '')}"
            res = self.analyze_text(text_to_eval)
            art["sentiment_score"] = res["sentiment_score"]
            art["sentiment_label"] = res["sentiment_label"]
            art["macro_bias"] = res["macro_bias"]
            art["confidence"] = res["confidence"]
        return articles
