"""
Unit tests for macroeconomic lexicon and VADER sentiment analyzer.
"""
import pytest
from src.sentiment_analysis import MacroSentimentAnalyzer


def test_sentiment_scoring_polarity():
    analyzer = MacroSentimentAnalyzer()

    # Dovish / Bullish text
    pos_res = analyzer.analyze_text("CPI inflation cools significantly as Fed prepares interest rate cut and soft landing.")
    assert pos_res["sentiment_score"] > 0.1
    assert pos_res["sentiment_label"] in ["positive", "mixed"]

    # Recessionary / Bearish text
    neg_res = analyzer.analyze_text("Severe recession fears mount as manufacturing slumps and widespread layoffs accelerate.")
    assert neg_res["sentiment_score"] < -0.1
    assert neg_res["sentiment_label"] in ["negative", "mixed"]

    # Neutral text
    neu_res = analyzer.analyze_text("Treasury Department releases scheduled auction calendar for upcoming quarter.")
    assert -0.2 <= neu_res["sentiment_score"] <= 0.2


def test_sentiment_bounds_and_empty():
    analyzer = MacroSentimentAnalyzer()

    empty_res = analyzer.analyze_text("")
    assert empty_res["sentiment_score"] == 0.0
    assert empty_res["sentiment_label"] == "neutral"

    none_res = analyzer.analyze_text(None)
    assert none_res["sentiment_score"] == 0.0
