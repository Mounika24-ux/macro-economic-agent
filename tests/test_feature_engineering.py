"""
Unit tests for feature engineering, time alignment, and no-lookahead assertions.
"""
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import FeatureEngineer


def test_build_feature_matrix_columns(sample_market_df, sample_events, sample_news):
    fe = FeatureEngineer()
    f_df = fe.build_feature_matrix(sample_market_df, sample_events, sample_news)

    # Verify key columns exist
    expected_cols = [
        "ret_1d_lag1", "ret_3d_lag1", "realized_vol_10d", "normalized_hl_range",
        "news_sentiment_daily", "news_sentiment_7d", "news_volume_1d",
        "event_count_today", "event_surprise_mean_today", "hours_to_next_high_impact_event",
        "target_pnl_1d", "target_pnl_5d", "target_win_1d"
    ]
    for col in expected_cols:
        assert col in f_df.columns, f"Missing feature column: {col}"

    assert len(f_df) == len(sample_market_df)


def test_no_lookahead_bias(sample_market_df, sample_events, sample_news):
    """
    CRITICAL QUANT TEST:
    Modifying market prices or news in future periods (e.g. days 40-60)
    must strictly NOT alter feature values at day 20.
    """
    fe = FeatureEngineer()

    # Base feature matrix
    f_df_base = fe.build_feature_matrix(sample_market_df, sample_events, sample_news)

    # Create mutated future dataframe (inject huge price surge from day 40 onward)
    mutated_market = sample_market_df.copy()
    mutated_market.iloc[40:, mutated_market.columns.get_loc("close")] *= 5.0
    mutated_market.iloc[40:, mutated_market.columns.get_loc("high")] *= 5.0

    f_df_mutated = fe.build_feature_matrix(mutated_market, sample_events, sample_news)

    # Feature columns to check at day 20 (strictly preceding day 40)
    check_cols = [
        "ret_1d_lag1", "ret_3d_lag1", "realized_vol_10d", "dist_sma_20",
        "news_sentiment_7d", "event_surprise_mean_today"
    ]

    for col in check_cols:
        base_val = f_df_base.iloc[20][col]
        mut_val = f_df_mutated.iloc[20][col]
        assert np.isclose(base_val, mut_val, atol=1e-7), (
            f"Lookahead bias detected! Column {col} at index 20 changed from {base_val} to {mut_val}."
        )
