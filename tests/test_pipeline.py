"""
End-to-end integration test for the Macroeconomic Pipeline in demonstration mode.
"""
from pathlib import Path
import pytest
from src.database import DatabaseManager
from src.pipeline import MacroPipeline


def test_full_pipeline_demo_execution(tmp_path):
    # Isolated test database
    db_file = tmp_path / "pipeline_test.db"
    db_mgr = DatabaseManager(db_url=f"sqlite:///{db_file}")

    pipeline = MacroPipeline(mode="demo", db_manager=db_mgr)
    summary = pipeline.run_full_pipeline(symbol="SPY", lookback_days=90)

    assert "run_id" in summary
    assert summary["mode"] == "demo"
    assert summary["total_events"] > 0
    assert summary["total_news"] > 0
    assert summary["total_price_bars"] > 0
    assert "best_model" in summary
    assert Path(summary["macro_report_path"]).exists()
    assert Path(summary["eval_report_path"]).exists()
    assert len(summary["figures"]) == 5
