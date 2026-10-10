"""
CrowdWisdomTrading Macroeconomic Sentiment Agent - CLI Entry Point.

Execute complete or modular pipeline stages, run tests, import data,
or export CSV tables from the command line.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import get_settings
from src.pipeline import MacroPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MacroAgentCLI")


def main():
    parser = argparse.ArgumentParser(
        description="CrowdWisdomTrading Macroeconomic Intelligence & Strategy Evaluation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["live", "fallback", "demo"],
        default="demo",
        help="Pipeline operation mode: 'live' (APIs), 'fallback' (public sources/yfinance), or 'demo' (synthetic data)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Market asset symbol to analyze"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=180,
        help="Lookback window in days"
    )
    parser.add_argument(
        "--import-csv",
        type=str,
        default=None,
        help="Path to custom user OHLCV CSV file"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Execute all pipeline stages sequentially"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export all database tables to data/processed/ CSV files"
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Execute automated pytest test suite"
    )

    args = parser.parse_args()

    if args.run_tests:
        import pytest
        logger.info("Executing pytest test suite...")
        exit_code = pytest.main(["-v", "tests"])
        sys.exit(exit_code)

    pipeline = MacroPipeline(mode=args.mode)

    if args.run_all or len(sys.argv) == 1:
        logger.info(f"Running full pipeline in '{args.mode}' mode for symbol '{args.symbol}' ({args.lookback} days)...")
        results = pipeline.run_full_pipeline(
            symbol=args.symbol,
            lookback_days=args.lookback,
            custom_csv_path=args.import_csv
        )
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*70)
        print(f"Run ID:        {results.get('run_id')}")
        print(f"Mode:          {results.get('mode')}")
        print(f"Events:        {results.get('total_events')}")
        print(f"News Articles: {results.get('total_news')}")
        print(f"Price Bars:    {results.get('total_price_bars')}")
        print(f"Best Model:    {results.get('best_model')}")
        print(f"Macro Outlook: {results.get('macro_report_path')}")
        print(f"Eval Report:   {results.get('eval_report_path')}")
        print("="*70 + "\n")

    if args.export_csv:
        logger.info("Exporting database tables to CSV...")
        pipeline.export_processed_data()
        logger.info("CSV export completed successfully.")


if __name__ == "__main__":
    main()
