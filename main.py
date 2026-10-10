"""
CrowdWisdomTrading Macroeconomic Sentiment Agent - CLI Entry Point.

Execute complete or modular pipeline stages, run tests, verify API credentials,
import data, or export CSV tables from the command line.
"""
import argparse
import io
import logging
import sys
from pathlib import Path

# Set UTF-8 encoding for standard output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import get_settings
from src.apify_ingestion import ApifyCalendarIngestion
from src.exa_ingestion import ExaNewsIngestion
from src.pipeline import MacroPipeline
from src.tavily_ingestion import TavilyNewsIngestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MacroAgentCLI")


def check_api_connections():
    """Test and print status of Apify, Tavily, and Exa API credentials."""
    print("\n" + "="*70)
    print("API CONNECTIVITY & CREDENTIAL VERIFICATION")
    print("="*70)

    # 1. Apify
    apify = ApifyCalendarIngestion()
    print(f"\n1. Apify API (Actor ID: {apify.actor_id}):")
    if not apify.is_configured:
        print("   Status:        [UNCONFIGURED] (Using synthetic calendar demonstration)")
        print("   Setup Action:  Set APIFY_API_TOKEN in .env (Get token at https://apify.com)")
    else:
        res = apify.test_connection()
        status = res.get("status")
        if status == "OK":
            print(f"   Auth Status:   [CONNECTED] (User: {res.get('username')}, Email: {res.get('email')})")
            act_info = res.get("actor_details", {})
            if "error" in act_info:
                print(f"   Actor Status:  [!] {act_info['error']}")
            else:
                print(f"   Actor Name:    {act_info.get('title') or act_info.get('name')}")
                print(f"   Pricing Model: {act_info.get('pricing_model')}")
                print(f"   Public Actor:  {'Yes' if act_info.get('is_public') else 'No'}")
        elif status == "AUTH_FAILED":
            print(f"   Status:        [AUTH_FAILED] ({res.get('error_message')})")
        else:
            print(f"   Status:        [{status}] ({res.get('error_message')})")

    # 2. Tavily
    tavily = TavilyNewsIngestion()
    print(f"\n2. Tavily Search API:")
    if not tavily.is_configured:
        print("   Status:        [UNCONFIGURED] (Using synthetic news demonstration)")
        print("   Setup Action:  Set TAVILY_API_KEY in .env (Get key at https://tavily.com)")
    else:
        res = tavily.test_connection()
        status = res.get("status")
        if status == "OK":
            print(f"   Status:        [CONNECTED] (Live search available)")
        elif status == "AUTH_FAILED":
            print(f"   Status:        [AUTH_FAILED] ({res.get('error_message')})")
        else:
            print(f"   Status:        [{status}] ({res.get('error_message')})")

    # 3. Exa
    exa = ExaNewsIngestion()
    print(f"\n3. Exa Search API:")
    if not exa.is_configured:
        print("   Status:        [UNCONFIGURED] (Using synthetic news demonstration)")
        print("   Setup Action:  Set EXA_API_KEY in .env (Get key at https://exa.ai)")
    else:
        res = exa.test_connection()
        status = res.get("status")
        if status == "OK":
            print(f"   Status:        [CONNECTED] (Live search available)")
        elif status == "AUTH_FAILED":
            print(f"   Status:        [AUTH_FAILED] ({res.get('error_message')})")
        else:
            print(f"   Status:        [{status}] ({res.get('error_message')})")

    print("\n" + "="*70 + "\n")


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
        "--check-apis",
        action="store_true",
        help="Test connectivity and authentication for Apify, Tavily, and Exa APIs"
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Execute automated pytest test suite"
    )

    args = parser.parse_args()

    if args.check_apis:
        check_api_connections()
        return

    if args.run_tests:
        import pytest
        logger.info("Executing pytest test suite...")
        exit_code = pytest.main(["-v", "tests"])
        sys.exit(exit_code)

    pipeline = MacroPipeline(mode=args.mode)

    if args.run_all or (not args.export_csv and not args.check_apis and not args.run_tests):
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
