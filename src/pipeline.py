"""
End-to-End Macroeconomic Intelligence and Trading Strategy Evaluation Pipeline.

Coordinates ingestion, sentiment analysis, feature engineering, strategy simulation,
walk-forward ML validation, visualization generation, and reporting.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.database import DatabaseManager, PipelineRun, get_db
from src.economic_calendar import EconomicCalendarCoordinator
from src.exa_ingestion import ExaNewsIngestion
from src.feature_engineering import FeatureEngineer
from src.macro_summary import MacroOutlookGenerator
from src.market_data import MarketDataLoader
from src.sentiment_analysis import MacroSentimentAnalyzer
from src.strategy_engine import StrategyBacktester
from src.visualizations import Visualizer
from src.walk_forward_validation import WalkForwardValidator

logger = logging.getLogger(__name__)


class MacroPipeline:
    """Master pipeline orchestrator."""

    def __init__(
        self,
        mode: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        settings = get_settings()
        self.mode = mode or settings.pipeline_mode
        self.db = db_manager or get_db()
        self.settings = settings

        # Initialize subcomponents
        self.calendar_coord = EconomicCalendarCoordinator()
        self.exa_client = ExaNewsIngestion()
        self.market_loader = MarketDataLoader()
        self.sentiment_analyzer = MacroSentimentAnalyzer()
        self.feature_engineer = FeatureEngineer()
        self.strategy_backtester = StrategyBacktester()
        self.wf_validator = WalkForwardValidator()
        self.outlook_generator = MacroOutlookGenerator()
        self.visualizer = Visualizer()

    def run_full_pipeline(
        self,
        symbol: Optional[str] = None,
        lookback_days: int = 180,
        custom_csv_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute all 8 stages of the macroeconomic intelligence pipeline sequentially.

        Returns:
            Dict containing full execution summary and metric results.
        """
        sym = symbol or self.settings.default_symbol
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info(f"Starting Macroeconomic Intelligence Pipeline (Run ID: {run_id}, Mode: {self.mode}, Symbol: {sym})...")

        executed_stages = []
        pipeline_status = "SUCCESS"
        error_msg = None

        try:
            # -------------------------------------------------------------
            # Stage 1: Macro Calendar & News Ingestion
            # -------------------------------------------------------------
            logger.info("=== STAGE 1: Data Ingestion (Calendar & News) ===")
            events = self.calendar_coord.get_events(lookback_days=lookback_days, mode=self.mode)
            self.db.insert_economic_events(events)
            executed_stages.append("ingest_calendar")

            news_articles = self.fetch_news_articles(lookback_days=lookback_days)
            # Stage 2: Sentiment Scoring & Storage
            logger.info("=== STAGE 2: Sentiment Analysis & Storage ===")
            news_articles = self.sentiment_analyzer.batch_analyze_news(news_articles)
            self.db.insert_news_articles(news_articles)
            executed_stages.append("sentiment_analysis")

            # Ingest Market Data
            logger.info("=== STAGE 3: Market Data Loading ===")
            market_source = "csv" if custom_csv_path else ("synthetic" if self.mode == "demo" else self.settings.market_data_source)
            market_df = self.market_loader.get_market_data(
                symbol=sym,
                lookback_days=lookback_days,
                source=market_source,
                csv_path=custom_csv_path
            )
            is_mkt_synthetic = market_df["is_synthetic"].iloc[0] if "is_synthetic" in market_df.columns else False
            self.db.insert_market_prices(sym, market_df, source=market_source, is_synthetic=is_mkt_synthetic)
            executed_stages.append("ingest_market")

            # -------------------------------------------------------------
            # Stage 4: Feature Engineering
            # -------------------------------------------------------------
            logger.info("=== STAGE 4: Feature Engineering ===")
            feature_df = self.feature_engineer.build_feature_matrix(market_df, events, news_articles)
            executed_stages.append("feature_engineering")

            # -------------------------------------------------------------
            # Stage 5: One-Page Macroeconomic Outlook Report
            # -------------------------------------------------------------
            logger.info("=== STAGE 5: Macro Outlook Report Generation ===")
            macro_report_path = self.outlook_generator.generate_report(
                events=events,
                news_articles=news_articles,
                market_df=market_df,
                symbol=sym,
                is_synthetic=(self.mode == "demo" or is_mkt_synthetic)
            )
            executed_stages.append("macro_outlook_report")

            # -------------------------------------------------------------
            # Stage 6: Strategy Permutation Grid Backtest
            # -------------------------------------------------------------
            logger.info("=== STAGE 6: Strategy Permutation Grid Backtest ===")
            strategy_grid_df = self.strategy_backtester.run_strategy_grid(feature_df)
            executed_stages.append("strategy_grid")

            # -------------------------------------------------------------
            # Stage 7: Chronological Walk-Forward ML Validation
            # -------------------------------------------------------------
            logger.info("=== STAGE 7: Walk-Forward ML Model Validation ===")
            wf_results = self.wf_validator.run_walk_forward_validation(feature_df, target_col="target_pnl_1d")
            executed_stages.append("walk_forward_validation")

            # -------------------------------------------------------------
            # Stage 8: Visualizations & Evaluation Report
            # -------------------------------------------------------------
            logger.info("=== STAGE 8: Visualizations & Evaluation Report ===")
            fig_paths = self.generate_all_visualizations(feature_df, events, strategy_grid_df, wf_results)
            eval_report_path = self.generate_evaluation_report(
                run_id=run_id,
                symbol=sym,
                lookback_days=lookback_days,
                feature_df=feature_df,
                strategy_grid_df=strategy_grid_df,
                wf_results=wf_results,
                is_synthetic=(self.mode == "demo" or is_mkt_synthetic)
            )
            executed_stages.append("visualizations_and_evaluation_report")

            summary_results = {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": self.mode,
                "symbol": sym,
                "total_events": len(events),
                "total_news": len(news_articles),
                "total_price_bars": len(market_df),
                "features_count": len(feature_df.columns),
                "best_model": wf_results["best_model_name"],
                "model_metrics": wf_results["aggregate_metrics"],
                "top_strategy": strategy_grid_df.iloc[0].to_dict() if not strategy_grid_df.empty else {},
                "macro_report_path": str(macro_report_path),
                "eval_report_path": str(eval_report_path),
                "figures": [str(p) for p in fig_paths],
            }

            # Export processed tables to CSV
            self.export_processed_data()

        except Exception as e:
            pipeline_status = "FAILED"
            error_msg = str(e)
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            summary_results = {"run_id": run_id, "error": error_msg}

        # Log pipeline run into database
        with self.db.session_scope() as session:
            rec = PipelineRun(
                run_id=run_id,
                mode=self.mode,
                stages_run=json.dumps(executed_stages),
                status=pipeline_status,
                summary_json=json.dumps(summary_results, default=str),
                error_message=error_msg
            )
            session.add(rec)

        logger.info(f"Pipeline completed with status: {pipeline_status}")
        return summary_results

    def fetch_news_articles(self, lookback_days: int = 180) -> List[Dict[str, Any]]:
        """Fetch news from Tavily, Exa, or generate synthetic demonstration news."""
        from src.tavily_ingestion import TavilyNewsIngestion
        tavily_client = TavilyNewsIngestion()

        if self.mode == "live":
            articles = []
            if tavily_client.is_configured:
                articles.extend(tavily_client.fetch_macro_news(days_back=min(lookback_days, 30)))
            if self.exa_client.is_configured:
                articles.extend(self.exa_client.fetch_macro_news())
            if articles:
                return articles

        # Synthetic Demonstration news generator
        logger.info("Generating explicitly labeled synthetic macroeconomic news stream (Demo Mode)...")
        return self._generate_synthetic_news(lookback_days=lookback_days)

    def _generate_synthetic_news(self, lookback_days: int = 180, seed: int = 42) -> List[Dict[str, Any]]:
        """Generate reproducible synthetic news with realistic headlines and snippets."""
        import numpy as np
        np.random.seed(seed)

        headlines_templates = [
            ("CPI inflation cools to 2.9%, reinforcing market expectations of Fed rate cut", "positive", "Financial Times"),
            ("Federal Reserve holds benchmark rate steady, signals patient approach to monetary easing", "neutral", "Bloomberg"),
            ("US Nonfarm payrolls beat forecast with 215k additions, showcasing labor resilience", "positive", "Wall Street Journal"),
            ("Manufacturing PMI contracts unexpectedly amid rising supply chain uncertainty", "negative", "Reuters"),
            ("Treasury yields surge following hawkish comments from central bank governors", "negative", "CNBC"),
            ("Consumer confidence rebounds strongly as gas prices and inflation stabilize", "positive", "MarketWatch"),
            ("GDP expands at 2.8% annualized rate, surpassing economists' consensus estimates", "positive", "Barron's"),
            ("Tech earnings volatility and rate uncertainty weigh on risk sentiment", "mixed", "Yahoo Finance"),
        ]

        end_dt = datetime.now(timezone.utc)
        articles = []
        for i in range(min(lookback_days * 2, 80)):
            title, label, source = headlines_templates[i % len(headlines_templates)]
            days_ago = float(np.random.uniform(0.5, lookback_days))
            pub_ts = end_dt - pd.Timedelta(days=days_ago)

            articles.append({
                "url_hash": f"synth_news_{i:04d}",
                "title": f"[Demo] {title} ({i+1})",
                "normalized_title": f"demo {title.lower()}",
                "url": f"https://example.com/macro-news/{i+1}",
                "canonical_url": f"https://example.com/macro-news/{i+1}",
                "source": source,
                "query": "macroeconomic inflation fed policy",
                "published_at_utc": pub_ts,
                "snippet": f"Summary analysis of macroeconomic conditions and {title.lower()}.",
                "provider": "synthetic",
                "is_synthetic": True,
            })

        articles.sort(key=lambda x: x["published_at_utc"])
        return articles

    def generate_all_visualizations(
        self,
        feature_df: pd.DataFrame,
        events: List[Dict],
        strategy_grid_df: pd.DataFrame,
        wf_results: Dict[str, Any]
    ) -> List[Path]:
        """Generate all figures for reports and dashboard."""
        paths = []
        # 1. Sentiment timeline
        paths.append(self.visualizer.plot_sentiment_timeline(feature_df))
        # 2. Event surprises
        paths.append(self.visualizer.plot_event_surprises(events))
        # 3. Strategy heatmap
        paths.append(self.visualizer.plot_strategy_heatmap(strategy_grid_df))
        # 4. Model comparison
        paths.append(self.visualizer.plot_model_comparison(wf_results["aggregate_metrics"]))

        # 5. Equity and Drawdown
        oos_dates = pd.to_datetime(wf_results["oos_dates"], utc=True)
        best_model = wf_results["best_model_name"]
        best_preds = np.array(wf_results["oos_predictions"][best_model])
        oos_actuals = np.array(wf_results["oos_actuals"])

        strat_returns = pd.Series(np.sign(best_preds) * oos_actuals, index=oos_dates)
        bench_returns = pd.Series(oos_actuals, index=oos_dates)
        paths.append(self.visualizer.plot_equity_and_drawdown(
            strategy_returns=strat_returns,
            benchmark_returns=bench_returns,
            strategy_label=f"ML Strategy ({best_model})",
            benchmark_label="SPY Buy & Hold"
        ))

        return paths

    def generate_evaluation_report(
        self,
        run_id: str,
        symbol: str,
        lookback_days: int,
        feature_df: pd.DataFrame,
        strategy_grid_df: pd.DataFrame,
        wf_results: Dict[str, Any],
        is_synthetic: bool = False
    ) -> Path:
        """Generate `reports/evaluation_report.md`."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        agg = wf_results["aggregate_metrics"]
        best_model = wf_results["best_model_name"]
        folds = wf_results["folds"]

        tag = "⚠️ **DEMONSTRATION RUN (SYNTHETIC / HYBRID DATA)**" if is_synthetic else "✅ **EMPIRICAL HISTORICAL DATA**"

        doc = f"""# CrowdWisdomTrading: Quantitative Strategy & ML Evaluation Report

**Run Identifier**: `{run_id}`  
**Evaluation Timestamp**: {now_str}  
**Data Integrity Status**: {tag}  
**Underlying Asset**: `{symbol}` | **Lookback**: `{lookback_days} days`  
**Total Feature Samples**: `{len(feature_df)}` | **Feature Count**: `{len(feature_df.columns)}`

---

## 1. Executive Quantitative Summary

This report evaluates the out-of-sample predictability of macroeconomic sentiment signals and event surprises combined with market technical indicators across multiple machine learning algorithms and strategy execution parameter grids.

| Algorithm / Strategy | OOS Directional Accuracy | OOS Sharpe Ratio | Total OOS Return | Max Drawdown | Sortino Ratio |
|:---|:---:|:---:|:---:|:---:|:---:|
"""
        for model_name, m in agg.items():
            dir_acc = f"{m.get('oos_directional_accuracy', 0.0):.1f}%" if "oos_directional_accuracy" in m else "N/A"
            doc += f"| **{model_name}** | {dir_acc} | **{m['strategy_sharpe']:.2f}** | {m['strategy_total_return_pct']:+.2f}% | {m['strategy_max_drawdown_pct']:.2f}% | {m['strategy_sortino']:.2f} |\n"

        doc += f"""
---

## 2. Chronological Walk-Forward Validation Methodology

- **Validation Protocol**: Chronological expanding window cross-validation (strictly non-leaking).
- **Purging & Embargo**: `{self.settings.wf_purge_days}`-day embargo gap between training and validation periods to prevent autocorrelation leakage from multi-day forward return labels.
- **Normalization Hygiene**: `StandardScaler` fitted strictly on training observations for each fold.

### Fold-Level Breakdown

| Fold # | Training Period | Validation Period | Train N | Test N | Best Fold Model Test MSE | Directional Acc % |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
"""
        for f in folds:
            b_model_res = f["models"].get(best_model, {})
            mse_val = f"{b_model_res.get('mse_test', 0.0):.6f}"
            acc_val = f"{b_model_res.get('directional_accuracy', 0.0)*100.0:.1f}%"
            doc += f"| **Fold {f['fold_idx']}** | {f['train_start']} → {f['train_end']} | {f['test_start']} → {f['test_end']} | {f['train_samples']} | {f['test_samples']} | `{mse_val}` | **{acc_val}** |\n"

        doc += f"""
---

## 3. Strategy Parameter Grid Permutations

Top performing parameter combinations from backtesting across stop-loss, take-profit, and lookback windows:

| Strategy Permutation | Stop-Loss | Take-Profit | Lookback | Gross P&L | Net P&L | Sharpe Ratio | Win Rate | Max Drawdown |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
        if not strategy_grid_df.empty:
            for _, row in strategy_grid_df.head(5).iterrows():
                gross_str = f"${row['gross_pnl']:,.2f}"
                net_str = f"${row['net_pnl']:,.2f}"
                doc += f"| `{row['strategy_name']}` | {row['stop_loss_pct']*100:.1f}% | {row['take_profit_pct']*100:.1f}% | {int(row['lookback_window'])}d | {gross_str} | **{net_str}** | **{row['sharpe_ratio']:.2f}** | {row['win_rate_pct']:.1f}% | {row['max_drawdown_pct']:.2f}% |\n"

        comm_str = f"${self.settings.commission_per_share}"
        doc += f"""
---

## 4. Execution & Friction Assumptions

- **Commission**: {comm_str}/share
- **Slippage**: {self.settings.slippage_bps} basis points (0.015%) applied on entry and exit.
- **Risk-Free Rate**: {self.settings.annual_risk_free_rate * 100:.1f}% annualized.
- **Trading Calendar**: 252 business days per annum.

---

## 5. Visual Artifacts

The following visual artifacts were generated and saved in `reports/figures/`:
1. `reports/figures/sentiment_timeline.png` — Rolling macroeconomic sentiment vs news volume.
2. `reports/figures/event_surprises.png` — Economic indicator surprises (Actual vs Forecast).
3. `reports/figures/equity_and_drawdown.png` — Out-of-sample equity growth and underwater drawdown.
4. `reports/figures/strategy_heatmap.png` — Strategy parameter Sharpe ratio heatmap.
5. `reports/figures/model_comparison.png` — Out-of-sample directional accuracy and Sharpe comparison.

---

## 6. Risk Warnings & Quantitative Limitations

1. **Regime Non-Stationarity**: Macro sentiment relationships with asset returns evolve across interest rate regimes.
2. **Execution Latency**: Fast economic indicator spikes may experience higher intra-bar slippage than modeled.
3. **Overfitting Avoidance**: Real capital deployment requires ongoing walk-forward monitoring.

---
*Report generated automatically by the CrowdWisdomTrading Quantitative Evaluation Pipeline.*
"""
        report_path = self.settings.reports_dir / "evaluation_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(doc.strip() + "\n")

        logger.info(f"Generated evaluation report at: {report_path}")
        return report_path

    def export_processed_data(self) -> None:
        """Export core SQLite tables to data/processed/ CSV files."""
        for table in ["economic_events", "news_articles", "market_prices"]:
            try:
                target = self.settings.processed_data_dir / f"{table}.csv"
                self.db.export_table_to_csv(table, target)
            except Exception as e:
                logger.warning(f"Could not export table {table} to CSV: {e}")
