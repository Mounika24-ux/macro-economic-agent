# CrowdWisdomTrading: Macroeconomic Intelligence & Strategy Evaluation Agent

A production-grade, reproducible, and fully tested Python 3.12 macroeconomic intelligence pipeline and quantitative trading strategy evaluation framework.

---

## 📌 1. Project Overview

The **Macroeconomic Sentiment Agent** provides automated macroeconomic event tracking, news collection, natural language processing sentiment analysis, quantitative feature engineering, and non-leaking machine learning evaluation of trading strategies.

### Key Capabilities:
1. **Zero-Cost Public Macro Ingestion (FRED & Central Bank Feeds)**: Ingests 100% free, authentic historical releases (CPI, Nonfarm Payrolls, Unemployment Rate, Fed Funds Rate, GDP, Initial Claims) from the Federal Reserve Bank of St. Louis (FRED) and real-time press releases from the Federal Reserve Board RSS feed without requiring paid subscriptions or API credits.
2. **Apify Economic Calendar Ingestion (Optional)**: Configurable actor runner supporting Investing.com/ForexFactory calendar scrapers via Apify's free tier credits.
3. **AI Search Ingestion (Tavily & Exa)**: Ingests macroeconomic news via Tavily and Exa search APIs within their free tiers, featuring automatic URL canonicalization and title deduplication.
4. **Transparent Financial Sentiment Analysis**: Rule and lexicon-based NLP scoring (combining VADER and macroeconomic domain dictionaries) classifying sentiment, regime bias (*dovish expansion*, *hawkish tightening*, *recessionary*), and confidence.
5. **Idempotent Data Layer**: SQLite database orchestrated with SQLAlchemy models, relational indexes, and CSV import/export facilities.
6. **No-Lookahead Feature Engineering**: Generates event surprise deltas, countdown timers to high-impact releases, rolling sentiment momentum, and market technicals with strict temporal alignment.
7. **Strategy Permutation Simulation**: Simulates parameter grids (stop-loss %, take-profit %, lookbacks, sentiment filters) with realistic commissions and basis-point slippage.
8. **Chronological Walk-Forward ML Validation**: Evaluates Ridge Regression, ElasticNet, and Tree Ensembles (RandomForest, GradientBoosting) with expanding/rolling windows and a purge embargo to prevent label autocorrelation leakage.
9. **Financial Metrics Engine**: Calculates annualized Sharpe ratio, Sortino ratio (with downside semi-variance), Maximum Drawdown, Win Rate, and Calmar ratio across out-of-sample periods.
10. **Visual Artifacts & Reporting**: Produces static publication charts, interactive Plotly visualizations, a one-page `reports/macro_outlook.md`, and a comprehensive quantitative `reports/evaluation_report.md`.
11. **Interactive Dashboard**: Full-featured multi-tab Streamlit dashboard (`app.py`).

---

## 🏗️ 2. Project Architecture

```
macro-economic-agent/
├── app.py                         # Interactive Streamlit Web Dashboard
├── main.py                        # CLI entry point for executing pipeline & tests
├── requirements.txt               # Pinned dependencies for Python 3.12
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules for data, models, and environments
├── README.md                      # Comprehensive documentation and reproduction guide
├── config/
│   ├── __init__.py
│   └── settings.py                # Pydantic-based configuration management
├── src/
│   ├── __init__.py
│   ├── apify_ingestion.py         # Apify economic calendar client & schema mapping
│   ├── tavily_ingestion.py        # Tavily macroeconomic news client & deduplication
│   ├── exa_ingestion.py           # Exa search client for macro research
│   ├── economic_calendar.py       # Calendar coordinator & synthetic generator
│   ├── preprocessing.py           # UTC normalization, numeric parsing, text cleaning
│   ├── database.py                # SQLAlchemy SQLite models, idempotency & CSV handlers
│   ├── sentiment_analysis.py      # VADER & macro lexicon sentiment engine
│   ├── feature_engineering.py     # Surprise features, rolling sentiment, technicals
│   ├── market_data.py             # Yahoo Finance (yfinance) & synthetic OHLCV generator
│   ├── strategy_engine.py         # Strategy backtesting engine with realistic frictions
│   ├── model_training.py          # Machine learning model training & artifact persistence
│   ├── walk_forward_validation.py # Chronological expanding/rolling walk-forward validator
│   ├── financial_metrics.py       # Sharpe, Sortino, Drawdown, Win Rate, Gross vs Net
│   ├── macro_summary.py           # One-page macro outlook intelligence generator
│   ├── visualizations.py          # Matplotlib & Plotly chart visualizer
│   └── pipeline.py                # End-to-end master pipeline orchestrator
├── data/
│   ├── raw/                       # Stored raw JSON API payloads for auditability
│   └── processed/                 # Exported database tables in CSV format
├── reports/
│   ├── macro_outlook.md           # One-page executive macroeconomic intelligence summary
│   ├── evaluation_report.md       # Full quantitative model & strategy evaluation report
│   └── figures/                   # Generated PNG visual artifacts
├── artifacts/                     # Persisted joblib models and metadata JSONs
└── tests/
    ├── conftest.py                # Isolated SQLite test fixtures and mock datasets
    ├── test_preprocessing.py      # Tests for timezone parsing & numeric conversion
    ├── test_database.py           # Tests for idempotency and table uniqueness
    ├── test_ingestion.py          # Tests for Apify, Tavily, Exa, and mock fallback
    ├── test_sentiment.py          # Tests for sentiment bounds and financial polarity
    ├── test_feature_engineering.py# Tests for feature construction & no-lookahead assertions
    ├── test_financial_metrics.py  # Tests for Sharpe, Sortino, Drawdown, 0 vol edge cases
    ├── test_strategy_engine.py    # Tests for execution accounting & parameter grid
    ├── test_walk_forward.py       # Tests for chronological split boundaries & purge gap
    ├── test_model_training.py     # Tests for regression model training & serialization
    └── test_pipeline.py           # Integration test for full pipeline in demo mode
```

---

## ⚙️ 3. Installation & Environment Setup (Windows / Python 3.12)

### Step 1: Clone Repository & Open Directory
```powershell
cd d:\macro-economic-agent
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🔑 4. API Account Setup & Configuration

Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

### 1. Apify Calendar Actor Configuration
1. Sign up for an account at [Apify](https://apify.com/).
2. Navigate to **Settings > Integrations** to copy your **Personal API Token**.
3. Choose an economic calendar actor in the Apify Store, such as `trudax/economic-calendar` or `miso88/economic-calendar`.
4. Add credentials to `.env`:
   ```env
   APIFY_API_TOKEN=apify_api_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   APIFY_ACTOR_ID=trudax/economic-calendar
   APIFY_LOOKBACK_DAYS=180
   ```

### 2. Tavily Macroeconomic News Search
1. Register at [Tavily](https://tavily.com/) and obtain an API key.
2. Add to `.env`:
   ```env
   TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### 3. Exa Search API (Optional)
1. Register at [Exa](https://exa.ai/) and generate an API key.
2. Add to `.env`:
   ```env
   EXA_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   ```

### 4. Resilient Fallback & Demonstration Modes
- **`PIPELINE_MODE=fallback` (Recommended for Zero-Cost Real Data)**: Fetches 100% free authentic macroeconomic data from the **Federal Reserve Bank of St. Louis (FRED)**, official central bank news releases from the **Federal Reserve Board RSS**, and market prices from **Yahoo Finance (`yfinance`)** without spending any credits or API tokens.
- **`PIPELINE_MODE=live`**: Uses active API keys for Apify, Tavily, and Exa within their free allowances.
- **`PIPELINE_MODE=demo`**: Runs the entire pipeline seamlessly using **explicitly labeled synthetic benchmark data** (`is_synthetic=True`). Synthetic results are never misrepresented as real trading results.

---

## 🏛️ 5. Zero-Cost Public Data Sources & Methodology Disclosures

### Public Economic Indicators (FRED / U.S. Government)
The zero-cost pipeline retrieves authentic, unrevised historical series directly from the Federal Reserve Bank of St. Louis public data portal:
- `CPIAUCSL`: Consumer Price Index for All Urban Consumers (Monthly)
- `PAYEMS`: Total Nonfarm Payroll Employment (Monthly)
- `UNRATE`: Civilian Unemployment Rate (Monthly)
- `FEDFUNDS`: Federal Funds Effective Rate (Monthly)
- `ICSA`: Initial Jobless Claims (Weekly)
- `GDP`: Gross Domestic Product (Quarterly)

> [!NOTE]
> **Consensus Forecast Disclosure**: Official government statistical portals (FRED, Bureau of Labor Statistics) publish authentic *actual* releases but do not provide commercial Wall Street consensus forecasts (which are proprietary to Bloomberg, Refinitiv, or Dow Jones surveys). In free public mode, the system calculates the historical trailing prior delta $\Delta P_t = \text{Actual}_t - \text{Actual}_{t-1}$ as a transparent indicator surprise reference. When commercial scrapers (e.g. Apify/Investing.com) are connected, consensus forecasts are ingested directly.

---

## 🚀 5. Execution Commands

### 1. Execute Full End-to-End Pipeline
```powershell
# Run full pipeline in demonstration mode
python main.py --mode demo --run-all

# Run full pipeline with live market data (SPY)
python main.py --mode fallback --symbol SPY --lookback 180 --run-all

# Export processed SQLite tables to data/processed/*.csv
python main.py --mode demo --run-all --export-csv
```

### 2. Import Custom OHLCV CSV Data
```powershell
python main.py --mode fallback --import-csv path/to/my_data.csv --symbol CUSTOM --run-all
```

### 3. Run Automated Tests
```powershell
python -m pytest -v
```

### 4. Launch Interactive Streamlit Dashboard
```powershell
streamlit run app.py
```
*Open `http://localhost:8501` in your browser.*

---

## 🔬 6. Quantitative Methodology

### Feature Engineering (No-Lookahead Guarantee)
All indicators at timestamp $t$ are computed strictly using information available $\le t$:
- **Event Surprise**: $S_t = \text{Actual}_t - \text{Forecast}_t$.
- **Previous Change**: $\Delta P_t = \text{Actual}_t - \text{Previous}_t$.
- **Time-to-Event**: Hours and days countdown to scheduled high-impact releases.
- **Rolling Sentiment**: $EMA_{3d}$, $SMA_{7d}$, and sentiment momentum ($SMA_{3d} - SMA_{14d}$).
- **Technicals**: Lagged returns ($R_{t-1, 1d}$, $R_{t-1, 3d}$, $R_{t-1, 5d}$), annualized rolling realized volatility ($10d$, $20d$), and distance from 20-day moving average.

### Chronological Walk-Forward Validation
1. Expanding / rolling window cross-validation across chronological folds.
2. **Purge Embargo**: 2-day gap enforced between training and validation periods to eliminate forward return label overlap leakage.
3. **Data Scaling**: `StandardScaler` fitted strictly on training observations for each fold.
4. **Model Comparison**:
   - **Baseline**: Ridge Regression with $L_2$ regularization.
   - **Alternatives**: ElasticNet ($L_1 + L_2$), Random Forest Regressor, Gradient Boosting Regressor.

### Financial Performance & Risk Metrics
- **Sharpe Ratio**: $\frac{\mathbb{E}[R_p - R_f]}{\sigma_p} \cdot \sqrt{252}$ (annualized, default $R_f = 4.5\%$).
- **Sortino Ratio**: $\frac{\mathbb{E}[R_p - R_f]}{\sqrt{\mathbb{E}[\min(0, R_p - R_f)^2]}} \cdot \sqrt{252}$ (handles zero downside deviation safely).
- **Maximum Drawdown**: $\max_{\tau \le t} \frac{\text{Peak}_\tau - \text{Equity}_t}{\text{Peak}_\tau}$.
- **Friction Assumptions**: Commission (\$0.001/share) and 1.5 bps slippage per side.

---

## 📊 7. Visual Matrix & Output Deliverables

| Deliverable | Location | Description |
|:---|:---|:---|
| **One-Page Macro Outlook** | [`reports/macro_outlook.md`](file:///d:/macro-economic-agent/reports/macro_outlook.md) | Formatted macro summary with inflation, labor, and policy commentary. |
| **Quantitative Evaluation Report** | [`reports/evaluation_report.md`](file:///d:/macro-economic-agent/reports/evaluation_report.md) | In-depth model metrics, fold tables, strategy parameters, and risks. |
| **Sentiment Timeline** | `reports/figures/sentiment_timeline.png` | Daily and rolling 7-day sentiment vs news volume. |
| **Event Surprises** | `reports/figures/event_surprises.png` | Indicator surprise deltas with impact weighting. |
| **Equity & Drawdown** | `reports/figures/equity_and_drawdown.png` | Out-of-sample portfolio growth and underwater drawdowns. |
| **Strategy Heatmap** | `reports/figures/strategy_heatmap.png` | Sharpe ratio matrix across Stop-Loss and Take-Profit grids. |
| **Model Comparison** | `reports/figures/model_comparison.png` | Out-of-sample directional accuracy and Sharpe across algorithms. |
| **Trained ML Models** | `artifacts/*.joblib` | Serialized model objects and metadata JSONs. |

---

## 🔒 8. Security & Data Integrity

- No API keys, credentials, or secrets are committed to version control.
- All environment variables are stored in `.env` (git-ignored).
- Databases (`*.db`), serialized models (`*.joblib`), and temporary cache directories are strictly excluded in `.gitignore`.
- Explicit synthetic data labeling (`is_synthetic=True`) ensures transparency and prevents simulated results from being misrepresented as live returns.
