"""
Application Configuration and Settings Management.

Loads environment variables from `.env` and provides strongly typed
configuration parameters with validated defaults.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

# Inject native Windows certificate store for SSL verification if available
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
except ImportError:
    from pydantic import BaseModel as BaseSettings, Field  # type: ignore
    SettingsConfigDict = dict  # type: ignore

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Configuration settings for Macroeconomic Intelligence Pipeline."""

    # Operation Mode
    pipeline_mode: Literal["live", "fallback", "demo"] = Field(
        default="demo",
        description="Pipeline execution mode: 'live' (APIs), 'fallback' (public sources), or 'demo' (synthetic data)"
    )

    # Base Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    reports_dir: Path = BASE_DIR / "reports"
    figures_dir: Path = BASE_DIR / "reports" / "figures"
    artifacts_dir: Path = BASE_DIR / "artifacts"

    # Database
    database_url: str = Field(
        default=f"sqlite:///{BASE_DIR / 'data' / 'macro_economic.db'}",
        description="SQLAlchemy database connection URL"
    )

    # Apify Configuration
    apify_api_token: Optional[str] = Field(default=None, description="Apify API authentication token")
    apify_actor_id: str = Field(
        default="trudax/economic-calendar",
        description="Apify actor ID for economic calendar scraping"
    )
    apify_lookback_days: int = Field(default=180, description="Calendar lookback window in days")
    apify_timeout_secs: int = Field(default=120, description="Max seconds to wait for Apify actor completion")

    # Tavily Search Configuration
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key for macro news search")
    tavily_max_results_per_topic: int = Field(default=10, description="Max search results per economic topic")

    # Exa Search Configuration
    exa_api_key: Optional[str] = Field(default=None, description="Exa Search API key")
    exa_max_results: int = Field(default=10, description="Max search results for Exa search queries")

    # Market Data
    default_symbol: str = Field(default="SPY", description="Default market ticker symbol for strategy backtests")
    market_data_source: Literal["yfinance", "csv", "synthetic"] = Field(
        default="yfinance",
        description="Primary source for market price bars"
    )

    # Financial & Backtesting Assumptions
    commission_per_share: float = Field(default=0.001, description="Trading commission per share/unit in USD")
    slippage_bps: float = Field(default=1.5, description="Expected slippage in basis points (1 bp = 0.01%)")
    annual_risk_free_rate: float = Field(default=0.045, description="Annual risk-free benchmark rate (e.g. 4.5% = 0.045)")
    trading_days_per_year: int = Field(default=252, description="Annualization factor for daily trading metrics")

    # Walk-Forward Validation Configuration
    wf_n_splits: int = Field(default=5, description="Number of chronological walk-forward folds")
    wf_window_type: Literal["expanding", "rolling"] = Field(default="expanding", description="Walk-forward window style")
    wf_train_size_days: int = Field(default=90, description="Training window duration in days")
    wf_test_size_days: int = Field(default=30, description="Test window duration in days")
    wf_purge_days: int = Field(default=2, description="Purge embargo gap in days to eliminate label leak")

    # Logging
    log_level: str = Field(default="INFO", description="Logging verbosity (DEBUG, INFO, WARNING, ERROR)")

    try:
        model_config = SettingsConfigDict(
            env_file=str(BASE_DIR / ".env"),
            env_file_encoding="utf-8",
            extra="ignore"
        )
    except NameError:
        pass


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
