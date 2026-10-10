"""
Database Management Layer using SQLAlchemy and SQLite.

Defines schemas, indexes, uniqueness constraints, idempotent inserts,
and CSV export/import capabilities.
"""
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union
import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from config.settings import get_settings
from src.preprocessing import compute_url_hash, normalize_title, parse_timestamp_to_utc

Base = declarative_base()


class EconomicEvent(Base):
    """Stores scraped/synthetic macroeconomic indicator releases."""
    __tablename__ = "economic_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_name = Column(String(255), nullable=False, index=True)
    country = Column(String(50), nullable=True, index=True)
    currency = Column(String(10), nullable=True, index=True)
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    original_timestamp = Column(String(100), nullable=True)
    source_timezone = Column(String(50), nullable=True)
    actual = Column(Float, nullable=True)
    forecast = Column(Float, nullable=True)
    previous = Column(Float, nullable=True)
    surprise = Column(Float, nullable=True)
    impact_level = Column(String(20), nullable=True)  # High, Medium, Low
    raw_payload = Column(Text, nullable=True)
    is_synthetic = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        UniqueConstraint("event_name", "timestamp_utc", "country", name="uq_event_instance"),
        Index("idx_events_time_country", "timestamp_utc", "country"),
    )


class NewsArticle(Base):
    """Stores macroeconomic news collected via Tavily, Exa, or fallback."""
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url_hash = Column(String(64), nullable=False, unique=True, index=True)
    title = Column(String(512), nullable=False)
    normalized_title = Column(String(512), nullable=False, index=True)
    url = Column(Text, nullable=False)
    source = Column(String(100), nullable=True, index=True)
    query = Column(String(255), nullable=True)
    published_at_utc = Column(DateTime, nullable=False, index=True)
    snippet = Column(Text, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    sentiment_label = Column(String(20), nullable=True)  # positive, neutral, negative, mixed
    provider = Column(String(50), nullable=False)  # tavily, exa, synthetic
    is_synthetic = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index("idx_news_pubtime_provider", "published_at_utc", "provider"),
    )


class MarketPrice(Base):
    """Stores OHLCV market price series."""
    __tablename__ = "market_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    source = Column(String(50), default="yfinance", nullable=False)
    is_synthetic = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp_utc", name="uq_market_symbol_time"),
        Index("idx_market_symbol_time", "symbol", "timestamp_utc"),
    )


class EngineeredFeature(Base):
    """Stores precomputed feature matrices aligned on UTC timestamps."""
    __tablename__ = "engineered_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_utc = Column(DateTime, nullable=False, unique=True, index=True)
    symbol = Column(String(20), default="SPY", nullable=False)
    features_json = Column(Text, nullable=False)
    target_pnl_1d = Column(Float, nullable=True)
    target_pnl_5d = Column(Float, nullable=True)
    target_win_1d = Column(Integer, nullable=True)
    target_win_5d = Column(Integer, nullable=True)
    is_synthetic = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class StrategyResult(Base):
    """Stores outcome of strategy backtest permutations."""
    __tablename__ = "strategy_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    parameters_json = Column(Text, nullable=False)
    gross_pnl = Column(Float, nullable=False)
    net_pnl = Column(Float, nullable=False)
    total_return_pct = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    sortino_ratio = Column(Float, nullable=False)
    max_drawdown_pct = Column(Float, nullable=False)
    win_rate_pct = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    profit_factor = Column(Float, nullable=True)
    is_synthetic = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index("idx_strat_run_name", "run_id", "strategy_name"),
    )


class ModelPrediction(Base):
    """Stores out-of-sample machine learning predictions per fold."""
    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, index=True)
    fold_index = Column(Integer, nullable=False)
    model_name = Column(String(100), nullable=False)
    target_name = Column(String(50), nullable=False)
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    y_true = Column(Float, nullable=False)
    y_pred = Column(Float, nullable=False)
    is_synthetic = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index("idx_pred_run_fold_model", "run_id", "fold_index", "model_name"),
    )


class PipelineRun(Base):
    """Stores metadata and status of end-to-end pipeline executions."""
    __tablename__ = "pipeline_runs"

    run_id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    mode = Column(String(20), nullable=False)  # live, fallback, demo
    stages_run = Column(Text, nullable=False)   # JSON list of stages
    status = Column(String(20), nullable=False) # SUCCESS, FAILED, RUNNING
    summary_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class DatabaseManager:
    """Database connection and transactional helper."""

    def __init__(self, db_url: Optional[str] = None):
        settings = get_settings()
        self.db_url = db_url or settings.database_url
        # Ensure SQLite path directory exists
        if self.db_url.startswith("sqlite:///"):
            sqlite_path = Path(self.db_url.replace("sqlite:///", ""))
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            self.db_url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()

    def create_tables(self) -> None:
        """Create all tables and indexes if they do not exist."""
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # --- Idempotent Insert Helpers ---

    def insert_economic_events(self, events: List[Dict[str, Any]]) -> int:
        """Idempotently insert economic events using UNIQUE constraints."""
        if not events:
            return 0

        inserted_count = 0
        with self.session_scope() as session:
            for ev in events:
                # Ensure UTC datetime object
                ts = ev.get("timestamp_utc")
                if not isinstance(ts, datetime):
                    ts = parse_timestamp_to_utc(ts, ev.get("source_timezone"))
                if not ts:
                    continue

                event_name = str(ev.get("event_name", "")).strip()
                country = str(ev.get("country", "")).strip() if ev.get("country") else None

                # Query existing record
                existing = session.query(EconomicEvent).filter(
                    EconomicEvent.event_name == event_name,
                    EconomicEvent.timestamp_utc == ts,
                    EconomicEvent.country == country
                ).first()

                if existing:
                    # Update fields if needed
                    existing.actual = ev.get("actual", existing.actual)
                    existing.forecast = ev.get("forecast", existing.forecast)
                    existing.previous = ev.get("previous", existing.previous)
                    existing.surprise = ev.get("surprise", existing.surprise)
                    existing.impact_level = ev.get("impact_level", existing.impact_level)
                else:
                    new_rec = EconomicEvent(
                        event_name=event_name,
                        country=country,
                        currency=ev.get("currency"),
                        timestamp_utc=ts,
                        original_timestamp=str(ev.get("original_timestamp")),
                        source_timezone=ev.get("source_timezone"),
                        actual=ev.get("actual"),
                        forecast=ev.get("forecast"),
                        previous=ev.get("previous"),
                        surprise=ev.get("surprise"),
                        impact_level=ev.get("impact_level"),
                        raw_payload=json.dumps(ev.get("raw_payload", {})) if isinstance(ev.get("raw_payload"), (dict, list)) else ev.get("raw_payload"),
                        is_synthetic=bool(ev.get("is_synthetic", False))
                    )
                    session.add(new_rec)
                    inserted_count += 1

        return inserted_count

    def insert_news_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Idempotently insert news articles using URL hash deduplication."""
        if not articles:
            return 0

        inserted_count = 0
        with self.session_scope() as session:
            for art in articles:
                url = art.get("url", "").strip()
                if not url:
                    continue
                url_hash = art.get("url_hash") or compute_url_hash(url)
                title = art.get("title", "").strip()
                if not title:
                    continue

                existing = session.query(NewsArticle).filter(NewsArticle.url_hash == url_hash).first()
                if not existing:
                    pub_ts = art.get("published_at_utc")
                    if not isinstance(pub_ts, datetime):
                        pub_ts = parse_timestamp_to_utc(pub_ts) or datetime.now(timezone.utc)

                    new_art = NewsArticle(
                        url_hash=url_hash,
                        title=title,
                        normalized_title=art.get("normalized_title") or normalize_title(title),
                        url=url,
                        source=art.get("source"),
                        query=art.get("query"),
                        published_at_utc=pub_ts,
                        snippet=art.get("snippet"),
                        sentiment_score=art.get("sentiment_score"),
                        sentiment_label=art.get("sentiment_label"),
                        provider=art.get("provider", "unknown"),
                        is_synthetic=bool(art.get("is_synthetic", False))
                    )
                    session.add(new_art)
                    inserted_count += 1
                else:
                    # Update sentiment if computed later
                    if art.get("sentiment_score") is not None:
                        existing.sentiment_score = art.get("sentiment_score")
                        existing.sentiment_label = art.get("sentiment_label")

        return inserted_count

    def insert_market_prices(self, symbol: str, df: pd.DataFrame, source: str = "yfinance", is_synthetic: bool = False) -> int:
        """Idempotently insert OHLCV price DataFrame."""
        if df.empty:
            return 0

        inserted_count = 0
        with self.session_scope() as session:
            for idx, row in df.iterrows():
                ts = idx if isinstance(idx, (datetime, pd.Timestamp)) else row.get("timestamp_utc")
                if not isinstance(ts, datetime):
                    ts = parse_timestamp_to_utc(ts)
                if not ts:
                    continue

                existing = session.query(MarketPrice).filter(
                    MarketPrice.symbol == symbol,
                    MarketPrice.timestamp_utc == ts
                ).first()

                if not existing:
                    rec = MarketPrice(
                        symbol=symbol,
                        timestamp_utc=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume", 0)),
                        source=source,
                        is_synthetic=is_synthetic
                    )
                    session.add(rec)
                    inserted_count += 1
                else:
                    existing.open = float(row["open"])
                    existing.high = float(row["high"])
                    existing.low = float(row["low"])
                    existing.close = float(row["close"])
                    existing.volume = float(row.get("volume", 0))

        return inserted_count

    # --- CSV Export & Import ---

    def export_table_to_csv(self, table_name: str, target_csv_path: Union[str, Path]) -> Path:
        """Export any database table to CSV."""
        path = Path(target_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.engine.connect() as conn:
            df = pd.read_sql_table(table_name, conn)
            df.to_csv(path, index=False)
        return path

    def import_table_from_csv(self, table_name: str, source_csv_path: Union[str, Path]) -> int:
        """Import records from a CSV file into a database table."""
        path = Path(source_csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Source CSV not found: {path}")
        df = pd.read_csv(path)
        with self.engine.connect() as conn:
            df.to_sql(table_name, conn, if_exists="append", index=False)
        return len(df)


_db_manager_instance: Optional[DatabaseManager] = None


def get_db(db_url: Optional[str] = None) -> DatabaseManager:
    """Singleton getter for DatabaseManager."""
    global _db_manager_instance
    if _db_manager_instance is None or db_url is not None:
        _db_manager_instance = DatabaseManager(db_url)
    return _db_manager_instance
