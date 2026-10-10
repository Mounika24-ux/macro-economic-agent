"""
Streamlit Interactive Dashboard for Macroeconomic Intelligence & Strategy Evaluation.

Provides real-time inspection of:
- Pipeline Status & Data Coverage
- Economic Event Calendar & Surprises
- Macro News Sentiment Timeline & Sources
- Strategy Permutation Parameter Matrix & Heatmap
- Machine Learning Out-of-Sample Models & Walk-Forward Folds
- Executive Macro Outlook & Evaluation Reports
"""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.settings import get_settings
from src.database import EconomicEvent, MarketPrice, NewsArticle, PipelineRun, get_db
from src.pipeline import MacroPipeline

# Page configuration
st.set_page_config(
    page_title="CrowdWisdomTrading Macro Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

settings = get_settings()
db = get_db()

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #475569; margin-bottom: 1.5rem; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .badge-demo { background: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-live { background: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


def load_database_data():
    """Load tables from local SQLite database."""
    with db.session_scope() as session:
        events = session.query(EconomicEvent).order_by(EconomicEvent.timestamp_utc.desc()).all()
        news = session.query(NewsArticle).order_by(NewsArticle.published_at_utc.desc()).all()
        market = session.query(MarketPrice).order_by(MarketPrice.timestamp_utc.asc()).all()
        runs = session.query(PipelineRun).order_by(PipelineRun.timestamp.desc()).all()

        events_data = [{
            "Event Name": e.event_name,
            "Country": e.country,
            "Date (UTC)": e.timestamp_utc.strftime("%Y-%m-%d %H:%M"),
            "Actual": e.actual,
            "Forecast": e.forecast,
            "Previous": e.previous,
            "Surprise": e.surprise,
            "Impact": e.impact_level,
            "Synthetic": e.is_synthetic
        } for e in events]

        news_data = [{
            "Title": n.title,
            "Source": n.source,
            "Published": n.published_at_utc.strftime("%Y-%m-%d %H:%M"),
            "Sentiment Score": n.sentiment_score,
            "Label": n.sentiment_label,
            "Provider": n.provider,
            "URL": n.url,
            "Snippet": n.snippet,
            "Synthetic": n.is_synthetic
        } for n in news]

        market_data = [{
            "Date": m.timestamp_utc,
            "Open": m.open,
            "High": m.high,
            "Low": m.low,
            "Close": m.close,
            "Volume": m.volume,
            "Symbol": m.symbol,
            "Synthetic": m.is_synthetic
        } for m in market]

        runs_data = [{
            "Run ID": r.run_id,
            "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Mode": r.mode,
            "Status": r.status,
            "Summary": r.summary_json
        } for r in runs]

        return (
            pd.DataFrame(events_data),
            pd.DataFrame(news_data),
            pd.DataFrame(market_data),
            pd.DataFrame(runs_data)
        )


# Sidebar controls
st.sidebar.title("🎛️ Pipeline Controls")
st.sidebar.markdown("Execute intelligence pipeline locally:")

selected_mode = st.sidebar.selectbox("Pipeline Mode", ["demo", "fallback", "live"], index=0, help="demo: synthetic data, fallback: yfinance/local, live: Apify/Tavily APIs")
selected_symbol = st.sidebar.text_input("Asset Symbol", value="SPY")
lookback = st.sidebar.slider("Lookback (Days)", min_value=30, max_value=365, value=180, step=15)

if st.sidebar.button("🚀 Run Full Pipeline", use_container_width=True):
    with st.spinner(f"Running pipeline in '{selected_mode}' mode for {selected_symbol}..."):
        pipe = MacroPipeline(mode=selected_mode)
        res = pipe.run_full_pipeline(symbol=selected_symbol, lookback_days=lookback)
        st.sidebar.success(f"Pipeline Finished: {res.get('status', 'SUCCESS')}")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 API Status")
apify_ok = bool(settings.apify_api_token and not settings.apify_api_token.startswith("your_"))
tavily_ok = bool(settings.tavily_api_key and not settings.tavily_api_key.startswith("your_"))
exa_ok = bool(settings.exa_api_key and not settings.exa_api_key.startswith("your_"))

st.sidebar.markdown(f"**Apify**: {'🟢 Connected' if apify_ok else '⚪ Unconfigured'}")
st.sidebar.markdown(f"**Tavily**: {'🟢 Connected' if tavily_ok else '⚪ Unconfigured'}")
st.sidebar.markdown(f"**Exa**: {'🟢 Connected' if exa_ok else '⚪ Unconfigured'}")

# Header
st.markdown('<div class="main-header">CrowdWisdomTrading: Macroeconomic Intelligence Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Macroeconomic Event Tracking, NLP News Sentiment, Strategy Permutations & Walk-Forward ML Evaluation</div>', unsafe_allow_html=True)

# Load data
events_df, news_df, market_df, runs_df = load_database_data()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview & Status",
    "📅 Economic Calendar",
    "📰 News & Sentiment",
    "⚡ Strategy Grid",
    "🤖 ML Models & Folds",
    "📑 Reports & Outlook"
])

# -------------------------------------------------------------
# TAB 1: Overview & Status
# -------------------------------------------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Economic Events", len(events_df))
    with col2:
        st.metric("News Articles Analyzed", len(news_df))
    with col3:
        st.metric("Market Price Bars", len(market_df))
    with col4:
        st.metric("Pipeline Runs", len(runs_df))

    st.markdown("### Recent Pipeline Runs")
    if not runs_df.empty:
        st.dataframe(runs_df.drop(columns=["Summary"]), use_container_width=True)
    else:
        st.info("No pipeline runs recorded yet. Use the sidebar to trigger your first run.")

    if not market_df.empty:
        st.markdown("### Asset Price Series (OHLCV)")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=market_df["Date"],
            open=market_df["Open"],
            high=market_df["High"],
            low=market_df["Low"],
            close=market_df["Close"],
            name="OHLC"
        ))
        fig.update_layout(title=f"{selected_symbol} Daily Price Action", xaxis_rangeslider_visible=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Economic Calendar
# -------------------------------------------------------------
with tab2:
    st.markdown("### Macroeconomic Event Calendar & Indicator Surprises")
    if not events_df.empty:
        # Surprise Chart
        surp_df = events_df.dropna(subset=["Surprise"]).copy()
        if not surp_df.empty:
            surp_df["Color"] = surp_df["Surprise"].apply(lambda x: "Surprise Above Forecast" if x >= 0 else "Surprise Below Forecast")
            fig = px.scatter(
                surp_df,
                x="Date (UTC)",
                y="Surprise",
                color="Color",
                size="Impact",
                hover_data=["Event Name", "Actual", "Forecast", "Previous"],
                color_discrete_map={"Surprise Above Forecast": "#22c55e", "Surprise Below Forecast": "#ef4444"},
                title="Economic Indicator Surprises (Actual - Forecast)"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(events_df, use_container_width=True)
    else:
        st.info("No economic events in local database. Run the pipeline to ingest data.")

# -------------------------------------------------------------
# TAB 3: News & Sentiment
# -------------------------------------------------------------
with tab3:
    st.markdown("### Macroeconomic News Sentiment & Volume Timeline")
    if not news_df.empty:
        # Sentiment distribution
        col1, col2 = st.columns([1, 2])
        with col1:
            label_counts = news_df["Label"].value_counts().reset_index()
            label_counts.columns = ["Label", "Count"]
            fig_pie = px.pie(label_counts, names="Label", values="Count", title="Sentiment Breakdown", color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_hist = px.histogram(news_df, x="Sentiment Score", nbins=20, title="Sentiment Score Distribution [-1.0, 1.0]", color_discrete_sequence=["#3b82f6"])
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#64748b")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("### Latest News Headlines")
        st.dataframe(news_df[["Title", "Source", "Published", "Sentiment Score", "Label", "Provider"]], use_container_width=True)
    else:
        st.info("No news articles in local database. Run the pipeline to collect news.")

# -------------------------------------------------------------
# TAB 4: Strategy Grid
# -------------------------------------------------------------
with tab4:
    st.markdown("### Strategy Permutation Matrix & Performance Heatmap")
    heatmap_img = settings.figures_dir / "strategy_heatmap.png"
    equity_img = settings.figures_dir / "equity_and_drawdown.png"

    col1, col2 = st.columns(2)
    with col1:
        if heatmap_img.exists():
            st.image(str(heatmap_img), caption="Strategy Sharpe Ratio Heatmap", use_column_width=True)
        else:
            st.info("Run the pipeline to generate strategy heatmap.")

    with col2:
        if equity_img.exists():
            st.image(str(equity_img), caption="Out-of-Sample Strategy Cumulative Equity & Drawdown", use_column_width=True)
        else:
            st.info("Run the pipeline to generate equity curve.")

# -------------------------------------------------------------
# TAB 5: ML Models & Walk-Forward Folds
# -------------------------------------------------------------
with tab5:
    st.markdown("### Machine Learning Model Evaluation & Chronological Folds")
    model_img = settings.figures_dir / "model_comparison.png"
    if model_img.exists():
        st.image(str(model_img), caption="Model Out-of-Sample Performance Comparison", use_column_width=True)

    meta_files = list(settings.artifacts_dir.glob("*_metadata.json"))
    if meta_files:
        st.markdown("### Persisted Model Artifacts")
        for mf in meta_files:
            with open(mf, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.json(meta)

# -------------------------------------------------------------
# TAB 6: Reports & Outlook
# -------------------------------------------------------------
with tab6:
    rep_choice = st.radio("Select Report", ["Macroeconomic Outlook (One-Page)", "Quantitative Evaluation Report"], horizontal=True)

    if rep_choice == "Macroeconomic Outlook (One-Page)":
        macro_path = settings.reports_dir / "macro_outlook.md"
        if macro_path.exists():
            with open(macro_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Macro outlook report has not been generated yet.")
    else:
        eval_path = settings.reports_dir / "evaluation_report.md"
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Evaluation report has not been generated yet.")
