from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ecopulse.config import AppConfig
from ecopulse.knowledge_base import KnowledgeBase
from ecopulse.pipeline import EcoPulsePipeline
from ecopulse.rag import EcoPulseRAG


st.set_page_config(
    page_title="Eco-Pulse",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(78, 173, 115, 0.16), transparent 32%),
            radial-gradient(circle at top right, rgba(34, 122, 94, 0.14), transparent 28%),
            linear-gradient(180deg, #f5fbf6 0%, #eef5ef 100%);
    }
    .hero-card, .insight-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(38, 102, 71, 0.12);
        border-radius: 20px;
        padding: 1rem 1.2rem;
        box-shadow: 0 18px 40px rgba(32, 73, 46, 0.08);
        backdrop-filter: blur(8px);
    }
    .hero-kicker {
        color: #2b6e4b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .hero-title {
        color: #153a27;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.35rem 0;
    }
    .hero-copy {
        color: #365646;
        font-size: 1rem;
        line-height: 1.5;
    }
    .insight-label {
        color: #5b7d6b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .insight-value {
        color: #173a29;
        font-size: 1.45rem;
        font-weight: 700;
    }
    .insight-copy {
        color: #456454;
        font-size: 0.94rem;
        margin-top: 0.3rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, rgba(26, 84, 59, 0.96), rgba(46, 126, 78, 0.9));
        border-radius: 22px;
        padding: 1.2rem 1.3rem;
        color: #f5fff8;
        box-shadow: 0 24px 48px rgba(22, 61, 42, 0.18);
    }
    .recommendation-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .recommendation-meta {
        color: rgba(245, 255, 248, 0.78);
        font-size: 0.88rem;
        margin-top: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_services() -> tuple[AppConfig, EcoPulsePipeline, KnowledgeBase, EcoPulseRAG]:
    config = AppConfig.from_env()
    pipeline = EcoPulsePipeline(config)
    kb = KnowledgeBase(config)
    kb.ensure_index()
    rag = EcoPulseRAG(config, kb)
    return config, pipeline, kb, rag


def metric_card(label: str, value: str, help_text: str) -> None:
    st.metric(label, value, help=help_text)


def render_hero(snapshot: dict) -> None:
    forecast_summary = snapshot.get("forecast_summary") or {}
    best_window = forecast_summary.get("best_time", "unavailable")
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">Urban Environmental Intelligence</div>
            <div class="hero-title">{snapshot["city"]}</div>
            <div class="hero-copy">
                Live air-quality and weather signals are now connected to local guidance and forecast-aware decision support.
                The best upcoming exposure window is <strong>{best_window}</strong>, based on the latest telemetry and forecast scoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_timeseries(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No environmental records are available yet.")
        return

    chart_df = df.copy()
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
    figure = px.line(
        chart_df,
        x="timestamp",
        y=["pm2_5", "temperature_2m", "uv_index"],
        markers=True,
        title="Environmental Signal Overview",
    )
    figure.update_layout(legend_title_text="Metric")
    st.plotly_chart(figure, use_container_width=True)


def render_forecast_outlook(snapshot: dict) -> None:
    forecast = snapshot.get("forecast") or []
    if not forecast:
        return

    forecast_df = pd.DataFrame(forecast).copy()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    forecast_df["hour"] = forecast_df["timestamp"].dt.strftime("%I:%M %p")

    st.subheader("Next-Hours Outlook")
    figure = px.line(
        forecast_df,
        x="hour",
        y=["pm2_5", "exposure_score", "uv_index"],
        markers=True,
        title="Best Window Forecast",
    )
    figure.update_layout(legend_title_text="Metric")
    st.plotly_chart(figure, use_container_width=True)

    columns = ["hour", "pm2_5", "uv_index", "temperature_2m", "exposure_score", "aqi_category"]
    st.dataframe(forecast_df[columns], use_container_width=True)


def render_insight_cards(snapshot: dict) -> None:
    forecast_summary = snapshot.get("forecast_summary") or {}
    if not forecast_summary:
        return

    first, second, third = st.columns(3)
    cards = [
        (
            "Best Window",
            str(forecast_summary["best_time"]),
            f'{forecast_summary["best_aqi_category"]} conditions with exposure score {forecast_summary["best_exposure_score"]:.1f}.',
        ),
        (
            "Pollution Outlook",
            f'{forecast_summary["best_pm2_5"]:.1f} ug/m3',
            "Forecast PM2.5 during the lowest-exposure hour.",
        ),
        (
            "UV Outlook",
            f'{forecast_summary["best_uv_index"]:.1f}',
            "UV intensity expected during the best upcoming outdoor window.",
        ),
    ]

    for column, (label, value, copy) in zip((first, second, third), cards):
        with column:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-label">{label}</div>
                    <div class="insight-value">{value}</div>
                    <div class="insight-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_evidence(evidence: list[dict[str, str]]) -> None:
    st.subheader("Retrieved Evidence")
    for item in evidence:
        source = item.get("source", "Unknown source")
        content = item.get("content", "").strip()
        with st.container(border=True):
            st.markdown(f"**{source}**")
            st.caption(content[:320] + ("..." if len(content) > 320 else ""))


def main() -> None:
    config, pipeline, _, rag = load_services()

    st.title("Eco-Pulse")
    st.caption("Real-time environmental intelligence for healthy, sustainable city living.")

    cities = sorted(config.cities.keys())
    default_index = cities.index(config.default_city) if config.default_city in cities else 0

    with st.sidebar:
        st.header("Controls")
        preset_city = st.selectbox("Quick pick", cities, index=default_index)
        custom_city = st.text_input(
            "Or type any city",
            value=st.session_state.get("custom_city_input", ""),
            placeholder="Hyderabad, Paris, Tokyo, Singapore...",
        )
        selected_city = custom_city.strip() or preset_city
        refresh = st.button("Refresh live data", use_container_width=True, type="primary")
        st.caption(f"Current target: {selected_city}")
        st.markdown(
            "Eco-Pulse combines live telemetry with a document-backed knowledge layer to answer practical questions."
        )

    try:
        if refresh or "snapshot" not in st.session_state or st.session_state.get("city") != selected_city:
            snapshot = pipeline.ingest_city(selected_city)
            history = pipeline.load_city_history(snapshot["city"])
            st.session_state["snapshot"] = snapshot
            st.session_state["history"] = history
            st.session_state["city"] = selected_city
            st.session_state["custom_city_input"] = custom_city.strip()
        else:
            snapshot = st.session_state["snapshot"]
            history = st.session_state["history"]
    except ValueError as exc:
        st.error(str(exc))
        return

    render_hero(snapshot)

    left, middle, right = st.columns(3)
    with left:
        metric_card("AQI Category", snapshot["aqi_category"], "Derived from PM2.5 concentration.")
        metric_card("PM2.5", f'{snapshot["pm2_5"]:.1f} ug/m3', "Fine particulate matter concentration.")
    with middle:
        metric_card("Temperature", f'{snapshot["temperature_2m"]:.1f} C', "Live air temperature.")
        metric_card("Humidity", f'{snapshot["relative_humidity_2m"]:.0f}%', "Relative humidity reading.")
    with right:
        metric_card("Wind Speed", f'{snapshot["wind_speed_10m"]:.1f} km/h', "Surface wind speed.")
        metric_card("UV Index", f'{snapshot["uv_index"]:.1f}', "Ultraviolet exposure risk.")

    forecast_summary = snapshot.get("forecast_summary") or {}
    if forecast_summary:
        best_time_col, score_col, pm_col = st.columns(3)
        with best_time_col:
            metric_card("Best Upcoming Hour", str(forecast_summary["best_time"]), "Lowest forecast exposure window.")
        with score_col:
            metric_card(
                "Forecast Exposure",
                f'{forecast_summary["best_exposure_score"]:.1f}',
                "Lower is better for outdoor comfort and exposure.",
            )
        with pm_col:
            metric_card(
                "Best-Hour PM2.5",
                f'{forecast_summary["best_pm2_5"]:.1f} ug/m3',
                "Forecast fine particulate concentration for the best window.",
            )

    render_insight_cards(snapshot)

    st.subheader(f'Live Snapshot for {snapshot["city"]}')
    snapshot_table = pd.DataFrame([snapshot]).drop(columns=["raw_payload", "forecast"], errors="ignore")
    st.dataframe(snapshot_table, use_container_width=True)

    render_timeseries(history)
    render_forecast_outlook(snapshot)

    st.subheader("Ask Eco-Pulse")
    user_query = st.text_input(
        "Try questions like: When should I go for a run today? or Is it safe to commute by bike this evening?",
        value="What is the best time today for a 30-minute outdoor walk with low pollution exposure?",
    )

    if st.button("Generate recommendation", use_container_width=True):
        with st.spinner("Synthesizing live data with environmental guidance..."):
            response = rag.answer_question(selected_city, snapshot, user_query)
        st.markdown(
            f"""
            <div class="recommendation-card">
                <div class="recommendation-title">Recommendation</div>
                <div>{response["answer"]}</div>
                <div class="recommendation-meta">
                    Forecast-aware guidance is blended with retrieved environmental evidence for this answer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if response.get("used_fallback"):
            st.caption("Using the resilient local recommendation engine because the hosted LLM path is unavailable.")
        render_evidence(response["evidence"])


if __name__ == "__main__":
    main()
