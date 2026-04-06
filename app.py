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


def render_evidence(evidence: list[dict[str, str]]) -> None:
    st.subheader("Retrieved Evidence")
    for item in evidence:
        source = item.get("source", "Unknown source")
        content = item.get("content", "").strip()
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
        selected_city = st.selectbox("City", cities, index=default_index)
        refresh = st.button("Refresh live data", use_container_width=True, type="primary")
        st.markdown(
            "Eco-Pulse combines live telemetry with a document-backed knowledge layer to answer practical questions."
        )

    if refresh or "snapshot" not in st.session_state or st.session_state.get("city") != selected_city:
        snapshot = pipeline.ingest_city(selected_city)
        history = pipeline.load_city_history(selected_city)
        st.session_state["snapshot"] = snapshot
        st.session_state["history"] = history
        st.session_state["city"] = selected_city
    else:
        snapshot = st.session_state["snapshot"]
        history = st.session_state["history"]

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

    st.subheader(f"Live Snapshot for {selected_city}")
    snapshot_table = pd.DataFrame([snapshot]).drop(columns=["raw_payload"])
    st.dataframe(snapshot_table, use_container_width=True)

    render_timeseries(history)

    st.subheader("Ask Eco-Pulse")
    user_query = st.text_input(
        "Try questions like: When should I go for a run today? or Is it safe to commute by bike this evening?",
        value="What is the best time today for a 30-minute outdoor walk with low pollution exposure?",
    )

    if st.button("Generate recommendation", use_container_width=True):
        with st.spinner("Synthesizing live data with environmental guidance..."):
            response = rag.answer_question(selected_city, snapshot, user_query)
        st.markdown("### Recommendation")
        st.write(response["answer"])
        render_evidence(response["evidence"])


if __name__ == "__main__":
    main()
