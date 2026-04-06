from __future__ import annotations

from langchain_core.documents import Document
from unittest.mock import patch

from ecopulse.config import AppConfig
from ecopulse.rag import EcoPulseRAG


class StubKnowledgeBase:
    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        return [
            Document(
                page_content="Prefer early morning activity when particulate pollution is moderate.",
                metadata={"source": "health_advisory.md"},
            ),
            Document(
                page_content="Shift commutes away from heavy roadside congestion when PM2.5 is elevated.",
                metadata={"source": "urban_mobility.md"},
            ),
        ]


def build_config() -> AppConfig:
    return AppConfig(
        project_root=None,
        chroma_dir=None,
        bronze_dir=None,
        silver_dir=None,
        knowledge_dir=None,
        default_city="Delhi",
        openai_api_key=None,
        cities={"Delhi": {"latitude": 28.6139, "longitude": 77.2090}},
    )


def test_answer_question_returns_fallback_answer_and_evidence() -> None:
    rag = EcoPulseRAG(build_config(), StubKnowledgeBase())
    snapshot = {
        "aqi_category": "Moderate",
        "pm2_5": 26.0,
        "temperature_2m": 30.0,
        "relative_humidity_2m": 68.0,
        "wind_speed_10m": 8.0,
        "uv_index": 7.5,
        "exposure_score": 22.4,
        "forecast_summary": {
            "best_time": "2026-04-06T06:00:00+00:00",
            "best_exposure_score": 10.5,
            "best_pm2_5": 16.0,
            "best_uv_index": 2.0,
            "best_aqi_category": "Moderate",
        },
    }

    response = rag.answer_question(
        "Delhi",
        snapshot,
        "Is this a good time for an outdoor walk?",
    )

    assert "Delhi" in response["answer"]
    assert "PM2.5" in response["answer"]
    assert "best upcoming window" in response["answer"] or "best upcoming" in response["answer"].lower()
    assert len(response["evidence"]) == 2
    assert response["evidence"][0]["source"] == "health_advisory.md"


def test_snapshot_text_formats_key_fields() -> None:
    snapshot = {
        "aqi_category": "Good",
        "pm2_5": 9.2,
        "temperature_2m": 24.3,
        "relative_humidity_2m": 55.0,
        "wind_speed_10m": 10.2,
        "uv_index": 3.1,
        "exposure_score": 8.8,
        "forecast_summary": {
            "best_time": "2026-04-06T07:00:00+00:00",
            "best_exposure_score": 7.2,
            "best_pm2_5": 8.4,
            "best_uv_index": 1.8,
            "best_aqi_category": "Good",
        },
    }

    text = EcoPulseRAG._snapshot_text(snapshot)

    assert "AQI category: Good" in text
    assert "PM2.5: 9.2 ug/m3" in text
    assert "exposure score: 8.8" in text

    forecast_text = EcoPulseRAG._forecast_text(snapshot)
    assert "Best upcoming hour" in forecast_text
    assert "8.4 ug/m3" in forecast_text


def test_answer_question_falls_back_when_llm_errors() -> None:
    rag = EcoPulseRAG(build_config(), StubKnowledgeBase())
    rag.llm = object()
    snapshot = {
        "aqi_category": "Moderate",
        "pm2_5": 26.0,
        "temperature_2m": 30.0,
        "relative_humidity_2m": 68.0,
        "wind_speed_10m": 8.0,
        "uv_index": 7.5,
        "exposure_score": 22.4,
        "forecast_summary": {
            "best_time": "2026-04-06T06:00:00+00:00",
            "best_exposure_score": 10.5,
            "best_pm2_5": 16.0,
            "best_uv_index": 2.0,
            "best_aqi_category": "Moderate",
        },
    }

    with patch.object(EcoPulseRAG, "_generate_answer", return_value=("fallback answer", True)):
        response = rag.answer_question("Delhi", snapshot, "Is this a good time for a walk?")

    assert response["answer"] == "fallback answer"
    assert response["used_fallback"] is True
