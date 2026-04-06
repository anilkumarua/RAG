from __future__ import annotations

from langchain_core.documents import Document

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
    }

    response = rag.answer_question(
        "Delhi",
        snapshot,
        "Is this a good time for an outdoor walk?",
    )

    assert "Delhi" in response["answer"]
    assert "PM2.5" in response["answer"]
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
    }

    text = EcoPulseRAG._snapshot_text(snapshot)

    assert "AQI category: Good" in text
    assert "PM2.5: 9.2 ug/m3" in text
    assert "exposure score: 8.8" in text
