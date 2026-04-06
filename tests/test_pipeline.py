from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pandas as pd

from ecopulse.config import AppConfig
from ecopulse.pipeline import EcoPulsePipeline


class StubClient:
    def fetch_snapshot(self, city: str, latitude: float, longitude: float) -> dict:
        return {
            "city": city,
            "timestamp": "2026-04-06T12:00:00+00:00",
            "temperature_2m": 29.5,
            "relative_humidity_2m": 62.0,
            "wind_speed_10m": 11.0,
            "uv_index": 5.5,
            "pm2_5": 28.0,
            "pm10": 52.0,
            "carbon_monoxide": 320.0,
            "nitrogen_dioxide": 18.0,
            "raw_payload": {
                "weather": {"source": "stub"},
                "air_quality": {"source": "stub"},
            },
        }


def build_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        project_root=tmp_path,
        chroma_dir=tmp_path / "chroma",
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        knowledge_dir=tmp_path / "knowledge",
        default_city="Delhi",
        cities={"Delhi": {"latitude": 28.6139, "longitude": 77.2090}},
    )


def test_ingest_city_writes_bronze_and_silver_records() -> None:
    root = Path.cwd() / "test_artifacts" / f"pipeline_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        config = build_config(root)
        pipeline = EcoPulsePipeline(config)
        pipeline.client = StubClient()

        snapshot = pipeline.ingest_city("Delhi")

        bronze_path = config.bronze_dir / "delhi_bronze.parquet"
        silver_path = config.silver_dir / "delhi_silver.parquet"

        assert bronze_path.exists()
        assert silver_path.exists()
        assert snapshot["aqi_category"] == "Moderate"
        assert "exposure_score" in snapshot

        history = pipeline.load_city_history("Delhi")
        assert len(history) == 1
        assert history.iloc[0]["city"] == "Delhi"
        assert isinstance(history.iloc[0]["raw_payload"], dict)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_exposure_score_increases_with_pollution_and_uv() -> None:
    cleaner = pd.Series(
        {
            "pm2_5": 10.0,
            "uv_index": 2.0,
            "wind_speed_10m": 12.0,
            "relative_humidity_2m": 50.0,
        }
    )
    harsher = pd.Series(
        {
            "pm2_5": 80.0,
            "uv_index": 9.0,
            "wind_speed_10m": 3.0,
            "relative_humidity_2m": 80.0,
        }
    )

    assert EcoPulsePipeline._exposure_score(harsher) > EcoPulsePipeline._exposure_score(cleaner)
