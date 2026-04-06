from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd

from ecopulse.config import AppConfig
from ecopulse.data_sources import OpenMeteoClient
from ecopulse.storage import append_parquet, ensure_directory

try:
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    DELTA_AVAILABLE = True
except Exception:
    SparkSession = None
    DELTA_AVAILABLE = False


AQI_BANDS = [
    (12.0, "Good"),
    (35.4, "Moderate"),
    (55.4, "Unhealthy for Sensitive Groups"),
    (150.4, "Unhealthy"),
    (250.4, "Very Unhealthy"),
    (float("inf"), "Hazardous"),
]


class EcoPulsePipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = OpenMeteoClient()
        ensure_directory(config.bronze_dir)
        ensure_directory(config.silver_dir)
        self.spark = None

    def ingest_city(self, city: str) -> dict:
        resolved_city = self.client.resolve_city(city, self.config.cities)
        city_name = str(resolved_city["name"])
        city_key = self._city_storage_key(city_name)
        latitude = float(resolved_city["latitude"])
        longitude = float(resolved_city["longitude"])
        snapshot = self.client.fetch_snapshot(
            city_name,
            latitude,
            longitude,
        )
        forecast = self.client.fetch_hourly_forecast(city_name, latitude, longitude)
        snapshot["aqi_category"] = self._aqi_category(snapshot["pm2_5"])
        snapshot["city_key"] = city_key
        snapshot["forecast"] = self._prepare_forecast(forecast)
        snapshot["forecast_summary"] = self._summarize_forecast(snapshot["forecast"])

        bronze_df = pd.DataFrame([snapshot])
        bronze_df["forecast"] = bronze_df["forecast"].map(json.dumps)
        bronze_df["forecast_summary"] = bronze_df["forecast_summary"].map(json.dumps)
        bronze_df["raw_payload"] = bronze_df["raw_payload"].map(json.dumps)
        append_parquet(bronze_df, self.config.bronze_dir, f"{city_key}_bronze.parquet")

        silver_record = bronze_df.copy()
        silver_record["exposure_score"] = silver_record.apply(self._exposure_score, axis=1)
        append_parquet(silver_record, self.config.silver_dir, f"{city_key}_silver.parquet")

        if self.config.enable_spark:
            self._write_with_spark(bronze_df, self.config.bronze_dir / f"{city_key}_delta")
            self._write_with_spark(silver_record, self.config.silver_dir / f"{city_key}_delta")

        snapshot["raw_payload"] = json.loads(bronze_df.iloc[0]["raw_payload"])
        snapshot["forecast"] = json.loads(bronze_df.iloc[0]["forecast"])
        snapshot["forecast_summary"] = json.loads(bronze_df.iloc[0]["forecast_summary"])
        snapshot["exposure_score"] = float(silver_record.iloc[0]["exposure_score"])
        return snapshot

    def load_city_history(self, city: str) -> pd.DataFrame:
        target = self.config.silver_dir / f"{self._city_storage_key(city)}_silver.parquet"
        if not target.exists():
            return pd.DataFrame()
        df = pd.read_parquet(target)
        df["raw_payload"] = df["raw_payload"].map(json.loads)
        if "forecast" in df.columns:
            df["forecast"] = df["forecast"].map(json.loads)
        if "forecast_summary" in df.columns:
            df["forecast_summary"] = df["forecast_summary"].map(json.loads)
        return df.sort_values("timestamp").tail(24)

    def ingest_all_cities(self) -> list[dict]:
        return [self.ingest_city(city) for city in self.config.cities]

    @staticmethod
    def _city_storage_key(city: str) -> str:
        key = re.sub(r"[^a-z0-9]+", "_", city.strip().lower()).strip("_")
        return key or "city"

    def _build_spark_session(self):
        if not DELTA_AVAILABLE or SparkSession is None:
            return None
        builder = (
            SparkSession.builder.appName("EcoPulse")
            .master("local[*]")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        try:
            return configure_spark_with_delta_pip(builder).getOrCreate()
        except Exception:
            return None

    def _write_with_spark(self, df: pd.DataFrame, path: Path) -> None:
        if self.spark is None:
            self.spark = self._build_spark_session()
        if self.spark is None:
            return
        spark_df = self.spark.createDataFrame(df)
        format_name = "delta" if DELTA_AVAILABLE else "parquet"
        spark_df.write.format(format_name).mode("append").save(str(path))

    @staticmethod
    def _aqi_category(pm2_5: float) -> str:
        for threshold, label in AQI_BANDS:
            if pm2_5 <= threshold:
                return label
        return "Unknown"

    @staticmethod
    def _exposure_score(row: pd.Series) -> float:
        pollution_weight = min(float(row["pm2_5"]) / 150.0, 1.0) * 60
        uv_weight = min(float(row["uv_index"]) / 12.0, 1.0) * 20
        wind_bonus = max(0.0, min(float(row["wind_speed_10m"]) / 20.0, 1.0)) * 10
        humidity_penalty = abs(float(row["relative_humidity_2m"]) - 50.0) / 50.0 * 10
        return round(pollution_weight + uv_weight + humidity_penalty - wind_bonus, 2)

    def _prepare_forecast(self, forecast: list[dict]) -> list[dict]:
        if not forecast:
            return []
        df = pd.DataFrame(forecast)
        df["exposure_score"] = df.apply(self._exposure_score, axis=1)
        df["aqi_category"] = df["pm2_5"].map(self._aqi_category)
        return df.to_dict(orient="records")

    @staticmethod
    def _summarize_forecast(forecast: list[dict]) -> dict:
        if not forecast:
            return {}
        ranked = sorted(forecast, key=lambda item: (float(item["exposure_score"]), float(item["pm2_5"])))
        best = ranked[0]
        return {
            "best_time": best["timestamp"],
            "best_exposure_score": round(float(best["exposure_score"]), 2),
            "best_pm2_5": round(float(best["pm2_5"]), 1),
            "best_uv_index": round(float(best["uv_index"]), 1),
            "best_aqi_category": best["aqi_category"],
        }
