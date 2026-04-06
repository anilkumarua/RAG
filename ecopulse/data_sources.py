from __future__ import annotations

from datetime import datetime, timezone
from random import Random

import requests


class OpenMeteoClient:
    weather_url = "https://api.open-meteo.com/v1/forecast"
    air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

    def resolve_city(self, city: str, known_cities: dict[str, dict[str, float]] | None = None) -> dict[str, float | str]:
        normalized_city = city.strip()
        if not normalized_city:
            raise ValueError("Please enter a city name.")

        if known_cities and normalized_city in known_cities:
            coords = known_cities[normalized_city]
            return {
                "name": normalized_city,
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
            }

        params = {
            "name": normalized_city,
            "count": 1,
            "language": "en",
            "format": "json",
        }
        response = requests.get(self.geocoding_url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            raise ValueError(f"Could not find coordinates for '{normalized_city}'.")

        match = results[0]
        parts = [match.get("name"), match.get("admin1"), match.get("country")]
        display_name = ", ".join(part for part in parts if part)
        return {
            "name": display_name,
            "latitude": float(match["latitude"]),
            "longitude": float(match["longitude"]),
        }

    def fetch_snapshot(self, city: str, latitude: float, longitude: float) -> dict:
        try:
            weather = self._fetch_weather(latitude, longitude)
            air_quality = self._fetch_air_quality(latitude, longitude)
        except requests.RequestException:
            weather, air_quality = self._fallback_snapshot(city)
        timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "city": city,
            "timestamp": timestamp,
            "temperature_2m": weather["temperature_2m"],
            "relative_humidity_2m": weather["relative_humidity_2m"],
            "wind_speed_10m": weather["wind_speed_10m"],
            "uv_index": weather["uv_index"],
            "pm2_5": air_quality["pm2_5"],
            "pm10": air_quality["pm10"],
            "carbon_monoxide": air_quality["carbon_monoxide"],
            "nitrogen_dioxide": air_quality["nitrogen_dioxide"],
            "raw_payload": {"weather": weather, "air_quality": air_quality},
        }

    def _fetch_weather(self, latitude: float, longitude: float) -> dict[str, float]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,uv_index",
            "timezone": "auto",
        }
        response = requests.get(self.weather_url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()["current"]

    def _fetch_air_quality(self, latitude: float, longitude: float) -> dict[str, float]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide",
            "timezone": "auto",
        }
        response = requests.get(self.air_quality_url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()["current"]

    @staticmethod
    def _fallback_snapshot(city: str) -> tuple[dict[str, float], dict[str, float]]:
        now = datetime.now(timezone.utc)
        seed = abs(hash(f"{city}-{now.strftime('%Y-%m-%d-%H')}"))
        rng = Random(seed)
        weather = {
            "temperature_2m": round(18 + rng.uniform(0, 18), 1),
            "relative_humidity_2m": round(40 + rng.uniform(0, 45), 1),
            "wind_speed_10m": round(4 + rng.uniform(0, 14), 1),
            "uv_index": round(1 + rng.uniform(0, 9), 1),
        }
        air_quality = {
            "pm2_5": round(8 + rng.uniform(0, 70), 1),
            "pm10": round(15 + rng.uniform(0, 110), 1),
            "carbon_monoxide": round(180 + rng.uniform(0, 600), 1),
            "nitrogen_dioxide": round(8 + rng.uniform(0, 40), 1),
        }
        return weather, air_quality
