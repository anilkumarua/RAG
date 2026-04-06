from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecopulse.config import AppConfig
from ecopulse.knowledge_base import KnowledgeBase
from ecopulse.pipeline import EcoPulsePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest live or fallback environmental telemetry into Eco-Pulse bronze/silver layers."
    )
    parser.add_argument("--city", help="Single city to ingest, for example Delhi.")
    parser.add_argument(
        "--all-cities",
        action="store_true",
        help="Ingest every city configured in ecopulse.config.AppConfig.",
    )
    parser.add_argument(
        "--bootstrap-kb",
        action="store_true",
        help="Ensure the Chroma knowledge index is initialized before ingestion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.from_env()

    if args.bootstrap_kb:
        KnowledgeBase(config).ensure_index()

    pipeline = EcoPulsePipeline(config)

    if args.all_cities:
        targets = list(config.cities.keys())
    elif args.city:
        if args.city not in config.cities:
            print(f"Unknown city '{args.city}'. Available cities: {', '.join(sorted(config.cities))}")
            return 1
        targets = [args.city]
    else:
        targets = [config.default_city]

    summaries = []
    for city in targets:
        snapshot = pipeline.ingest_city(city)
        summaries.append(
            {
                "city": city,
                "timestamp": snapshot["timestamp"],
                "aqi_category": snapshot["aqi_category"],
                "pm2_5": snapshot["pm2_5"],
                "temperature_2m": snapshot["temperature_2m"],
                "exposure_score": snapshot["exposure_score"],
            }
        )

    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
