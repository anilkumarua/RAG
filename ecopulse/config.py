from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    chroma_dir: Path
    bronze_dir: Path
    silver_dir: Path
    knowledge_dir: Path
    default_city: str
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    streamlit_port: int = 8501
    enable_spark: bool = False
    cities: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        root = Path(__file__).resolve().parent.parent
        cities = {
            "Delhi": {"latitude": 28.6139, "longitude": 77.2090},
            "Mumbai": {"latitude": 19.0760, "longitude": 72.8777},
            "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946},
            "London": {"latitude": 51.5072, "longitude": -0.1276},
            "New York": {"latitude": 40.7128, "longitude": -74.0060},
        }
        return cls(
            project_root=root,
            chroma_dir=root / os.getenv("CHROMA_DIR", "data/chroma"),
            bronze_dir=root / os.getenv("BRONZE_DIR", "data/bronze"),
            silver_dir=root / os.getenv("SILVER_DIR", "data/silver"),
            knowledge_dir=root / "knowledge_corpus",
            default_city=os.getenv("DEFAULT_CITY", "Delhi"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            streamlit_port=int(os.getenv("PORT", "8501")),
            enable_spark=os.getenv("ENABLE_SPARK", "false").lower() == "true",
            cities=cities,
        )
