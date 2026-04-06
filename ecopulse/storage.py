from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_parquet(df: pd.DataFrame, directory: Path, filename: str) -> Path:
    ensure_directory(directory)
    path = directory / filename
    if path.exists():
        existing = pd.read_parquet(path)
        merged = pd.concat([existing, df], ignore_index=True)
    else:
        merged = df
    merged.to_parquet(path, index=False)
    return path
