# Eco-Pulse

Eco-Pulse is a smart-city environmental intelligence dashboard that combines live urban air-quality and weather feeds with retrieval-augmented generation (RAG) over local policy and health guidance documents.

## What it does

- Ingests real-time weather and air-quality observations from Open-Meteo APIs.
- Processes environmental records through a Spark-ready pipeline with Delta-style storage helpers.
- Indexes regulations, advisories, and planning notes into ChromaDB.
- Answers human-centric questions with contextual recommendations grounded in both live data and retrieved evidence.
- Exposes everything in a Streamlit dashboard for analysts, residents, and city planners.

## Architecture

1. `ecopulse/data_sources.py` fetches virtual IoT readings from REST APIs.
2. `ecopulse/pipeline.py` converts raw readings into Spark/Pandas tables and persists them in bronze/silver layers.
3. `ecopulse/knowledge_base.py` chunks and indexes domain documents into ChromaDB.
4. `ecopulse/rag.py` fuses live telemetry with retrieved knowledge to generate recommendations.
5. `app.py` renders the interactive dashboard.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

If you set `OPENAI_API_KEY`, Eco-Pulse will use an OpenAI chat model through LangChain. Without a key, the app still runs using a deterministic retrieval pipeline and a rule-based narrative generator for demo purposes.
Set `ENABLE_SPARK=true` when you want to activate the heavier Spark/Delta write path locally.

For local development with tests:

```bash
pip install -r requirements-dev.txt
```

For local Spark and Delta support:

```bash
pip install -r requirements-spark.txt
```

## Key features

- AQI-aware commuting and outdoor-activity guidance
- Health-sensitive advice tied to real-time PM2.5, UV, humidity, and wind
- Source-backed answers with retrieved urban policy snippets
- Spark session helper with Delta Lake configuration when available
- Local-first demo data for fast evaluation
- Docker and Docker Compose support for reproducible local runs
- Standalone ingestion job for scheduled bronze/silver refreshes
- Deployment manifests for Streamlit Community Cloud and generic container platforms

## Project layout

```text
.
|-- app.py
|-- Dockerfile
|-- docker-compose.yml
|-- Procfile
|-- runtime.txt
|-- requirements.txt
|-- requirements-dev.txt
|-- requirements-spark.txt
|-- .env.example
|-- .streamlit/
|   `-- config.toml
|-- knowledge_corpus/
|   |-- health_advisory.md
|   |-- urban_mobility.md
|   `-- air_quality_regulation.md
|-- scripts/
|   `-- run_ingestion.py
`-- ecopulse/
    |-- __init__.py
    |-- config.py
    |-- data_sources.py
    |-- pipeline.py
    |-- knowledge_base.py
    |-- rag.py
    `-- storage.py
```

## Notes

- The ingestion layer uses Open-Meteo because it provides free weather and air-quality APIs suitable for prototypes and classroom demos.
- Delta persistence is attempted when `delta-spark` is installed correctly; otherwise the pipeline falls back to Parquet while keeping the same bronze/silver layout.
- Spark startup is disabled by default for fast local development and tests; enable it explicitly with `ENABLE_SPARK=true` when you want the Delta-backed path.
- The knowledge corpus ships with seed documents, and you can drop additional Markdown or text files into `knowledge_corpus/` before relaunching the app.

## Docker

```bash
docker compose up --build
```

The app will be available at `http://localhost:8501`. The Compose file mounts the project directory so the `data/` folder and knowledge corpus stay editable during development.

## Scheduled ingestion job

You can refresh one or more cities without opening the dashboard:

```bash
python scripts/run_ingestion.py --city Delhi
python scripts/run_ingestion.py --all-cities
```

This creates or appends bronze/silver records under `data/` and prints a compact summary for each ingested city. The job uses the same fallback behavior as the app if live APIs are temporarily unavailable.

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app pointing to `app.py`.
3. Keep the default install command so Streamlit Cloud uses the lightweight `requirements.txt`.
4. Add secrets in the app settings if you want LLM-backed responses.

Recommended Streamlit secrets:

```toml
OPENAI_API_KEY="your_key_here"
OPENAI_MODEL="gpt-4o-mini"
DEFAULT_CITY="Delhi"
ENABLE_SPARK="false"
```

Notes:

- `.streamlit/config.toml` is included for a hosted-friendly Streamlit setup.
- Spark is intentionally off for Streamlit Cloud because the hosted environment is better suited to the lighter runtime path.
- The app will still support custom typed cities through the Open-Meteo geocoding API.
- If Streamlit Cloud deploys your app with Python 3.14 and you see a `protobuf` or `chromadb` import error, redeploy the app and choose Python 3.11 in Advanced settings. Streamlit Community Cloud documents that Python version is selected at deploy time from Advanced settings, and changing it later requires deleting and redeploying the app.

### Generic container platforms

Use the included `Dockerfile` and expose port `8501`. Suitable targets include Render, Railway, Azure Container Apps, and Google Cloud Run. The Docker image installs both the lightweight runtime dependencies and the optional Spark dependencies.

### Procfile-based hosts

The repository also includes a `Procfile` for platforms that expect a web process declaration.

## CI/CD

GitHub Actions workflows are included under `.github/workflows/`.

### Continuous integration

`ci.yml` runs on pushes and pull requests. It:

- installs Python and Java dependencies
- compiles the application modules
- runs a smoke test of the ingestion job
- verifies the Streamlit entry point imports cleanly

### Continuous delivery

`cd.yml` publishes a Docker image to GitHub Container Registry on pushes to `main` or `master`, on version tags like `v1.0.0`, and when triggered manually from the Actions tab.

Published image format:

```text
ghcr.io/<owner>/<repo>:latest
ghcr.io/<owner>/<repo>:<git-sha>
ghcr.io/<owner>/<repo>:<tag-or-branch>
```

### Required setup

1. Push the repository to GitHub.
2. Enable GitHub Actions for the repository.
3. If you want private package pulls, ensure the repository has permission to publish packages to GHCR.
4. If your deployment platform needs runtime secrets, add `OPENAI_API_KEY` there after the image is published.

No extra secret is required for GHCR publishing in the default GitHub Actions setup because the workflow uses the built-in `GITHUB_TOKEN`.
