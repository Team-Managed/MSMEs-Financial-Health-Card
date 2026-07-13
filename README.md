# MSME Financial Health Card

Stress-tested MSME credit scoring. IDBI Innovate Track 03.

## Quick Start

### Backend

```bash
cd backend
uv sync
cp .env.example .env          # add your GOOGLE_API_KEY
# Optional: add PDFs to backend/app/rag/corpus/ then run:
# uv run --project backend python -m app.rag.build_index

# Start from project root (keeps backend.app.* imports on the path):
cd ..
uv run --project backend uvicorn backend.app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev                      # → http://localhost:3000
```

## Running Tests

```bash
cd backend
uv run pytest tests/ -v --ignore=tests/test_integration.py
# With API key:
GOOGLE_API_KEY=... uv run pytest tests/test_integration.py -v
```

## Build Priority (time-box)

1. Synthetic data + Risk Engine + frontend shell — non-negotiable core
2. RAG index + weight-setter — differentiator layer
3. If RAG isn't stable before demo: default weights are already wired in as fallback
