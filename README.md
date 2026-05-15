# OMNIMIND

OMNIMIND is a production-grade multi-agent AI research platform that combines a FastAPI backend, a Streamlit frontend, a LlamaIndex + ChromaDB retrieval pipeline, and a LangChain-driven web search agent behind an intelligent query router.

## Overview

OMNIMIND answers complex research questions by dynamically choosing the best execution path:

- **RAG Agent** for knowledge grounded in your own ingested documents.
- **Web Search Agent** (LangChain + Tavily) for live, time-sensitive information.
- **Reasoning Agent** for synthesis, multi-step logic, and open-ended questions.

A central **Query Router** inspects each incoming question, scores retrieval confidence, scans for "live" keywords, and routes execution accordingly.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI, Uvicorn, Pydantic v2 |
| Frontend | Streamlit |
| RAG | LlamaIndex, ChromaDB, sentence-transformers |
| Agents | LangChain, LangChain-OpenAI, Tavily |
| LLM | OpenAI GPT-4o, `text-embedding-3-small` |
| OCR | PyMuPDF, Tesseract (pytesseract), Pillow |
| Reranking | FlashRank |
| Fine-Tuning | QLoRA (PEFT + bitsandbytes) |
| Observability | Loguru |
| Packaging | Docker, Docker Compose |
| CI/CD | GitHub Actions, AWS ECR/ECS |

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url> omnimind
cd omnimind

# 2. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY and TAVILY_API_KEY

# 3. Launch the full stack
docker compose up --build

# 4. Open the UI
# Streamlit:  http://localhost:8501
# FastAPI:    http://localhost:8080/docs
# ChromaDB:   http://localhost:8000
```

## Architecture

```
                         +---------------------+
                         |   Streamlit UI      |
                         +----------+----------+
                                    |
                                    v
                         +---------------------+
                         |   FastAPI Backend   |
                         +----------+----------+
                                    |
                                    v
                         +---------------------+
                         |   Query Router      |
                         +----------+----------+
                  /-----------------+-----------------\
                  v                 v                 v
        +-----------------+ +---------------+ +----------------+
        |   RAG Agent     | |   Web Agent   | | Reasoning Agent|
        | LlamaIndex +    | | LangChain +   | |  GPT-4o        |
        | ChromaDB        | | Tavily        | |                |
        +-----------------+ +---------------+ +----------------+
```

### Routing Logic

1. If the query contains a **live keyword** (e.g. `today`, `latest`, `2025`), route to the **Web Search Agent**.
2. Otherwise, probe the vector store. If top-K retrieval confidence is above `RETRIEVAL_CONFIDENCE_THRESHOLD`, route to **RAG**.
3. Else, fall back to the **Reasoning Agent**.

## Features

- **OCR-Enhanced Ingestion** — PDFs are parsed with PyMuPDF; image-only pages are transparently OCR'd via Tesseract.
- **Hybrid Retrieval** — dense embeddings (MiniLM) with FlashRank reranking for precision.
- **QLoRA Fine-Tuning** — scripts to fine-tune open-weights models on domain corpora.
- **CI/CD** — GitHub Actions build, test, and push images to AWS ECR; deploy to ECS Fargate.
- **Observability** — structured Loguru logs to stdout and rotating files.
- **Type-Safe** — Pydantic v2 request/response contracts.

## Project Layout

```
omnimind/
  backend/
    api/          # FastAPI routes & middleware
    core/         # Settings & logging
    ingestion/    # PDF/OCR/chunking pipeline
    retrieval/    # Vector store + reranker
    agents/       # Router, RAG, Web, Reasoning
    models/       # Pydantic models
    tests/        # Pytest suite
  frontend/       # Streamlit app
  docker-compose.yml
```

## Development

```bash
# Install backend deps locally
pip install -r backend/requirements.txt

# Run the API outside Docker
uvicorn backend.api.main:app --reload --port 8080

# Run tests
pytest backend/tests
```

## Deployment

The repo ships with a GitHub Actions workflow that builds the backend and frontend Docker images, pushes them to AWS ECR, and rolls out a new task revision on ECS Fargate. See `.github/workflows/` for details.

## License

MIT
