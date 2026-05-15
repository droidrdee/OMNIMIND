"""OMNIMIND FastAPI application entry point.

Initializes all heavy singletons (OCR, chunker, embedder, vector store,
reranker, RAG pipeline, web/reasoning agents, router) inside a lifespan
context so they are ready before the first request and cleanly torn
down on shutdown.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure imports work whether launched via ``uvicorn main:app`` from inside
# the ``backend`` directory or as part of a wider package import.
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from loguru import logger  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.logger import logger as _logger  # noqa: E402,F401  (configure logger)
from core.router import QueryRouter  # noqa: E402
from ingestion.ocr import OCRPipeline  # noqa: E402
from ingestion.chunker import DocumentChunker  # noqa: E402
from ingestion.embedder import EmbeddingModel  # noqa: E402
from retrieval.vector_store import ChromaVectorStore  # noqa: E402
from retrieval.reranker import Reranker  # noqa: E402
from retrieval.rag_pipeline import RAGPipeline  # noqa: E402
from agents.web_search_agent import WebSearchAgent  # noqa: E402
from agents.reasoning_agent import ReasoningAgent  # noqa: E402
from api.routes import health, ingest, query  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down OMNIMIND service-wide singletons."""
    settings = get_settings()
    logger.info("Initializing OMNIMIND backend...")

    app.state.settings = settings
    app.state.ocr = OCRPipeline()
    app.state.chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    app.state.embedder = EmbeddingModel(model_name=settings.embedding_model_name)
    app.state.vector_store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection_name,
    )
    app.state.reranker = Reranker()
    app.state.rag = RAGPipeline(
        vector_store=app.state.vector_store,
        embedder=app.state.embedder,
        reranker=app.state.reranker,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        top_k=settings.top_k_retrieval,
    )
    app.state.web_agent = WebSearchAgent(
        tavily_api_key=settings.tavily_api_key,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )
    app.state.reasoning_agent = ReasoningAgent(
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )
    app.state.router = QueryRouter(
        rag_pipeline=app.state.rag,
        web_agent=app.state.web_agent,
        reasoning_agent=app.state.reasoning_agent,
        embedder=app.state.embedder,
        vector_store=app.state.vector_store,
        reranker=app.state.reranker,
        confidence_threshold=settings.retrieval_confidence_threshold,
        live_keywords=settings.live_query_keywords,
    )
    logger.info("OMNIMIND backend started")
    try:
        yield
    finally:
        logger.info("OMNIMIND backend shutting down")


app = FastAPI(
    title="OMNIMIND API",
    version="1.0.0",
    description="Multi-agent AI research platform with intelligent query routing",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])


@app.get("/")
async def root() -> dict:
    """Service banner endpoint."""
    return {"service": "OMNIMIND", "version": "1.0.0", "docs": "/docs"}
