"""Tests for the QueryRouter.

These tests stub every downstream dependency (RAG pipeline, web agent,
reasoning agent, embedder, vector store, reranker) so that the router's
decision logic can be exercised in pure isolation. The goal is to lock down
the routing matrix:

* very recent / live queries  -> web_search
* high vector confidence       -> rag
* medium vector confidence     -> reasoning
* low vector confidence        -> web_search
* complex/comparative queries  -> reasoning (regardless of vector score)
* explicit ``force_route``     -> always honoured
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Ensure backend modules are importable.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def make_router(
    top_score: float = 0.9,
    rag_answer: str = "rag answer",
    web_answer: str = "web answer",
    reasoning_answer: str = "reasoning answer",
) -> Any:
    """Build a QueryRouter wired up with deterministic mock collaborators."""
    from core.router import QueryRouter

    rag_pipeline = MagicMock()
    rag_pipeline.query.return_value = {
        "answer": rag_answer,
        "sources": ["a"],
        "route": "rag",
        "confidence": top_score,
    }

    web_agent = MagicMock()
    web_agent.run.return_value = {
        "answer": web_answer,
        "sources": [],
        "route": "web_search",
    }

    reasoning_agent = MagicMock()
    reasoning_agent.run.return_value = {
        "answer": reasoning_answer,
        "sources": [],
        "route": "reasoning",
    }

    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 384

    vector_store = MagicMock()
    vector_store.query.return_value = {
        "results": [
            {
                "text": "x",
                "metadata": {"source": "a"},
                "distance": 1 - top_score,
                "score": top_score,
            }
        ],
        "top_score": top_score,
    }

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, passages, **kwargs: passages

    router = QueryRouter(
        rag_pipeline=rag_pipeline,
        web_agent=web_agent,
        reasoning_agent=reasoning_agent,
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
    )
    # Expose mocks for assertions.
    router._mock_rag = rag_pipeline
    router._mock_web = web_agent
    router._mock_reasoning = reasoning_agent
    return router


def test_live_query_routes_to_web_search() -> None:
    router = make_router(top_score=0.9)
    result = router.route("What is the latest AI news today?")
    assert result["route"] == "web_search"
    router._mock_web.run.assert_called_once()


def test_high_confidence_routes_to_rag() -> None:
    router = make_router(top_score=0.9)
    result = router.route("What is X?")
    assert result["route"] == "rag"
    router._mock_rag.query.assert_called_once()


def test_medium_confidence_routes_to_reasoning() -> None:
    router = make_router(top_score=0.6)
    result = router.route("What is X?")
    assert result["route"] == "reasoning"
    router._mock_reasoning.run.assert_called_once()


def test_low_confidence_routes_to_web_search() -> None:
    router = make_router(top_score=0.2)
    result = router.route("What is X?")
    assert result["route"] == "web_search"
    router._mock_web.run.assert_called_once()


def test_complex_query_routes_to_reasoning() -> None:
    router = make_router(top_score=0.5)
    query = "Compare and contrast neural networks and decision trees in terms of interpretability"
    result = router.route(query)
    assert result["route"] == "reasoning"
    router._mock_reasoning.run.assert_called_once()


def test_force_route_override() -> None:
    router = make_router(top_score=0.95)
    result = router.route("Anything goes here", force_route="reasoning")
    assert result["route"] == "reasoning"
    router._mock_reasoning.run.assert_called_once()
    router._mock_rag.query.assert_not_called()
    router._mock_web.run.assert_not_called()


def test_analytics_counter_increments() -> None:
    router = make_router(top_score=0.9)
    for _ in range(3):
        router.route("What is X?")
    analytics = router.get_analytics()
    assert analytics["total_queries"] == 3
    assert sum(analytics["routing_counter"].values()) == 3
    assert analytics["routing_counter"]["rag"] == 3
