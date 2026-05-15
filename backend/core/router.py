"""Intelligent query router: decides between RAG / Web Search / Reasoning based on query semantics + retrieval confidence."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger


class QueryRouter:
    """Brain of the OMNIMIND platform.

    Routes incoming queries to the most appropriate downstream agent
    (RAG, Web Search, or Reasoning) based on a combination of:

    * query semantics (live keywords, complexity markers)
    * retrieval confidence over the local knowledge base
    * optional forced routing override
    """

    def __init__(
        self,
        rag_pipeline,
        web_agent,
        reasoning_agent,
        embedder,
        vector_store,
        reranker,
        confidence_threshold: float = 0.75,
        low_threshold: float = 0.4,
        live_keywords: Optional[List[str]] = None,
    ) -> None:
        """Store agent dependencies and routing configuration.

        Args:
            rag_pipeline: A RAGPipeline instance with a ``query(str)`` method.
            web_agent: A WebSearchAgent instance with a ``run(str)`` method.
            reasoning_agent: A ReasoningAgent with ``run(query, context_chunks=...)``.
            embedder: Embedding model exposing ``embed_query(str)``.
            vector_store: Vector store exposing ``query(emb, top_k=...)``.
            reranker: Reranker exposing ``rerank(query, passages, top_n=...)``.
            confidence_threshold: Minimum top score to route directly to RAG.
            low_threshold: Minimum top score for "medium" confidence routing.
            live_keywords: Tokens indicating the query requires fresh web data.
        """
        self.rag_pipeline = rag_pipeline
        self.web_agent = web_agent
        self.reasoning_agent = reasoning_agent
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.confidence_threshold = confidence_threshold
        self.low_threshold = low_threshold

        current_year = datetime.now().year
        if live_keywords is None:
            live_keywords = [
                "today",
                "latest",
                "current",
                "now",
                "recent",
                "this week",
                "this year",
                str(current_year),
                str(current_year - 1),
            ]
        self.live_keywords = live_keywords

        self.routing_counter: Dict[str, int] = {
            "rag": 0,
            "web_search": 0,
            "reasoning": 0,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _is_live_query(self, query: str) -> bool:
        """Return True if the query likely needs fresh, real-time information."""
        lowered = query.lower()
        for kw in self.live_keywords:
            if kw.lower() in lowered:
                return True

        current_year = datetime.now().year
        match = re.search(r"\b(20\d{2})\b", query)
        if match:
            year = int(match.group(1))
            if abs(year - current_year) <= 1:
                return True
        return False

    def _is_complex_query(self, query: str) -> bool:
        """Return True if the query is long or contains analytical markers."""
        if len(query) > 150:
            return True
        complex_markers = [
            "compare",
            "contrast",
            "analyze",
            "explain why",
            "what if",
            "how does",
            "evaluate",
            "pros and cons",
        ]
        lowered = query.lower()
        return any(kw in lowered for kw in complex_markers)

    def _retrieve(self, query: str) -> dict:
        """Retrieve top passages and the top similarity score for a query."""
        query_emb = self.embedder.embed_query(query)
        res = self.vector_store.query(query_emb, top_k=5)
        passages = res.get("results", [])
        if len(passages) > 1:
            passages = self.reranker.rerank(query, passages, top_n=3)
        return {"passages": passages, "top_score": res.get("top_score", 0.0)}

    def _dispatch_forced(self, route: str, query: str) -> dict:
        """Dispatch the query to a specific agent without semantic routing."""
        if route == "rag":
            result = self.rag_pipeline.query(query)
        elif route == "web_search":
            result = self.web_agent.run(query)
        elif route == "reasoning":
            result = self.reasoning_agent.run(query)
        else:
            raise ValueError(f"Unknown forced route: {route}")
        self.routing_counter[route] += 1
        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def route(self, query: str, force_route: Optional[str] = None) -> dict:
        """Route a query to the most appropriate downstream agent.

        Args:
            query: The natural-language user query.
            force_route: Optional override – one of ``"rag"``, ``"web_search"``,
                ``"reasoning"``. When set, bypasses semantic routing.

        Returns:
            A dictionary containing at minimum ``answer``, ``sources``,
            ``route`` and ``routing_reason``.
        """
        valid_routes = {"rag", "web_search", "reasoning"}

        if force_route in valid_routes:
            logger.info(f"Forced route '{force_route}' for query: {query[:80]!r}")
            result = self._dispatch_forced(force_route, query)
            routing_reason = f"forced to {force_route}"
            chosen_route = force_route
        else:
            is_complex = self._is_complex_query(query)
            is_live = self._is_live_query(query)

            if is_live:
                routing_reason = "live keywords detected"
                result = self.web_agent.run(query)
                self.routing_counter["web_search"] += 1
                chosen_route = "web_search"
            else:
                retrieval = self._retrieve(query)
                score = retrieval["top_score"]
                passages = retrieval["passages"]

                if is_complex and score >= self.low_threshold:
                    routing_reason = (
                        f"complex query (score={score:.2f}, using retrieved context)"
                    )
                    result = self.reasoning_agent.run(query, context_chunks=passages)
                    self.routing_counter["reasoning"] += 1
                    chosen_route = "reasoning"
                elif is_complex:
                    routing_reason = (
                        f"complex query (low retrieval score {score:.2f}, "
                        f"reasoning without context)"
                    )
                    result = self.reasoning_agent.run(query)
                    self.routing_counter["reasoning"] += 1
                    chosen_route = "reasoning"
                elif score >= self.confidence_threshold:
                    routing_reason = f"high retrieval confidence {score:.2f}"
                    result = self.rag_pipeline.query(query)
                    self.routing_counter["rag"] += 1
                    chosen_route = "rag"
                elif score >= self.low_threshold:
                    routing_reason = (
                        f"medium retrieval confidence {score:.2f}, "
                        f"reasoning with context"
                    )
                    result = self.reasoning_agent.run(query, context_chunks=passages)
                    self.routing_counter["reasoning"] += 1
                    chosen_route = "reasoning"
                else:
                    routing_reason = (
                        f"low retrieval confidence {score:.2f}, "
                        f"falling back to web search"
                    )
                    result = self.web_agent.run(query)
                    self.routing_counter["web_search"] += 1
                    chosen_route = "web_search"

        # Normalize result shape
        if not isinstance(result, dict):
            result = {"answer": str(result), "sources": [], "route": chosen_route}

        result.setdefault("answer", "")
        result.setdefault("sources", [])
        result.setdefault("route", chosen_route)
        result["routing_reason"] = routing_reason

        confidence = result.get("confidence") if isinstance(result, dict) else None
        logger.info(
            f"Routed query to '{result['route']}' | reason: {routing_reason} | "
            f"confidence: {confidence}"
        )
        return result

    def get_analytics(self) -> dict:
        """Return cumulative routing analytics for observability."""
        total = sum(self.routing_counter.values())
        distribution = {
            k: (v / total * 100 if total else 0.0)
            for k, v in self.routing_counter.items()
        }
        return {
            "routing_counter": self.routing_counter,
            "total_queries": total,
            "distribution": distribution,
        }
