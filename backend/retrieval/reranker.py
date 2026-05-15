"""Cross-encoder reranker using FlashRank (ms-marco-MiniLM-L-12-v2)."""

from typing import List

from flashrank import Ranker, RerankRequest
from loguru import logger


class Reranker:
    """Lightweight cross-encoder reranker backed by FlashRank."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        """Load the FlashRank cross-encoder model.

        Args:
            model_name: FlashRank model identifier.
        """
        self.model_name = model_name
        self.ranker = Ranker(model_name=model_name)
        logger.info(f"Initialized FlashRank reranker with model='{model_name}'")

    def rerank(
        self, query: str, passages: List[dict], top_n: int = 3
    ) -> List[dict]:
        """Rerank passages against the query and return the top_n results.

        Args:
            query: User query string.
            passages: Candidate passages with at least a "text" key.
            top_n: Maximum number of passages to return.

        Returns:
            Original passage dicts (top_n), each annotated with "rerank_score",
            sorted by descending rerank score. Falls back to the first top_n
            passages unchanged on failure.
        """
        if not passages:
            return []

        try:
            flashrank_passages = [
                {
                    "id": i,
                    "text": p["text"],
                    "meta": p.get("metadata", {}),
                }
                for i, p in enumerate(passages)
            ]

            request = RerankRequest(query=query, passages=flashrank_passages)
            results = self.ranker.rerank(request)

            scored: List[dict] = []
            for result in results:
                idx = int(result["id"])
                if idx < 0 or idx >= len(passages):
                    continue
                original = dict(passages[idx])
                original["rerank_score"] = float(result["score"])
                scored.append(original)

            scored.sort(key=lambda p: p["rerank_score"], reverse=True)
            return scored[:top_n]
        except Exception as exc:
            logger.warning(f"Reranking failed ({exc}); returning unranked passages")
            return passages[:top_n]
