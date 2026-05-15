"""Retrieval-Augmented Generation pipeline: embed query -> ChromaDB -> rerank -> GPT-4o."""

from typing import List, Optional

from loguru import logger
from openai import OpenAI


class RAGPipeline:
    """End-to-end RAG pipeline combining a vector store, reranker, and OpenAI LLM."""

    def __init__(
        self,
        vector_store,
        embedder,
        reranker,
        openai_api_key: str,
        openai_model: str = "gpt-4o",
        top_k: int = 5,
    ) -> None:
        """Initialize the pipeline with its dependencies.

        Args:
            vector_store: Object exposing query(embedding, top_k) -> dict.
            embedder: Object exposing embed_query(text) -> List[float].
            reranker: Object exposing rerank(query, passages, top_n) -> List[dict].
            openai_api_key: API key for OpenAI chat completions.
            openai_model: OpenAI chat model identifier (default "gpt-4o").
            top_k: Number of passages to retrieve from the vector store.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.openai_model = openai_model
        self.top_k = top_k
        self.client = OpenAI(api_key=openai_api_key)
        logger.info(
            f"Initialized RAGPipeline (model='{openai_model}', top_k={top_k})"
        )

    def retrieve(self, query: str) -> dict:
        """Embed the query, hit the vector store, and rerank the candidates.

        Args:
            query: User query string.

        Returns:
            Dict with keys "passages" (reranked passages) and "top_score" (best
            similarity score from the initial vector search).
        """
        try:
            query_emb = self.embedder.embed_query(query)
        except Exception as exc:
            logger.error(f"Query embedding failed: {exc}")
            return {"passages": [], "top_score": 0.0}

        try:
            result = self.vector_store.query(query_emb, top_k=self.top_k)
        except Exception as exc:
            logger.error(f"Vector store query failed: {exc}")
            return {"passages": [], "top_score": 0.0}

        passages: List[dict] = result.get("results", [])
        top_score: float = result.get("top_score", 0.0)

        if passages and len(passages) > 1:
            passages = self.reranker.rerank(query, passages, top_n=3)

        return {"passages": passages, "top_score": top_score}

    def query(self, query_text: str) -> dict:
        """Run the full RAG pipeline and return a grounded answer.

        Args:
            query_text: User question to answer.

        Returns:
            Dict with keys "answer", "sources", "confidence", "route", and
            "passages_used".
        """
        retrieval = self.retrieve(query_text)
        passages: List[dict] = retrieval["passages"]
        confidence: float = retrieval["top_score"]

        context = "\n\n---\n\n".join(
            f"[Source: {p['metadata'].get('source', 'unknown')}]\n{p['text']}"
            for p in passages
        )

        sources: List[str] = []
        seen = set()
        for p in passages:
            src = p.get("metadata", {}).get("source", "unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)

        prompt = (
            "Answer the question based on the context below. "
            "If the context is insufficient, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query_text}\n\n"
            "Answer:"
        )

        answer: Optional[str] = None
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are OMNIMIND, a precise research assistant. "
                            "Cite sources when possible."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.2,
            )
            answer = response.choices[0].message.content
        except Exception as exc:
            logger.error(f"OpenAI chat completion failed: {exc}")
            answer = (
                "I was unable to generate an answer due to an upstream error."
            )

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "route": "rag",
            "passages_used": len(passages),
        }
