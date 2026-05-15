"""ChromaDB-backed vector store with cosine similarity."""

import json
from typing import List

import chromadb
from loguru import logger


class ChromaVectorStore:
    """Vector store backed by a remote ChromaDB instance using HNSW cosine similarity."""

    def __init__(self, host: str, port: int, collection_name: str) -> None:
        """Connect to a ChromaDB HTTP server and get/create the target collection.

        Args:
            host: ChromaDB server hostname.
            port: ChromaDB server port.
            collection_name: Name of the collection to use (created if missing).
        """
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Connected to ChromaDB at {host}:{port} (collection='{collection_name}')"
        )

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """Coerce metadata values into types ChromaDB accepts (str, int, float, bool).

        Dicts and lists are JSON-encoded; None is dropped; other types are stringified.
        """
        clean: dict = {}
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (dict, list, tuple)):
                try:
                    clean[key] = json.dumps(value, default=str)
                except (TypeError, ValueError):
                    clean[key] = str(value)
            else:
                clean[key] = str(value)
        return clean

    def add_chunks(self, chunks: List[dict], embeddings: List[List[float]]) -> int:
        """Insert chunks with precomputed embeddings into the collection.

        Args:
            chunks: List of chunk dicts with keys "chunk_id", "text", "metadata".
            embeddings: Parallel list of embedding vectors.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [self._sanitize_metadata(c.get("metadata", {})) for c in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(chunks)} chunks to vector store")
        return len(chunks)

    def query(self, query_embedding: List[float], top_k: int = 5) -> dict:
        """Run a cosine similarity search and return ranked passages.

        Args:
            query_embedding: Embedding vector of the user query.
            top_k: Maximum number of passages to return.

        Returns:
            Dict with "results" (list of passages) and "top_score" (best similarity).
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        passages: List[dict] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            passages.append(
                {
                    "text": doc,
                    "metadata": meta or {},
                    "distance": dist,
                    "score": 1 - dist,
                }
            )

        top_score = max((p["score"] for p in passages), default=0.0)
        return {"results": passages, "top_score": top_score}

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document.

        Args:
            doc_id: Document identifier stored in chunk metadata under "doc_id".

        Returns:
            Number of chunks deleted.
        """
        existing = self.collection.get(where={"doc_id": doc_id})
        ids = existing.get("ids", []) or []
        count = len(ids)

        if count == 0:
            logger.info(f"No chunks found for doc_id='{doc_id}'")
            return 0

        self.collection.delete(where={"doc_id": doc_id})
        logger.info(f"Deleted {count} chunks for doc_id='{doc_id}'")
        return count

    def list_documents(self) -> List[dict]:
        """List unique documents stored in the collection.

        Returns:
            List of dicts with "doc_id", "source", and "chunk_count".
        """
        all_data = self.collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", []) or []

        by_doc: dict[str, dict] = {}
        for meta in metadatas:
            if not meta:
                continue
            doc_id = meta.get("doc_id", "unknown")
            entry = by_doc.get(doc_id)
            if entry is None:
                by_doc[doc_id] = {
                    "doc_id": doc_id,
                    "source": meta.get("source", "unknown"),
                    "chunk_count": 1,
                }
            else:
                entry["chunk_count"] += 1

        return list(by_doc.values())

    def count(self) -> int:
        """Return the total number of chunks stored in the collection."""
        return self.collection.count()
