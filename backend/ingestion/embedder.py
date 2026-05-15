"""Sentence-transformers wrapper for generating embeddings. Auto-detects GPU."""

from typing import List, Optional

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around a SentenceTransformer model with GPU auto-detection.

    Produces L2-normalized embeddings suitable for cosine similarity search.
    """

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        """Load the embedding model onto the chosen device.

        Args:
            model_name: HuggingFace / sentence-transformers model identifier.
            device: Torch device string. If None, CUDA is used when available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            logger.error(
                f"Failed to load embedding model '{model_name}' on device '{device}': {exc}"
            )
            raise

        self.model_name = model_name
        self.device = device
        logger.info(f"EmbeddingModel loaded: model='{model_name}', device='{device}'")

    def embed_texts(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Embed a batch of texts into normalized vectors.

        Args:
            texts: Input strings to embed.
            batch_size: Number of texts processed per forward pass.

        Returns:
            A list of embedding vectors (one per input text).
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except Exception as exc:
            logger.error(f"Failed to embed {len(texts)} texts: {exc}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.

        Args:
            query: Query text to embed.

        Returns:
            A single normalized embedding vector.
        """
        return self.embed_texts([query])[0]

    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.

        Returns:
            Embedding vector dimension as reported by the underlying model.
        """
        return self.model.get_sentence_embedding_dimension()
