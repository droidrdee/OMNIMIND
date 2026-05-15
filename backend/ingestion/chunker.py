"""Splits documents into overlapping chunks using LlamaIndex SentenceSplitter."""

from typing import List
from uuid import uuid4

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger


class DocumentChunker:
    """Chunk documents into overlapping sentence-aware segments.

    Wraps LlamaIndex's SentenceSplitter to produce chunks suitable for
    embedding and retrieval-augmented generation.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Number of overlapping tokens between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        logger.info(
            f"DocumentChunker initialized (chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap})"
        )

    def chunk_text(self, text: str, filename: str) -> List[dict]:
        """Split text into chunks tied to a shared document id.

        Args:
            text: Full document text to split.
            filename: Source filename, recorded in chunk metadata.

        Returns:
            A list of chunk dicts each containing chunk_id, text, metadata,
            source, and the shared doc_id.
        """
        try:
            doc_id = str(uuid4())
            document = Document(
                text=text, metadata={"source": filename, "doc_id": doc_id}
            )
            nodes = self.splitter.get_nodes_from_documents([document])

            chunks: List[dict] = []
            for index, node in enumerate(nodes):
                metadata = dict(node.metadata)
                metadata["chunk_index"] = index
                chunks.append(
                    {
                        "chunk_id": str(uuid4()),
                        "text": node.text,
                        "metadata": metadata,
                        "source": filename,
                        "doc_id": doc_id,
                    }
                )

            logger.info(
                f"Chunked '{filename}' into {len(chunks)} chunks (doc_id={doc_id})"
            )
            return chunks
        except Exception as exc:
            logger.error(f"Failed to chunk text for '{filename}': {exc}")
            raise
