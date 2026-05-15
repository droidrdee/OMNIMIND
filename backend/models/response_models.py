"""Pydantic response schemas for the OMNIMIND HTTP API."""

from typing import List, Optional

from pydantic import BaseModel


class QueryResponse(BaseModel):
    """Result returned by the ``/query`` endpoint.

    Attributes:
        answer: The final natural-language answer produced by the selected
            agent.
        sources: Ordered list of source identifiers or URLs that grounded the
            answer.
        route: The agent that handled the query (``"rag"``, ``"web_search"``,
            or ``"reasoning"``).
        routing_reason: Human-readable explanation of why the router chose
            this path.
        confidence: Optional retrieval confidence score in ``[0, 1]``.
        latency_ms: End-to-end server-side latency in milliseconds.
    """

    answer: str
    sources: List[str]
    route: str
    routing_reason: str
    confidence: Optional[float] = None
    latency_ms: float


class IngestResponse(BaseModel):
    """Result returned by the document ingestion endpoint.

    Attributes:
        doc_id: Unique identifier assigned to the ingested document.
        filename: Original filename of the uploaded artefact.
        chunks_created: Number of vector chunks written to the store.
        method: Parsing strategy used — one of ``"native"`` (PyMuPDF text),
            ``"ocr"`` (Tesseract on scanned PDF), ``"mixed"`` (both per page),
            ``"image"`` (single-image OCR), or ``"error"`` (batch failure).
        message: Human-readable status message.
    """

    doc_id: str
    filename: str
    chunks_created: int
    method: str
    message: str


class DocumentListResponse(BaseModel):
    """Paginated listing of documents currently in the vector store.

    Attributes:
        documents: List of document metadata dictionaries.
        total: Total number of documents available.
    """

    documents: List[dict]
    total: int
