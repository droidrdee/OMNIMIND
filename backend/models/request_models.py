"""Pydantic request schemas for the OMNIMIND HTTP API."""

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Inbound payload for the ``/query`` endpoint.

    Attributes:
        query: The natural-language question submitted by the user.
        force_route: Optional override that bypasses the query router and
            pins execution to a specific agent. Accepted values are
            ``"rag"``, ``"web_search"``, and ``"reasoning"``.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="User query",
    )
    force_route: Optional[str] = Field(
        None,
        description="Force route: rag | web_search | reasoning",
    )


class DeleteDocumentRequest(BaseModel):
    """Inbound payload for the document deletion endpoint.

    Attributes:
        doc_id: Identifier of the document to remove from the vector store.
    """

    doc_id: str
