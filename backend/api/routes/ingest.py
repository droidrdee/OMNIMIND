"""Document-ingestion FastAPI routes."""

from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from loguru import logger

from core.config import get_settings
from models.response_models import DocumentListResponse, IngestResponse

router = APIRouter()

ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def _validate_file(file: UploadFile, max_bytes: int, size: int) -> None:
    """Validate uploaded file extension and size."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXT)}",
        )
    if size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size} bytes); max is {max_bytes} bytes",
        )


async def _ingest_single(file: UploadFile, request: Request) -> IngestResponse:
    """Run the full ingestion pipeline for a single uploaded file."""
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    data = await file.read()
    _validate_file(file, max_bytes, len(data))

    app = request.app
    ocr_result = app.state.ocr.process_uploaded_file(
        data, file.filename, file.content_type
    )
    chunks = app.state.chunker.chunk_text(ocr_result["text"], file.filename)
    if not chunks:
        raise HTTPException(status_code=422, detail="No content extracted")

    doc_id = chunks[0]["doc_id"]
    embeddings = app.state.embedder.embed_texts([c["text"] for c in chunks])
    count = app.state.vector_store.add_chunks(chunks, embeddings)

    return IngestResponse(
        doc_id=doc_id,
        filename=file.filename,
        chunks_created=count,
        method=ocr_result["method"],
        message=f"Ingested {count} chunks",
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: Request, file: UploadFile = File(...)) -> IngestResponse:
    """Ingest a single document (PDF or image) into the knowledge base."""
    try:
        return await _ingest_single(file, request)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception(f"Failed to ingest {file.filename}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest/batch", response_model=List[IngestResponse])
async def ingest_batch(
    request: Request, files: List[UploadFile] = File(...)
) -> List[IngestResponse]:
    """Ingest up to 10 documents in a single request."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

    results: List[IngestResponse] = []
    for file in files:
        try:
            results.append(await _ingest_single(file, request))
        except HTTPException as exc:
            logger.error(f"Batch ingest skipped {file.filename}: {exc.detail}")
            results.append(
                IngestResponse(
                    doc_id="",
                    filename=file.filename or "",
                    chunks_created=0,
                    method="error",
                    message=str(exc.detail),
                )
            )
        except Exception as exc:
            logger.exception(f"Batch ingest failed for {file.filename}")
            results.append(
                IngestResponse(
                    doc_id="",
                    filename=file.filename or "",
                    chunks_created=0,
                    method="error",
                    message=str(exc),
                )
            )
    return results


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(request: Request) -> DocumentListResponse:
    """List all documents currently stored in the vector store."""
    try:
        docs = request.app.state.vector_store.list_documents()
        return DocumentListResponse(documents=docs, total=len(docs))
    except Exception as exc:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request) -> dict:
    """Delete a document and all of its chunks from the vector store."""
    try:
        n = request.app.state.vector_store.delete_document(doc_id)
        return {"doc_id": doc_id, "deleted": True, "chunks_deleted": n}
    except Exception as exc:
        logger.exception(f"Failed to delete document {doc_id}")
        raise HTTPException(status_code=500, detail=str(exc))
