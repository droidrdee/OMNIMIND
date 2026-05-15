"""Tests for the ingestion pipeline.

These tests exercise the document chunker, embedding model, and OCR pipeline
in isolation from heavyweight dependencies such as Tesseract or large
transformer models. The heavyweight checks are gated behind the
``integration`` marker so a default ``pytest`` run stays fast and offline.

Run only the unit slice (default in CI):
    pytest backend/tests/test_ingest.py -m "not integration"

Run the full slice (requires Tesseract + network + GPU optional):
    pytest backend/tests/test_ingest.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the backend package is importable regardless of pytest cwd.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

def test_chunker_creates_correct_chunk_count() -> None:
    """A 5000-char text with size=512/overlap=64 must yield >= 5 chunks."""
    from ingestion.chunker import DocumentChunker

    text = "lorem ipsum dolor sit amet " * 200  # >= 5000 chars
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_text(text, filename="synthetic.txt")

    assert len(chunks) >= 5, f"Expected at least 5 chunks, got {len(chunks)}"
    doc_ids = {chunk["doc_id"] for chunk in chunks}
    assert len(doc_ids) == 1, "All chunks should share a single doc_id"


def test_chunker_metadata_preserved() -> None:
    """Filename supplied at chunking time must be propagated onto each chunk."""
    from ingestion.chunker import DocumentChunker

    filename = "research_paper.pdf"
    chunker = DocumentChunker(chunk_size=256, chunk_overlap=32)
    chunks = chunker.chunk_text("hello world " * 100, filename=filename)

    assert chunks, "Chunker returned no chunks"
    assert chunks[0]["metadata"]["source"] == filename
    assert chunks[0]["source"] == filename


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_embedder_returns_correct_dimension() -> None:
    """all-MiniLM-L6-v2 produces 384-dim vectors. Requires the model on disk."""
    from ingestion.embedder import EmbeddingModel

    model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.embed_query("hello")
    assert len(emb) == 384


# ---------------------------------------------------------------------------
# OCR pipeline
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ocr_image_extraction(tmp_path: Path) -> None:
    """OCRPipeline.extract_text_from_image returns the tesseract output."""
    from PIL import Image
    from ingestion.ocr import OCRPipeline

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(image_path)

    with patch("pytesseract.image_to_string", return_value="hello world"):
        pipeline = OCRPipeline()
        result = pipeline.extract_text_from_image(str(image_path))

    assert "hello world" in result


def test_ocr_pdf_native_text() -> None:
    """If a PDF page has selectable text, the native path must be chosen."""
    from ingestion.ocr import OCRPipeline

    mock_page = MagicMock()
    mock_page.get_text.return_value = "long native text content " * 10

    mock_doc = MagicMock()
    mock_doc.__iter__.return_value = iter([mock_page])
    mock_doc.page_count = 1
    mock_doc.close.return_value = None

    with patch("ingestion.ocr.fitz.open", return_value=mock_doc):
        pipeline = OCRPipeline()
        text, page_count, method = pipeline.extract_text_from_pdf("ignored.pdf")

    assert method == "native"
    assert "long native text content" in text
    assert page_count == 1


def test_ocr_pdf_fallback_to_tesseract() -> None:
    """If the PDF has no embedded text, fall back to tesseract OCR."""
    from ingestion.ocr import OCRPipeline

    mock_pixmap = MagicMock()
    mock_pixmap.samples = b"\x00" * (10 * 10 * 3)
    mock_pixmap.width = 10
    mock_pixmap.height = 10

    mock_page = MagicMock()
    mock_page.get_text.return_value = ""  # forces OCR fallback
    mock_page.get_pixmap.return_value = mock_pixmap

    mock_doc = MagicMock()
    mock_doc.__iter__.return_value = iter([mock_page])
    mock_doc.page_count = 1
    mock_doc.close.return_value = None

    with patch("ingestion.ocr.fitz.open", return_value=mock_doc), \
         patch("ingestion.ocr.Image.frombytes", return_value=MagicMock()), \
         patch("ingestion.ocr.pytesseract.image_to_string", side_effect=lambda *a, **k: "ocr-text"):
        pipeline = OCRPipeline()
        text, _page_count, method = pipeline.extract_text_from_pdf("ignored.pdf")

    assert method == "ocr"
    assert "ocr-text" in text
