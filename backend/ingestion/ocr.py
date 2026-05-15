"""OCR pipeline using PyMuPDF for native PDF text + Tesseract OCR fallback for scanned PDFs/images."""

import os
import tempfile
from uuid import uuid4

import fitz
import pytesseract
from loguru import logger
from PIL import Image


class OCRPipeline:
    """Hybrid OCR pipeline that extracts text from PDFs and images.

    Uses PyMuPDF for native text extraction from PDFs and falls back to
    Tesseract OCR when pages contain scanned/image-only content.
    """

    def __init__(self, tesseract_lang: str = "eng") -> None:
        """Initialize the OCR pipeline.

        Args:
            tesseract_lang: Tesseract language code(s) (e.g. "eng", "eng+fra").
        """
        self.tesseract_lang = tesseract_lang
        logger.info(f"OCRPipeline initialized with tesseract_lang='{tesseract_lang}'")

    def extract_text_from_pdf(self, file_path: str) -> tuple[str, int, str]:
        """Extract text from a PDF file, using OCR fallback for scanned pages.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            A tuple of (text, page_count, method) where method is one of
            "native", "ocr", or "mixed".
        """
        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            logger.error(f"Failed to open PDF '{file_path}': {exc}")
            raise

        page_texts: list[str] = []
        used_native = False
        used_ocr = False

        try:
            for page_index, page in enumerate(doc, start=1):
                try:
                    native_text = page.get_text()
                except Exception as exc:
                    logger.error(
                        f"Failed to extract native text from page {page_index} of "
                        f"'{file_path}': {exc}"
                    )
                    native_text = ""

                if len(native_text) > 50:
                    page_text = native_text
                    used_native = True
                else:
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                        image = Image.frombytes(
                            "RGB", [pix.width, pix.height], pix.samples
                        )
                        page_text = pytesseract.image_to_string(
                            image, lang=self.tesseract_lang
                        )
                        used_ocr = True
                    except Exception as exc:
                        logger.error(
                            f"OCR failed on page {page_index} of '{file_path}': {exc}"
                        )
                        page_text = native_text
                        if native_text:
                            used_native = True

                page_texts.append(f"\n\n--- Page {page_index} ---\n\n{page_text}")

            page_count = doc.page_count
        finally:
            doc.close()

        if used_native and used_ocr:
            method = "mixed"
        elif used_ocr:
            method = "ocr"
        else:
            method = "native"

        text = "".join(page_texts).lstrip()
        logger.info(
            f"Extracted {len(text)} chars from '{file_path}' "
            f"({page_count} pages, method={method})"
        )
        return text, page_count, method

    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from a single image file using Tesseract OCR.

        Args:
            file_path: Absolute path to the image file.

        Returns:
            The OCR-extracted text.
        """
        try:
            with Image.open(file_path) as image:
                text = pytesseract.image_to_string(image, lang=self.tesseract_lang)
            logger.info(f"OCR extracted {len(text)} chars from image '{file_path}'")
            return text
        except Exception as exc:
            logger.error(f"Image OCR failed for '{file_path}': {exc}")
            raise

    def process_uploaded_file(
        self, file_bytes: bytes, filename: str, content_type: str
    ) -> dict:
        """Persist an uploaded file to a temp path and route it through OCR.

        Args:
            file_bytes: Raw file bytes.
            filename: Original filename (used for extension detection).
            content_type: MIME type reported by the client.

        Returns:
            A dict with keys: filename, text, page_count, method.
        """
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{uuid4()}_{filename}")

        try:
            with open(temp_path, "wb") as fh:
                fh.write(file_bytes)
            logger.debug(f"Wrote uploaded file to '{temp_path}' ({len(file_bytes)} bytes)")

            lowered_name = filename.lower()
            lowered_ct = (content_type or "").lower()

            is_pdf = lowered_ct == "application/pdf" or lowered_name.endswith(".pdf")
            image_exts = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
            is_image = lowered_ct.startswith("image/") or lowered_name.endswith(image_exts)

            if is_pdf:
                text, page_count, method = self.extract_text_from_pdf(temp_path)
            elif is_image:
                text = self.extract_text_from_image(temp_path)
                page_count = 1
                method = "image"
            else:
                msg = (
                    f"Unsupported file type for '{filename}' "
                    f"(content_type='{content_type}')"
                )
                logger.error(msg)
                raise ValueError(msg)

            return {
                "filename": filename,
                "text": text,
                "page_count": page_count,
                "method": method,
            }
        except Exception as exc:
            logger.error(f"Failed to process uploaded file '{filename}': {exc}")
            raise
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.debug(f"Removed temp file '{temp_path}'")
            except OSError as exc:
                logger.error(f"Failed to remove temp file '{temp_path}': {exc}")
