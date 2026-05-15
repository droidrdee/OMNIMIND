"""Seed the OMNIMIND vector store by uploading documents through the API.

Walks a local directory for supported file types (PDF + common image formats)
and POSTs each one to the backend's ``/api/v1/ingest`` endpoint. The backend
is responsible for OCR, chunking, embedding, and persisting into ChromaDB.

Usage::

    python seed_chromadb.py \
        --backend-url http://localhost:8080 \
        --docs-dir ./data/seed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("seed_chromadb")

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload PDFs and images to the OMNIMIND ingest endpoint.",
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8080",
        help="Base URL of the OMNIMIND backend (default: http://localhost:8080).",
    )
    parser.add_argument(
        "--docs-dir",
        required=True,
        help="Directory to walk for documents.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds per upload (default: 120).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded without sending anything.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def find_documents(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def upload_file(client: httpx.Client, endpoint: str, file_path: Path) -> tuple[bool, str]:
    suffix = file_path.suffix.lower()
    mime = CONTENT_TYPES.get(suffix, "application/octet-stream")
    try:
        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh, mime)}
            response = client.post(endpoint, files=files)
        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}: {response.text[:200]}"
        return True, response.text[:200]
    except httpx.HTTPError as exc:
        return False, f"transport error: {exc}"


def run(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir).expanduser().resolve()
    if not docs_dir.exists() or not docs_dir.is_dir():
        logger.error("Docs directory does not exist or is not a directory: %s", docs_dir)
        return 2

    files = find_documents(docs_dir)
    if not files:
        logger.warning("No supported documents found under %s", docs_dir)
        return 0

    endpoint = args.backend_url.rstrip("/") + "/api/v1/ingest"
    logger.info("Found %d documents under %s", len(files), docs_dir)
    logger.info("Target endpoint: %s", endpoint)

    if args.dry_run:
        for f in files:
            print(f"[dry-run] would upload {f}")
        return 0

    successes = 0
    failures = 0
    with httpx.Client(timeout=args.timeout) as client:
        for idx, path in enumerate(files, start=1):
            ok, message = upload_file(client, endpoint, path)
            if ok:
                successes += 1
                logger.info("[%d/%d] OK   %s", idx, len(files), path.name)
            else:
                failures += 1
                logger.error("[%d/%d] FAIL %s :: %s", idx, len(files), path.name, message)

    print("\n=== Seed summary ===")
    print(f"  total      : {len(files)}")
    print(f"  succeeded  : {successes}")
    print(f"  failed     : {failures}")
    return 0 if failures == 0 else 1


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
