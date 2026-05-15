"""Generate a synthetic query/positive/negative triplet dataset for fine-tuning.

The script reads a JSONL file of pre-chunked documents, asks an OpenAI chat
model to write a single likely user question for each chunk, then pairs the
chunk (positive) with a random chunk from a **different** source document
(negative). The resulting triplets are saved as a HuggingFace ``Dataset``
that can be consumed directly by ``finetune_qlora.py``.

Input format (one JSON object per line)::

    {
      "doc_id": "doc-001",
      "chunk_text": "Photosynthesis is the process by which ...",
      "metadata": {"source": "biology.pdf", "topic": "plants"}
    }

Usage::

    python prepare_dataset.py \
        --input ../data/chunks.jsonl \
        --output ./data/triplets \
        --sample-size 1000 \
        --model gpt-3.5-turbo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("prepare_dataset")

MIN_CHUNK_CHARS = 50
SYSTEM_PROMPT = (
    "You are a question-writing assistant for retrieval evaluation. "
    "Given a passage, write ONE concise, natural question that a real user "
    "might type into a search box and whose answer is contained in the passage. "
    "Respond with only the question, no preamble, no quotes."
)


@dataclass
class Chunk:
    doc_id: str
    text: str
    source: str
    topic: str

    @classmethod
    def from_jsonl_record(cls, record: dict[str, Any]) -> "Chunk":
        meta = record.get("metadata") or {}
        return cls(
            doc_id=str(record.get("doc_id", "")),
            text=str(record.get("chunk_text", "")),
            source=str(meta.get("source", "unknown")),
            topic=str(meta.get("topic", "general")),
        )


def load_chunks(path: Path) -> list[Chunk]:
    """Load and filter chunks from a JSONL file."""
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_no, exc)
                continue
            chunk = Chunk.from_jsonl_record(record)
            if len(chunk.text) < MIN_CHUNK_CHARS:
                continue
            chunks.append(chunk)
    return chunks


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(min=1, max=30),
    retry=retry_if_exception_type(Exception),
)
def generate_question(client: OpenAI, model: str, passage: str) -> str:
    """Ask the chat model for one question about ``passage`` with retries."""
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=64,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Passage:\n{passage}\n\nQuestion:"},
        ],
    )
    content = response.choices[0].message.content or ""
    return content.strip().strip('"').strip("'")


def pick_negative(rng: random.Random, pool: list[Chunk], positive: Chunk) -> Chunk:
    """Pick a chunk from a different source than ``positive``."""
    candidates = [c for c in pool if c.source != positive.source]
    if not candidates:
        # Fall back to any chunk that is not the positive itself.
        candidates = [c for c in pool if c.doc_id != positive.doc_id] or pool
    return rng.choice(candidates)


def build_triplets(
    chunks: list[Chunk],
    sample_size: int,
    model: str,
    client: OpenAI,
    rng: random.Random,
) -> list[dict[str, str]]:
    """Generate ``sample_size`` triplets using the OpenAI chat model."""
    if sample_size <= 0 or sample_size > len(chunks):
        sample_size = len(chunks)
    selected = rng.sample(chunks, sample_size)

    triplets: list[dict[str, str]] = []
    failures = 0
    for chunk in tqdm(selected, desc="Generating questions", unit="chunk"):
        try:
            query = generate_question(client, model, chunk.text)
        except Exception as exc:
            failures += 1
            logger.warning("Question generation failed for %s: %s", chunk.doc_id, exc)
            continue
        if not query:
            failures += 1
            continue
        negative = pick_negative(rng, chunks, chunk)
        triplets.append(
            {
                "query": query,
                "positive": chunk.text,
                "negative": negative.text,
                "positive_source": chunk.source,
                "negative_source": negative.source,
            }
        )

    if failures:
        logger.warning("Question generation failed for %d chunks", failures)
    return triplets


def print_stats(triplets: list[dict[str, str]]) -> None:
    if not triplets:
        logger.warning("No triplets generated; nothing to report")
        return
    avg_q = sum(len(t["query"]) for t in triplets) / len(triplets)
    avg_p = sum(len(t["positive"]) for t in triplets) / len(triplets)
    avg_n = sum(len(t["negative"]) for t in triplets) / len(triplets)
    sources = {t["positive_source"] for t in triplets}
    print("\n=== Dataset stats ===")
    print(f"  triplets       : {len(triplets)}")
    print(f"  unique sources : {len(sources)}")
    print(f"  avg query len  : {avg_q:.1f} chars")
    print(f"  avg positive   : {avg_p:.1f} chars")
    print(f"  avg negative   : {avg_n:.1f} chars")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a synthetic triplet dataset for retriever fine-tuning.",
    )
    parser.add_argument("--input", required=True, help="Path to chunks JSONL file.")
    parser.add_argument("--output", required=True, help="Output directory for the HF Dataset.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of chunks to sample (default: 1000).",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI chat model used for question generation (default: gpt-3.5-turbo).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return 2

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading chunks from %s", input_path)
    chunks = load_chunks(input_path)
    logger.info("Loaded %d chunks (>= %d chars)", len(chunks), MIN_CHUNK_CHARS)
    if not chunks:
        logger.error("No usable chunks found. Aborting.")
        return 1

    client = OpenAI(api_key=api_key)
    triplets = build_triplets(chunks, args.sample_size, args.model, client, rng)

    if not triplets:
        logger.error("No triplets generated. Aborting.")
        return 1

    dataset = Dataset.from_list(triplets)
    dataset.save_to_disk(str(output_path))
    logger.info("Saved %d triplets to %s", len(triplets), output_path)
    print_stats(triplets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
