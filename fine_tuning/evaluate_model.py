"""Side-by-side evaluation of a base vs fine-tuned SentenceTransformer model.

For each triplet ``(query, positive, _)`` we embed the query and rank it
against the pool of all unique positive passages in the evaluation slice
using cosine similarity. The position of the true positive within the
ranking gives us MRR@10, NDCG@10, and Hit@5. The script prints a tidy
comparison table including the percentage improvement of the fine-tuned
model over the base model.

Usage::

    python evaluate_model.py \
        --base-model sentence-transformers/all-MiniLM-L6-v2 \
        --finetuned-model ./checkpoints/omnimind-minilm-v1 \
        --eval-dataset ./data/triplets \
        --top-k 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("evaluate_model")

EVAL_SLICE = 200  # last N triplets used as the held-out set


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a base and fine-tuned retriever on MRR/NDCG/Hit metrics.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="HF id or path of the base SentenceTransformer.",
    )
    parser.add_argument(
        "--finetuned-model",
        required=True,
        help="Path to the fine-tuned SentenceTransformer.",
    )
    parser.add_argument(
        "--eval-dataset",
        required=True,
        help="HF Dataset (save_to_disk) or JSONL file with query/positive fields.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Cutoff used for MRR@k and NDCG@k (default: 10).",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=EVAL_SLICE,
        help=f"Number of triplets to use as held-out set (default: {EVAL_SLICE}).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_eval_triplets(path: Path, eval_size: int) -> list[dict[str, str]]:
    """Load the held-out triplets from either a HF Dataset or a JSONL file."""
    import json

    if path.is_dir():
        from datasets import load_from_disk  # type: ignore

        ds = load_from_disk(str(path))
        records = list(ds)
    else:
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                records.append(json.loads(raw))

    triplets: list[dict[str, str]] = []
    for record in records:
        query = record.get("query")
        positive = record.get("positive")
        if not (query and positive):
            continue
        triplets.append({"query": query, "positive": positive})
    return triplets[-eval_size:]


def compute_metrics(
    query_emb: np.ndarray,
    passage_emb: np.ndarray,
    positive_index: np.ndarray,
    top_k: int,
) -> tuple[float, float, float]:
    """Compute (MRR@k, NDCG@k, Hit@5) over the evaluation set."""
    # Normalize once so dot product == cosine similarity.
    q_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)
    p_norm = passage_emb / (np.linalg.norm(passage_emb, axis=1, keepdims=True) + 1e-12)
    sims = q_norm @ p_norm.T  # shape: (n_queries, n_passages)

    # argsort descending; ranks[i] is the array of passage indices sorted by
    # similarity to query i.
    ranks = np.argsort(-sims, axis=1)

    n = sims.shape[0]
    mrr_total = 0.0
    ndcg_total = 0.0
    hit5_total = 0
    log2_denoms = 1.0 / np.log2(np.arange(2, top_k + 2))  # for positions 1..top_k

    for i in range(n):
        # Position (0-indexed) of the true positive in the ranking.
        position = int(np.where(ranks[i] == positive_index[i])[0][0])
        rank_1based = position + 1

        # MRR@k: 0 if positive is beyond k.
        if rank_1based <= top_k:
            mrr_total += 1.0 / rank_1based

        # NDCG@k with binary relevance: dcg = 1/log2(rank+1), ideal = 1.
        if rank_1based <= top_k:
            ndcg_total += log2_denoms[position]

        # Hit@5
        if rank_1based <= 5:
            hit5_total += 1

    mrr = mrr_total / n
    ndcg = ndcg_total / n
    hit5 = hit5_total / n
    return mrr, ndcg, hit5


def encode_with(model, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def pct_delta(new: float, old: float) -> float:
    if old <= 0:
        return 0.0
    return (new - old) / old * 100.0


def run(args: argparse.Namespace) -> int:
    from sentence_transformers import SentenceTransformer  # type: ignore

    eval_path = Path(args.eval_dataset)
    if not eval_path.exists():
        logger.error("Eval dataset not found: %s", eval_path)
        return 2

    triplets = load_eval_triplets(eval_path, args.eval_size)
    if not triplets:
        logger.error("No usable triplets in eval set")
        return 1

    queries = [t["query"] for t in triplets]
    # Build the passage pool: unique positives only.
    seen: dict[str, int] = {}
    passages: list[str] = []
    positive_idx: list[int] = []
    for t in triplets:
        pos = t["positive"]
        if pos not in seen:
            seen[pos] = len(passages)
            passages.append(pos)
        positive_idx.append(seen[pos])
    positive_index = np.asarray(positive_idx, dtype=np.int64)

    logger.info(
        "Eval set: %d queries / %d unique passages", len(queries), len(passages),
    )

    logger.info("Loading base model: %s", args.base_model)
    base = SentenceTransformer(args.base_model)
    logger.info("Encoding with base model")
    base_q = encode_with(base, queries)
    base_p = encode_with(base, passages)
    base_metrics = compute_metrics(base_q, base_p, positive_index, args.top_k)

    logger.info("Loading fine-tuned model: %s", args.finetuned_model)
    ft = SentenceTransformer(args.finetuned_model)
    logger.info("Encoding with fine-tuned model")
    ft_q = encode_with(ft, queries)
    ft_p = encode_with(ft, passages)
    ft_metrics = compute_metrics(ft_q, ft_p, positive_index, args.top_k)

    b_mrr, b_ndcg, b_hit = base_metrics
    f_mrr, f_ndcg, f_hit = ft_metrics

    print("\n=== Retrieval evaluation ===")
    print(f"{'Model':<16}| {'MRR@'+str(args.top_k):<7}| {'NDCG@'+str(args.top_k):<8}| {'Hit@5':<6}")
    print("-" * 46)
    print(f"{'Base':<16}|  {b_mrr:.3f} |  {b_ndcg:.3f}  |  {b_hit:.3f}")
    print(f"{'Fine-tuned':<16}|  {f_mrr:.3f} |  {f_ndcg:.3f}  |  {f_hit:.3f}")
    print(
        f"{'Improvement':<16}| {pct_delta(f_mrr, b_mrr):+5.1f}% | "
        f"{pct_delta(f_ndcg, b_ndcg):+5.1f}%  | {pct_delta(f_hit, b_hit):+5.1f}%"
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
