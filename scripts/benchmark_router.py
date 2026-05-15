"""Benchmark OMNIMIND's QueryRouter against a labelled set of queries.

For each query, the script POSTs to the backend, parses the chosen route, and
compares it to the expected label. The output is a per-route precision/recall
table plus the overall accuracy. If no ``--queries-file`` is supplied, a
built-in suite of 30 queries (10 per route) is used so the script is useful
out of the box.

Usage::

    # Use built-in queries:
    python benchmark_router.py --backend-url http://localhost:8080

    # Use your own labelled JSONL:
    python benchmark_router.py \
        --backend-url http://localhost:8080 \
        --queries-file ./bench/router_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("benchmark_router")

VALID_ROUTES = {"rag", "web_search", "reasoning"}

DEFAULT_QUERIES: list[dict[str, str]] = [
    # ---- RAG (10): factual lookup inside the indexed corpus ----
    {"query": "What does the document say about the company's revenue in 2023?", "expected_route": "rag"},
    {"query": "Summarize the methodology section of the research paper.", "expected_route": "rag"},
    {"query": "What are the main findings of the uploaded report?", "expected_route": "rag"},
    {"query": "List the safety procedures described in the manual.", "expected_route": "rag"},
    {"query": "What is the warranty period mentioned in the contract?", "expected_route": "rag"},
    {"query": "Which authors are credited in the introduction?", "expected_route": "rag"},
    {"query": "Quote the definition of 'photosynthesis' from the biology document.", "expected_route": "rag"},
    {"query": "How does the document describe the manufacturing process?", "expected_route": "rag"},
    {"query": "What dataset is used in the experiments according to the paper?", "expected_route": "rag"},
    {"query": "What are the eligibility criteria stated in the policy?", "expected_route": "rag"},

    # ---- Web search (10): live / recent / outside-corpus ----
    {"query": "What is the latest news on AI regulation today?", "expected_route": "web_search"},
    {"query": "Current weather in Paris right now.", "expected_route": "web_search"},
    {"query": "Who won the 2025 Champions League final?", "expected_route": "web_search"},
    {"query": "What is the stock price of NVIDIA today?", "expected_route": "web_search"},
    {"query": "Latest release notes for Python 3.13.", "expected_route": "web_search"},
    {"query": "Recent earthquake reports in Japan this week.", "expected_route": "web_search"},
    {"query": "Today's exchange rate between USD and EUR.", "expected_route": "web_search"},
    {"query": "Who is the current CEO of OpenAI?", "expected_route": "web_search"},
    {"query": "Recent advancements in fusion energy in 2026.", "expected_route": "web_search"},
    {"query": "Latest movie releases on streaming this month.", "expected_route": "web_search"},

    # ---- Reasoning (10): comparative / explanatory / multi-step ----
    {"query": "Compare and contrast neural networks and decision trees in terms of interpretability.", "expected_route": "reasoning"},
    {"query": "Explain why backpropagation works mathematically.", "expected_route": "reasoning"},
    {"query": "Analyze the trade-offs between microservices and a monolith for a small startup.", "expected_route": "reasoning"},
    {"query": "If we doubled the batch size, what would happen to the gradient noise and why?", "expected_route": "reasoning"},
    {"query": "Walk me through how to derive the closed-form solution of linear regression.", "expected_route": "reasoning"},
    {"query": "Compare REST and GraphQL for a real-time chat application.", "expected_route": "reasoning"},
    {"query": "Explain step by step why quicksort is O(n log n) on average.", "expected_route": "reasoning"},
    {"query": "What is the reasoning behind using cross-entropy over MSE for classification?", "expected_route": "reasoning"},
    {"query": "Compare and contrast L1 and L2 regularization in terms of sparsity.", "expected_route": "reasoning"},
    {"query": "Explain why transformers outperformed RNNs for sequence modelling.", "expected_route": "reasoning"},
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark routing accuracy of the OMNIMIND QueryRouter.",
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8080",
        help="Base URL of the OMNIMIND backend (default: http://localhost:8080).",
    )
    parser.add_argument(
        "--queries-file",
        default=None,
        help="Optional JSONL file with {query, expected_route} per line.",
    )
    parser.add_argument(
        "--endpoint-path",
        default="/api/v1/query",
        help="Path of the query endpoint (default: /api/v1/query).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout per query in seconds (default: 60).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_queries(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return list(DEFAULT_QUERIES)
    queries: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", line_no, exc)
                continue
            query = record.get("query")
            expected = record.get("expected_route")
            if not query or expected not in VALID_ROUTES:
                logger.warning("Skipping invalid record on line %d: %s", line_no, record)
                continue
            queries.append({"query": query, "expected_route": expected})
    return queries


def call_router(client: httpx.Client, endpoint: str, query: str) -> str | None:
    try:
        response = client.post(endpoint, json={"query": query})
    except httpx.HTTPError as exc:
        logger.error("Transport error for %r: %s", query, exc)
        return None
    if response.status_code >= 400:
        logger.error("HTTP %d for %r :: %s", response.status_code, query, response.text[:200])
        return None
    try:
        data = response.json()
    except json.JSONDecodeError:
        logger.error("Non-JSON response for %r :: %s", query, response.text[:200])
        return None
    route = data.get("route") or (data.get("metadata") or {}).get("route")
    return route if route in VALID_ROUTES else None


def compute_per_route_metrics(
    results: list[tuple[str, str | None]],
) -> dict[str, dict[str, float]]:
    """Return precision/recall/F1 per route using the multi-class confusion."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for expected, actual in results:
        if actual is None:
            fn[expected] += 1
            continue
        if actual == expected:
            tp[expected] += 1
        else:
            fp[actual] += 1
            fn[expected] += 1

    metrics: dict[str, dict[str, float]] = {}
    for route in sorted(VALID_ROUTES):
        precision_denom = tp[route] + fp[route]
        recall_denom = tp[route] + fn[route]
        precision = tp[route] / precision_denom if precision_denom else 0.0
        recall = tp[route] / recall_denom if recall_denom else 0.0
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        metrics[route] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp[route] + fn[route],
        }
    return metrics


def run(args: argparse.Namespace) -> int:
    queries_path = Path(args.queries_file) if args.queries_file else None
    if queries_path is not None and not queries_path.exists():
        logger.error("Queries file does not exist: %s", queries_path)
        return 2

    queries = load_queries(queries_path)
    if not queries:
        logger.error("No queries to benchmark.")
        return 1

    endpoint = args.backend_url.rstrip("/") + args.endpoint_path
    logger.info("Benchmarking %d queries against %s", len(queries), endpoint)

    results: list[tuple[str, str | None]] = []
    with httpx.Client(timeout=args.timeout) as client:
        for idx, item in enumerate(queries, start=1):
            actual = call_router(client, endpoint, item["query"])
            results.append((item["expected_route"], actual))
            status = "OK" if actual == item["expected_route"] else ("?" if actual is None else "X")
            logger.info(
                "[%d/%d] %s expected=%s actual=%s :: %s",
                idx,
                len(queries),
                status,
                item["expected_route"],
                actual or "ERR",
                item["query"][:60] + ("..." if len(item["query"]) > 60 else ""),
            )

    total = len(results)
    correct = sum(1 for exp, act in results if exp == act and act is not None)
    accuracy = correct / total if total else 0.0
    metrics = compute_per_route_metrics(results)

    print("\n=== Router benchmark ===")
    print(f"{'Route':<12}| {'Prec':<6}| {'Rec':<6}| {'F1':<6}| {'N':<4}")
    print("-" * 42)
    for route, m in metrics.items():
        print(
            f"{route:<12}| {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {int(m['support']):<4}"
        )
    print("-" * 42)
    print(f"Overall accuracy: {accuracy:.3f}  ({correct}/{total})")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
