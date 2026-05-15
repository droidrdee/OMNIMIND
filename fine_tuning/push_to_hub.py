"""Push a fine-tuned SentenceTransformer to the HuggingFace Hub.

The script loads the merged model from disk, writes a model card describing
the fine-tuning recipe, and uploads everything to the requested repository.
The HF token can be supplied via ``--token`` or via the ``HF_TOKEN``
environment variable.

Usage::

    export HF_TOKEN=hf_xxx
    python push_to_hub.py \
        --model-path ./checkpoints/omnimind-minilm-v1 \
        --repo-id your-org/omnimind-minilm-v1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("push_to_hub")


MODEL_CARD_TEMPLATE = """---
language: en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - sentence-transformers
  - feature-extraction
  - sentence-similarity
  - retrieval
  - qlora
  - peft
license: apache-2.0
---

# {repo_id}

This is a [sentence-transformers](https://www.SBERT.net) model that was fine-tuned
with **QLoRA** (4-bit NF4 base + LoRA adapters, then merged) on synthetic
query/positive/negative triplets generated from the OMNIMIND knowledge base.

## Training recipe

| Setting       | Value                                       |
| ------------- | ------------------------------------------- |
| Base model    | `{base_model}`                              |
| Quantization  | 4-bit NF4 + double quant, compute fp16      |
| Adapters      | LoRA r=16, alpha=32, target=q,v, dropout 0.1|
| Loss          | MultipleNegativesRankingLoss                |
| Optimizer     | AdamW with linear warmup                    |
| Framework     | sentence-transformers + peft + bitsandbytes |

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{repo_id}")
embeddings = model.encode(["What is photosynthesis?", "How do plants make food?"])
```

## Intended use

Retrieval / semantic search inside the OMNIMIND multi-agent research stack.
The model was trained on domain-specific synthetic data and may underperform
on out-of-domain corpora compared to the base checkpoint.

## Limitations

* Synthetic questions inherit any bias of the upstream LLM used to generate them.
* No hard-negative mining beyond random sampling from different sources.
* Evaluated on a held-out slice of the same synthetic distribution.
"""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push a fine-tuned SentenceTransformer to the HuggingFace Hub.",
    )
    parser.add_argument("--model-path", required=True, help="Local path to the saved model.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target HF repo id, e.g. 'your-org/omnimind-minilm-v1'.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF write token. Falls back to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model id used in the model card (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create / push to a private repo (default: public).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def write_model_card(model_dir: Path, repo_id: str, base_model: str) -> Path:
    card_path = model_dir / "README.md"
    card_path.write_text(
        MODEL_CARD_TEMPLATE.format(repo_id=repo_id, base_model=base_model),
        encoding="utf-8",
    )
    logger.info("Wrote model card to %s", card_path)
    return card_path


def run(args: argparse.Namespace) -> int:
    from sentence_transformers import SentenceTransformer  # type: ignore

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model path does not exist: %s", model_path)
        return 2

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        logger.error("No HF token supplied (use --token or set HF_TOKEN).")
        return 2

    logger.info("Loading model from %s", model_path)
    model = SentenceTransformer(str(model_path))

    write_model_card(model_path, args.repo_id, args.base_model)

    logger.info(
        "Pushing to hub: repo=%s private=%s",
        args.repo_id,
        bool(args.private),
    )
    try:
        model.push_to_hub(
            repo_id=args.repo_id,
            token=token,
            private=bool(args.private),
        )
    except TypeError:
        # Older versions of sentence-transformers used positional arg name `repo_name`.
        model.push_to_hub(args.repo_id, token=token, private=bool(args.private))

    print(f"\nModel pushed: https://huggingface.co/{args.repo_id}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.warning("Upload interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
