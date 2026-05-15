"""QLoRA fine-tuning entry point for OMNIMIND's retriever.

This script loads a base ``sentence-transformers`` model in 4-bit NF4
quantization, attaches LoRA adapters to the attention projections, and
trains it on the triplet dataset produced by ``prepare_dataset.py`` using
``MultipleNegativesRankingLoss``. The merged model is saved as a standard
SentenceTransformer that can be loaded with one line of code.

Usage::

    python finetune_qlora.py \
        --dataset ./data/triplets \
        --output ./checkpoints/omnimind-minilm-v1 \
        --base-model sentence-transformers/all-MiniLM-L6-v2 \
        --epochs 3 \
        --batch-size 32 \
        --lr 2e-5
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("finetune_qlora")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a sentence-transformer with QLoRA on triplet data.",
    )
    parser.add_argument("--dataset", required=True, help="Path to HF Dataset (save_to_disk).")
    parser.add_argument("--output", required=True, help="Where to write the fine-tuned model.")
    parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base SentenceTransformer model to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size per device (default: 32).",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Max input sequence length in tokens (default: 256).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def run(args: argparse.Namespace) -> int:
    # Imports are inside ``run`` so ``--help`` does not require torch/peft.
    import torch  # type: ignore
    from datasets import load_from_disk  # type: ignore
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    from sentence_transformers import (  # type: ignore
        InputExample,
        SentenceTransformer,
        losses,
    )
    from torch.utils.data import DataLoader  # type: ignore
    from transformers import BitsAndBytesConfig  # type: ignore

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        logger.error("Dataset path does not exist: %s", dataset_path)
        return 2

    logger.info("Loading triplet dataset from %s", dataset_path)
    ds = load_from_disk(str(dataset_path))
    logger.info("Loaded %d triplets", len(ds))

    # ---- 4-bit quantization config ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    logger.info(
        "BitsAndBytes: load_in_4bit=True, quant=nf4, double_quant=True, compute=fp16",
    )

    # ---- LoRA config ----
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    logger.info(
        "LoRA: r=%d alpha=%d target=q,v dropout=0.1 bias=none",
        args.lora_r,
        args.lora_alpha,
    )

    # ---- Load base model and wrap with PEFT ----
    logger.info("Loading base model: %s", args.base_model)
    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length

    # The first module is a Transformer; wrap its underlying HF model with PEFT.
    base_transformer = model[0].auto_model
    try:
        base_transformer = base_transformer.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model[0].auto_model = base_transformer
    except Exception as exc:  # pragma: no cover - hardware specific
        logger.warning(
            "Could not load 4-bit quantized weights (%s); continuing in fp32/fp16.",
            exc,
        )

    model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
    if hasattr(model[0].auto_model, "print_trainable_parameters"):
        model[0].auto_model.print_trainable_parameters()

    # ---- Build training examples ----
    logger.info("Building InputExamples from triplets")
    train_examples: list[InputExample] = []
    for row in ds:
        query = row.get("query") or ""
        positive = row.get("positive") or ""
        negative = row.get("negative") or ""
        if not (query and positive and negative):
            continue
        train_examples.append(InputExample(texts=[query, positive, negative]))
    logger.info("Prepared %d training examples", len(train_examples))

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    logger.info(
        "Training: epochs=%d batch=%d lr=%.2e warmup=%d",
        args.epochs,
        args.batch_size,
        args.lr,
        args.warmup_steps,
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=str(output_path),
    )

    logger.info("Saving merged model to %s", output_path)
    model.save(str(output_path))

    print("\n=== Fine-tuning summary ===")
    print(f"  base model     : {args.base_model}")
    print(f"  output path    : {output_path}")
    print(f"  triplets used  : {len(train_examples)}")
    print(f"  epochs         : {args.epochs}")
    print(f"  batch size     : {args.batch_size}")
    print(f"  learning rate  : {args.lr}")
    print(f"  LoRA r/alpha   : {args.lora_r}/{args.lora_alpha}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 130
    except Exception:  # pragma: no cover - top-level safety net
        logger.error("Fine-tuning failed:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
