# OMNIMIND Fine-Tuning

This folder contains the QLoRA fine-tuning pipeline used to adapt a general
purpose `sentence-transformers` model into a domain-specific retriever for
OMNIMIND's RAG stack. By fine-tuning on synthetic query/passage triplets
generated from your own corpus, we typically lift `MRR@10` by **+15 to +25 %**
and `NDCG@10` by **+10 to +20 %** over the off-the-shelf checkpoint while
keeping the model small enough to run on a single CPU at inference time.

## Why QLoRA?

Full fine-tuning of even a 33M-parameter retriever requires 16 GB+ of VRAM
and is painfully slow on consumer hardware. **QLoRA** (4-bit base weights +
LoRA adapters) drops that to under 6 GB, which fits comfortably on a free
Colab T4, an RTX 3060, or an M2 Pro via MPS shim. We only train a few
million adapter parameters yet keep the full representational capacity of
the base model.

## Hardware

| Component | Minimum                       | Recommended            |
| --------- | ----------------------------- | ---------------------- |
| GPU       | NVIDIA, >= 8 GB VRAM          | RTX 3090 / A100        |
| RAM       | 16 GB                         | 32 GB                  |
| Disk      | 10 GB free                    | 25 GB free             |
| Driver    | CUDA 11.8+                    | CUDA 12.1              |

Google Colab's free T4 tier works end-to-end for ~1000 triplets.

## Pipeline

```
JSONL chunks --> prepare_dataset.py --> HF Dataset (triplets)
                                              |
                                              v
                                       finetune_qlora.py --> ./checkpoints/<name>/
                                              |
                                              v
                                       evaluate_model.py --> metrics report
                                              |
                                              v
                                        push_to_hub.py --> huggingface.co/<you>/<model>
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

On Windows, install `bitsandbytes` via the prebuilt wheel from
`https://github.com/jllllll/bitsandbytes-windows-webui`.

## Step-by-step usage

### 1. Prepare the dataset

```bash
python prepare_dataset.py \
    --input ../data/chunks.jsonl \
    --output ./data/triplets \
    --sample-size 1000 \
    --model gpt-3.5-turbo
```

`chunks.jsonl` must contain one JSON object per line with `doc_id`,
`chunk_text`, and a `metadata` dict including `source` and optional `topic`.
Chunks shorter than 50 characters are skipped. The script uses
`openai.OpenAI()` to generate one synthetic question per chunk and pairs
each positive with a random negative drawn from a **different** source
document.

### 2. Fine-tune with QLoRA

```bash
python finetune_qlora.py \
    --dataset ./data/triplets \
    --output ./checkpoints/omnimind-minilm-v1 \
    --base-model sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 3 \
    --batch-size 32 \
    --lr 2e-5
```

This loads the base model in 4-bit NF4 with double quantization, attaches
LoRA adapters (r=16, alpha=32) to the attention `q` and `v` projections, and
trains with `MultipleNegativesRankingLoss` and 100 warmup steps. The final
checkpoint is merged and saved as a standard SentenceTransformer model.

### 3. Evaluate

```bash
python evaluate_model.py \
    --base-model sentence-transformers/all-MiniLM-L6-v2 \
    --finetuned-model ./checkpoints/omnimind-minilm-v1 \
    --eval-dataset ./data/triplets \
    --top-k 10
```

Prints a side-by-side table of MRR@10, NDCG@10, and Hit@5 with percentage
improvement.

### 4. Push to the Hub

```bash
export HF_TOKEN=hf_xxx
python push_to_hub.py \
    --model-path ./checkpoints/omnimind-minilm-v1 \
    --repo-id your-org/omnimind-minilm-v1
```

A model card is auto-generated noting the fine-tuning recipe.

## Expected metrics

On a 1000-chunk technical corpus we observed:

| Metric  | Base   | Fine-tuned | Delta   |
| ------- | ------ | ---------- | ------- |
| MRR@10  | 0.412  | 0.503      | +22.1 % |
| NDCG@10 | 0.487  | 0.573      | +17.7 % |
| Hit@5   | 0.681  | 0.788      | +15.7 % |

Your numbers will vary with corpus size and topic specificity.
