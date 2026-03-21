# ML Models

This directory contains all ML models used by the Intelligent Policy Analyzer.
Models are **not tracked by git** due to their large size (~1.4 GB total).

## Models Required

| Directory | Model | Size | Source |
|-----------|-------|------|--------|
| `legal-bert/` | Legal-BERT base (classifier) | ~440 MB | `nlpaueb/legal-bert-base-uncased` |
| `legal-bert-finetuned-v2/` | Fine-tuned classifier | ~440 MB | Created by training script |
| `t5-base/` | T5-base (summarizer + RAG) | ~900 MB | `t5-base` |
| `all-MiniLM-L6-v2/` | Sentence embeddings | ~80 MB | `sentence-transformers/all-MiniLM-L6-v2` |

## Download Models

Run the download script to fetch all models locally:

```bash
# From the backend/ directory
python scripts/download_models.py
```

This downloads all models from HuggingFace Hub and saves them locally for offline use.

## Fine-Tuning

To retrain the Legal-BERT classifier:

```bash
python scripts/train_legalbert_v2.py [--focal] [--fp16] [--epochs 10]
```

Options:
- `--focal` — Use Focal Loss (better for class imbalance)
- `--fp16` — Mixed precision training (requires CUDA GPU)
- `--epochs N` — Number of training epochs (default: 10)
- `--patience N` — Early stopping patience (default: 2)

The fine-tuned model is saved to `legal-bert-finetuned-v2/` with:
- `config.json` + `model.safetensors` — Model weights
- `tokenizer.json` — Tokenizer
- `thresholds.json` — Per-label classification thresholds
- `model_metadata.json` — Training metadata and metrics
