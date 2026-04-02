"""
Export best checkpoint → final model with threshold tuning + test evaluation.
Uses checkpoint-68 from the first training run (epoch 1, F1 macro 0.555).
"""
import os, sys, json, csv, time, logging
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

TAXONOMY = [
    "DATA_COLLECTION", "DATA_SHARING", "USER_RIGHTS",
    "DATA_RETENTION", "SECURITY_MEASURES", "THIRD_PARTY_TRANSFER",
    "COOKIES_TRACKING", "CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE",
    "LIABILITY_LIMITATION",
]
NUM_LABELS = len(TAXONOMY)
MAX_LENGTH = 256
BATCH_SIZE = 16

# Paths
CHECKPOINT_DIR = Path("./training_output_v2/checkpoint-68")
EXPORT_DIR = Path("./models/legal-bert-finetuned-v2")
EVAL_DIR = Path("./evaluation")
DATA_DIR = Path("./data/processed")


class ClauseDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                text = row[0]
                labels = [int(x) for x in row[1:]]
                self.samples.append((text, labels))
        log.info(f"  Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, labels = self.samples[idx]
        enc = self.tokenizer(
            text, max_length=MAX_LENGTH, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


def predict_all(model, dataloader, device):
    """Get all predictions and labels."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def tune_thresholds(logits, labels):
    """Per-label threshold tuning on validation set."""
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    thresholds = {}
    for i, label in enumerate(TAXONOMY):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, i] >= t).astype(int)
            if preds.sum() == 0:
                continue
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = round(float(t), 2)
        thresholds[label] = {"threshold": best_t, "val_f1": round(best_f1, 4)}
        log.info(f"  {label:25s}: t={best_t:.2f}  val_f1={best_f1:.4f}")
    return thresholds


def evaluate_test(logits, labels, thresholds):
    """Final evaluation on test set with tuned thresholds."""
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = np.zeros_like(probs, dtype=int)
    for i, label in enumerate(TAXONOMY):
        t = thresholds[label]["threshold"]
        preds[:, i] = (probs[:, i] >= t).astype(int)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)

    # Exact match ratio
    exact_match = np.mean(np.all(preds == labels, axis=1))

    metrics = {
        "f1_macro": round(f1_macro, 4),
        "f1_micro": round(f1_micro, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "exact_match": round(exact_match, 4),
        "dataset_sizes": {
            "train": None,  # filled later
            "val": None,
            "test": int(labels.shape[0]),
        },
    }

    log.info(f"\n  F1 Macro:  {f1_macro:.4f}")
    log.info(f"  F1 Micro:  {f1_micro:.4f}")
    log.info(f"  Precision: {precision_macro:.4f}")
    log.info(f"  Recall:    {recall_macro:.4f}")
    log.info(f"  Exact Match: {exact_match:.4f}")

    # Per-label report
    report = classification_report(
        labels, preds, target_names=TAXONOMY, zero_division=0
    )
    log.info(f"\n{report}")

    return metrics, report


def main():
    start = time.time()
    device = torch.device("cpu")

    log.info("=" * 60)
    log.info("LOADING BEST CHECKPOINT")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("./models/legal-bert")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(CHECKPOINT_DIR), num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )
    model.to(device)
    log.info(f"Model loaded from {CHECKPOINT_DIR}")

    # Load datasets
    val_ds = ClauseDataset(DATA_DIR / "val.csv", tokenizer)
    test_ds = ClauseDataset(DATA_DIR / "test.csv", tokenizer)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Count train size
    train_count = sum(1 for _ in open(DATA_DIR / "train.csv", encoding="utf-8")) - 1

    # PHASE 1: Threshold tuning on validation set
    log.info("\n" + "=" * 60)
    log.info("PHASE 4 — Threshold Tuning (Validation Set)")
    log.info("=" * 60)

    val_logits, val_labels = predict_all(model, val_loader, device)
    thresholds = tune_thresholds(val_logits, val_labels)

    # PHASE 2: Final evaluation on test set
    log.info("\n" + "=" * 60)
    log.info("PHASE 5 — Final Test Evaluation")
    log.info("=" * 60)

    test_logits, test_labels = predict_all(model, test_loader, device)
    metrics, report = evaluate_test(test_logits, test_labels, thresholds)
    metrics["dataset_sizes"]["train"] = train_count
    metrics["dataset_sizes"]["val"] = len(val_ds)

    # PHASE 3: Save everything
    log.info("\n" + "=" * 60)
    log.info("PHASE 6 — Export Final Model")
    log.info("=" * 60)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(EVAL_DIR / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  Saved → {EVAL_DIR / 'final_metrics.json'}")

    # Save thresholds
    with open(EVAL_DIR / "final_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"  Saved → {EVAL_DIR / 'final_thresholds.json'}")

    # Save classification report
    with open(EVAL_DIR / "classification_report.txt", "w") as f:
        f.write(report)
    log.info(f"  Saved → {EVAL_DIR / 'classification_report.txt'}")

    # Export model
    model.save_pretrained(str(EXPORT_DIR))
    tokenizer.save_pretrained(str(EXPORT_DIR))

    # Save thresholds to model dir too
    with open(EXPORT_DIR / "optimal_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    # Save metadata
    metadata = {
        "version": "v2",
        "base_model": "nlpaueb/legal-bert-base-uncased",
        "training": "1 epoch focal loss on 2842-sample OPP-115-aligned dataset",
        "dataset_sizes": metrics["dataset_sizes"],
        "f1_macro": metrics["f1_macro"],
        "f1_micro": metrics["f1_micro"],
    }
    with open(EXPORT_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Saved model → {EXPORT_DIR}")

    elapsed = time.time() - start
    log.info(f"\n  Total time: {elapsed:.1f}s")
    log.info(f"  F1 Macro: {metrics['f1_macro']}")
    log.info(f"  F1 Micro: {metrics['f1_micro']}")

    # Quality gate
    if metrics["f1_macro"] >= 0.65:
        log.info("\n✅ QUALITY GATE PASSED — F1 Macro ≥ 0.65")
    else:
        log.info(f"\n⚠️  F1 Macro {metrics['f1_macro']} below 0.65 target")
        log.info("  This is expected for 1-epoch CPU training.")
        log.info("  For production, train more epochs with: --epochs 10 --focal")

    log.info("\n🎯 MODEL EXPORTED SUCCESSFULLY")


if __name__ == "__main__":
    main()
