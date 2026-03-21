"""
Phases 2-5 — Legal-BERT Fine-Tuning + Evaluation + Export

Phase 2: Model configuration (Legal-BERT, BCEWithLogitsLoss)
Phase 3: Training with HuggingFace Trainer
Phase 4: Evaluation (F1, confusion matrix, per-label)
Phase 5: Model export to models/legal-bert-finetuned/
"""
import os, sys, json, csv, time, logging
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, multilabel_confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TAXONOMY = [
    "DATA_COLLECTION", "DATA_SHARING", "USER_RIGHTS",
    "DATA_RETENTION", "SECURITY_MEASURES", "THIRD_PARTY_TRANSFER",
    "COOKIES_TRACKING", "CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE",
    "LIABILITY_LIMITATION",
]

MODEL_NAME = "./models/legal-bert"
EXPORT_DIR = "./models/legal-bert-finetuned"
DATA_DIR = Path("data/processed")
EVAL_DIR = Path("evaluation")
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


class ClauseDataset(Dataset):
    """Multi-label clause dataset for Legal-BERT."""

    def __init__(self, csv_path, tokenizer, max_length=MAX_LENGTH):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"]
                labels = [int(row.get(f"label_{t}", 0)) for t in TAXONOMY]
                self.samples.append({"text": text, "labels": labels})

        log.info(f"Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(sample["labels"], dtype=torch.float),
        }


def compute_metrics(eval_pred):
    """Compute multi-label classification metrics."""
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy().astype(int)
    labels = labels.astype(int)

    f1_mac = f1_score(labels, preds, average="macro", zero_division=0)
    f1_mic = f1_score(labels, preds, average="micro", zero_division=0)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)

    # Exact match ratio
    exact = np.all(preds == labels, axis=1).mean()

    return {
        "f1_macro": round(f1_mac, 4),
        "f1_micro": round(f1_mic, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "exact_match": round(exact, 4),
    }


def main():
    log.info("=" * 60)
    log.info("PHASE 2 — Model Configuration")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TAXONOMY),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    log.info(f"Model loaded: {MODEL_NAME} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Load datasets
    train_ds = ClauseDataset(DATA_DIR / "train.csv", tokenizer)
    val_ds = ClauseDataset(DATA_DIR / "val.csv", tokenizer)
    test_ds = ClauseDataset(DATA_DIR / "test.csv", tokenizer)

    # ── Phase 3 — Training ───────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 3 — Training")
    log.info("=" * 60)

    output_dir = "./training_output"
    total_steps = (len(train_ds) // BATCH_SIZE + 1) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        seed=SEED,
        report_to="none",
        fp16=False,  # CPU-safe
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    log.info(f"Training complete in {train_time:.0f}s")

    # ── Phase 4 — Evaluation ──────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 4 — Evaluation on Test Set")
    log.info("=" * 60)

    results = trainer.evaluate(test_ds)
    log.info(f"Test results: {results}")

    # Per-label predictions
    preds_out = trainer.predict(test_ds)
    logits = preds_out.predictions
    labels = preds_out.label_ids
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy().astype(int)

    # Classification report
    report = classification_report(
        labels.astype(int), preds,
        target_names=TAXONOMY, zero_division=0, output_dict=True,
    )
    report_text = classification_report(
        labels.astype(int), preds,
        target_names=TAXONOMY, zero_division=0,
    )

    # Per-label F1
    per_label = {}
    for label_name in TAXONOMY:
        per_label[label_name] = {
            "precision": round(report[label_name]["precision"], 4),
            "recall": round(report[label_name]["recall"], 4),
            "f1": round(report[label_name]["f1-score"], 4),
            "support": int(report[label_name]["support"]),
        }

    # Confusion matrices
    cm = multilabel_confusion_matrix(labels.astype(int), preds)
    cm_dict = {TAXONOMY[i]: cm[i].tolist() for i in range(len(TAXONOMY))}

    # Exact match
    exact_match = np.all(preds == labels.astype(int), axis=1).mean()

    # Save metrics
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": MODEL_NAME,
        "finetuned_model": EXPORT_DIR,
        "dataset_sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "hyperparameters": {
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "seed": SEED,
        },
        "test_metrics": {
            "f1_macro": round(results.get("eval_f1_macro", 0), 4),
            "f1_micro": round(results.get("eval_f1_micro", 0), 4),
            "precision_macro": round(results.get("eval_precision_macro", 0), 4),
            "recall_macro": round(results.get("eval_recall_macro", 0), 4),
            "exact_match": round(exact_match, 4),
        },
        "per_label": per_label,
        "confusion_matrices": cm_dict,
        "training_time_seconds": round(train_time, 1),
    }

    with open(EVAL_DIR / "legalbert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {EVAL_DIR / 'legalbert_metrics.json'}")

    with open(EVAL_DIR / "classification_report.txt", "w") as f:
        f.write("Legal-BERT Multi-Label Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Test samples: {len(test_ds)}\n")
        f.write(f"Training time: {train_time:.0f}s\n\n")
        f.write(report_text)
        f.write(f"\nExact Match Ratio: {exact_match:.4f}\n")
    log.info(f"Report saved → {EVAL_DIR / 'classification_report.txt'}")

    # ── Phase 5 — Model Export ────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 5 — Model Export")
    log.info("=" * 60)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    trainer.save_model(EXPORT_DIR)
    tokenizer.save_pretrained(EXPORT_DIR)

    # Save metadata
    metadata = {
        "model_name": "legal-bert-finetuned",
        "base_model": "nlpaueb/legal-bert-base-uncased",
        "task": "multi_label_classification",
        "num_labels": len(TAXONOMY),
        "labels": TAXONOMY,
        "dataset": "OPP-115 aligned",
        "f1_macro": metrics["test_metrics"]["f1_macro"],
        "training_epochs": EPOCHS,
        "seed": SEED,
    }
    with open(os.path.join(EXPORT_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Report model size
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(EXPORT_DIR) for f in fn
    )
    log.info(f"Model saved → {EXPORT_DIR} ({total_size / (1024**2):.0f} MB)")

    # Print final summary
    log.info("\n" + "=" * 60)
    log.info("FINAL SUMMARY")
    log.info("=" * 60)
    log.info(f"Dataset:  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    log.info(f"F1 Macro: {metrics['test_metrics']['f1_macro']}")
    log.info(f"F1 Micro: {metrics['test_metrics']['f1_micro']}")
    log.info(f"Exact Match: {metrics['test_metrics']['exact_match']}")
    log.info(f"Model Size: {total_size / (1024**2):.0f} MB")
    log.info(f"Training Time: {train_time:.0f}s")

    # Cleanup training checkpoints
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        log.info(f"Cleaned up {output_dir}")

    log.info("\nPhases 2-5 COMPLETE")


if __name__ == "__main__":
    main()
