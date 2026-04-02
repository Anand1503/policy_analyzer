"""
Resume training from checkpoint-186 (epoch 2) → run epochs 3-5.
Then: threshold calibration + test evaluation + model export.
"""
import os, sys, json, csv, time, logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TAXONOMY = [
    "DATA_COLLECTION", "DATA_SHARING", "USER_RIGHTS",
    "DATA_RETENTION", "SECURITY_MEASURES", "THIRD_PARTY_TRANSFER",
    "COOKIES_TRACKING", "CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE",
    "LIABILITY_LIMITATION",
]
NUM_LABELS = len(TAXONOMY)
MAX_LENGTH = 128
BATCH_SIZE = 16
GRAD_ACCUM = 2
EPOCHS = 5          # Total epochs (resume continues from epoch 2)
LR = 2e-5
WARMUP_RATIO = 0.1
FREEZE_LAYERS = 9
FOCAL_GAMMA = 2.0
PATIENCE = 2

BASE_MODEL_DIR = Path("./models/legal-bert")
CHECKPOINT_DIR = Path("./training_output_final/checkpoint-186")
OUTPUT_DIR = Path("./training_output_final")
EXPORT_DIR = Path("./models/legal-bert-final")
EVAL_DIR = Path("./evaluation")
DATA_DIR = Path("./data/processed")


class ClauseDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                text = row[0]
                labels = [int(x) for x in row[1:]]
                self.samples.append((text, labels))
        log.info(f"  Loaded {len(self.samples)} from {csv_path}")

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


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


class FocalTrainer(Trainer):
    def __init__(self, focal_loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "f1_macro": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "f1_micro": round(f1_score(labels, preds, average="micro", zero_division=0), 4),
        "precision_macro": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(labels, preds, average="macro", zero_division=0), 4),
        "exact_match": round(np.mean(np.all(preds == labels, axis=1)), 4),
    }


def predict_all(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def tune_thresholds(logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    thresholds = {}
    for i, label in enumerate(TAXONOMY):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.025):
            preds = (probs[:, i] >= t).astype(int)
            if preds.sum() == 0:
                continue
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = round(float(t), 3)
        thresholds[label] = {"threshold": best_t, "val_f1": round(best_f1, 4)}
        log.info(f"  {label:25s}: t={best_t:.3f}  val_f1={best_f1:.4f}")
    return thresholds


def evaluate_test(logits, labels, thresholds):
    probs = 1 / (1 + np.exp(-logits))
    preds = np.zeros_like(probs, dtype=int)
    for i, label in enumerate(TAXONOMY):
        t = thresholds[label]["threshold"]
        preds[:, i] = (probs[:, i] >= t).astype(int)

    metrics = {
        "f1_macro": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "f1_micro": round(f1_score(labels, preds, average="micro", zero_division=0), 4),
        "precision_macro": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(labels, preds, average="macro", zero_division=0), 4),
        "exact_match": round(np.mean(np.all(preds == labels, axis=1)), 4),
    }

    report = classification_report(labels, preds, target_names=TAXONOMY, zero_division=0, digits=4)
    log.info(f"\n  F1 Macro:  {metrics['f1_macro']}")
    log.info(f"  F1 Micro:  {metrics['f1_micro']}")
    log.info(f"  Precision: {metrics['precision_macro']}")
    log.info(f"  Recall:    {metrics['recall_macro']}")
    log.info(f"  Exact Match: {metrics['exact_match']}")
    log.info(f"\n{report}")
    return metrics, report


def main():
    start = time.time()
    device = torch.device("cpu")

    log.info("=" * 60)
    log.info("RESUMING FROM CHECKPOINT-186 (Epoch 2)")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(CHECKPOINT_DIR), num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    # Freeze layers (same as original run)
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i in range(FREEZE_LAYERS):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"  Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")

    model.to(device)

    # Load datasets
    train_ds = ClauseDataset(DATA_DIR / "train.csv", tokenizer)
    val_ds = ClauseDataset(DATA_DIR / "val.csv", tokenizer)
    test_ds = ClauseDataset(DATA_DIR / "test.csv", tokenizer)

    # Class weights
    label_matrix = np.zeros((len(train_ds), NUM_LABELS))
    for i, (_, labels) in enumerate(train_ds.samples):
        label_matrix[i] = labels
    pos_counts = label_matrix.sum(axis=0)
    neg_counts = len(train_ds) - pos_counts
    pos_weight = np.clip(neg_counts / (pos_counts + 1e-6), 1.0, 20.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float).to(device)

    focal_loss = FocalBCEWithLogitsLoss(gamma=FOCAL_GAMMA, pos_weight=pos_weight_tensor)

    # Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        logging_steps=10,
        seed=SEED,
        dataloader_pin_memory=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = FocalTrainer(
        focal_loss_fn=focal_loss,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    log.info("  Resuming training from checkpoint-186...")
    trainer.train(resume_from_checkpoint=str(CHECKPOINT_DIR))
    train_time = time.time() - start
    log.info(f"\n  Training time: {train_time:.0f}s ({train_time/60:.1f} min)")

    # Threshold calibration
    log.info("\n" + "=" * 60)
    log.info("THRESHOLD CALIBRATION (Validation Set)")
    log.info("=" * 60)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_logits, val_labels = predict_all(model, val_loader, device)
    thresholds = tune_thresholds(val_logits, val_labels)

    # Test evaluation
    log.info("\n" + "=" * 60)
    log.info("FINAL TEST EVALUATION")
    log.info("=" * 60)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_logits, test_labels = predict_all(model, test_loader, device)
    metrics, report = evaluate_test(test_logits, test_labels, thresholds)
    metrics["dataset_sizes"] = {
        "train": len(train_ds), "val": len(val_ds), "test": len(test_ds),
    }
    metrics["training_time_sec"] = round(train_time, 1)

    # Export
    log.info("\n" + "=" * 60)
    log.info("EXPORTING FINAL MODEL")
    log.info("=" * 60)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(EVAL_DIR / "final_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(EVAL_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    model.save_pretrained(str(EXPORT_DIR))
    tokenizer.save_pretrained(str(EXPORT_DIR))
    with open(EXPORT_DIR / "optimal_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(EXPORT_DIR / "model_metadata.json", "w") as f:
        json.dump({
            "version": "vFINAL",
            "base_model": "nlpaueb/legal-bert-base-uncased",
            "training": f"{EPOCHS} epochs focal loss, {FREEZE_LAYERS} frozen layers, max_length={MAX_LENGTH}",
            "dataset_sizes": metrics["dataset_sizes"],
            "f1_macro": metrics["f1_macro"],
            "f1_micro": metrics["f1_micro"],
        }, f, indent=2)

    total = time.time() - start
    log.info(f"\n  Model → {EXPORT_DIR}")
    log.info(f"  Eval → {EVAL_DIR}")
    log.info(f"  Total: {total:.0f}s ({total/60:.1f} min)")
    log.info(f"  F1 Macro: {metrics['f1_macro']}")
    log.info(f"  F1 Micro: {metrics['f1_micro']}")

    if metrics["f1_macro"] >= 0.70:
        log.info("\n✅ TARGET ACHIEVED — F1 Macro ≥ 0.70")
    elif metrics["f1_macro"] >= 0.65:
        log.info(f"\n✅ GOOD PERFORMANCE — F1 Macro {metrics['f1_macro']}")
    log.info("\n🎯 MODEL OPTIMIZED & READY")


if __name__ == "__main__":
    main()
