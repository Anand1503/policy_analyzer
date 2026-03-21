"""
Retrain LegalBERT v2 on leak-free dataset splits + Phase 3 Cross-Validation
===========================================================================
- Retrains on clean train.csv / val.csv (policy-level isolated)
- Performs threshold tuning on val set
- Evaluates on clean test set
- Runs 5-fold cross-validation using frozen-base + classifier head
- Saves models and metrics
"""

import csv, json, logging, os, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "evaluation"
BASE_MODEL = BASE_DIR / "models" / "legal-bert"
EXPORT_DIR = BASE_DIR / "models" / "legal-bert-finetuned-v2"
EVAL_DIR.mkdir(exist_ok=True)

TAXONOMY = [
    "DATA_COLLECTION", "DATA_SHARING", "USER_RIGHTS", "DATA_RETENTION",
    "SECURITY_MEASURES", "THIRD_PARTY_TRANSFER", "COOKIES_TRACKING",
    "CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE", "LIABILITY_LIMITATION",
]
NUM_LABELS = len(TAXONOMY)

# Hyperparameters
LR = 2e-5
BATCH_SIZE = 16
EPOCHS = 5
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42


class ClauseDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True,
            max_length=self.max_length, padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def load_data(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append([int(row.get(f"label_{t}", 0)) for t in TAXONOMY])
    return texts, np.array(labels)


def compute_pos_weights(labels):
    n = labels.shape[0]
    pos = labels.sum(axis=0)
    neg = n - pos
    weights = neg / (pos + 1e-6)
    weights = np.clip(weights, 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy().astype(int)
    labels = labels.astype(int)
    return {
        "f1_macro": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "f1_micro": round(f1_score(labels, preds, average="micro", zero_division=0), 4),
        "precision_macro": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(labels, preds, average="macro", zero_division=0), 4),
        "exact_match": round(np.all(preds == labels, axis=1).mean(), 4),
    }


class WeightedTrainer(Trainer):
    def __init__(self, pos_weight, num_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        return (loss, outputs) if return_outputs else loss


def retrain():
    """Phase: retrain on clean splits."""
    log.info("=" * 60)
    log.info("RETRAINING LegalBERT on leak-free splits")
    log.info("=" * 60)

    train_texts, train_labels = load_data(DATA_DIR / "train.csv")
    val_texts, val_labels     = load_data(DATA_DIR / "val.csv")
    test_texts, test_labels   = load_data(DATA_DIR / "test.csv")

    log.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE_MODEL), num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    # Datasets
    train_ds = ClauseDataset(train_texts, train_labels, tokenizer)
    val_ds   = ClauseDataset(val_texts, val_labels, tokenizer)

    # Pos weights
    pos_weight = compute_pos_weights(train_labels)
    log.info(f"Pos weights: {dict(zip(TAXONOMY, pos_weight.tolist()))}")

    # Training args
    output_dir = str(BASE_DIR / "training_output_v2_clean")
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,
        seed=SEED,
        report_to="none",
        logging_steps=10,
    )

    # Trainer
    trainer = WeightedTrainer(
        pos_weight=pos_weight,
        num_labels=NUM_LABELS,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    log.info(f"Training completed in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Threshold tuning on val set
    log.info("Tuning thresholds on validation set...")
    val_out = trainer.predict(val_ds)
    val_probs = torch.sigmoid(torch.tensor(val_out.predictions)).numpy()

    optimal_thresholds = {}
    threshold_list = []
    for i, t in enumerate(TAXONOMY):
        best_t, best_f1 = 0.5, 0.0
        for th in np.arange(0.1, 0.91, 0.05):
            preds = (val_probs[:, i] >= th).astype(int)
            f1 = f1_score(val_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, th
        threshold_list.append(round(best_t, 2))
        optimal_thresholds[t] = {"threshold": round(best_t, 2), "val_f1": round(best_f1, 4)}
        log.info(f"  {t}: threshold={best_t:.2f}, val_f1={best_f1:.4f}")

    # Save thresholds
    with open(EVAL_DIR / "optimal_thresholds.json", "w") as f:
        json.dump(optimal_thresholds, f, indent=2)

    # Evaluate on test set
    log.info("Evaluating on test set with tuned thresholds...")
    test_ds = ClauseDataset(test_texts, test_labels, tokenizer)
    test_out = trainer.predict(test_ds)
    test_probs = torch.sigmoid(torch.tensor(test_out.predictions)).numpy()

    test_preds = np.zeros_like(test_probs, dtype=int)
    for i in range(NUM_LABELS):
        test_preds[:, i] = (test_probs[:, i] >= threshold_list[i]).astype(int)

    f1_mac = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    f1_mic = f1_score(test_labels, test_preds, average="micro", zero_division=0)
    prec   = precision_score(test_labels, test_preds, average="macro", zero_division=0)
    rec    = recall_score(test_labels, test_preds, average="macro", zero_division=0)
    exact  = np.all(test_preds == test_labels, axis=1).mean()

    per_label = {}
    for i, t in enumerate(TAXONOMY):
        per_label[t] = {
            "precision": round(precision_score(test_labels[:, i], test_preds[:, i], zero_division=0), 4),
            "recall":    round(recall_score(test_labels[:, i], test_preds[:, i], zero_division=0), 4),
            "f1":        round(f1_score(test_labels[:, i], test_preds[:, i], zero_division=0), 4),
            "support":   int(test_labels[:, i].sum()),
            "threshold": threshold_list[i],
        }

    metrics = {
        "model": "LegalBERT v2 (retrained on leak-free splits)",
        "training_time_seconds": round(elapsed, 1),
        "dataset_sizes": {"train": len(train_texts), "val": len(val_texts), "test": len(test_texts)},
        "test_metrics": {
            "f1_macro": round(f1_mac, 4),
            "f1_micro": round(f1_mic, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "exact_match": round(exact, 4),
        },
        "per_label": per_label,
        "optimal_thresholds": optimal_thresholds,
    }

    with open(EVAL_DIR / "legalbert_final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Final metrics saved")

    # Export model
    log.info(f"Exporting model to {EXPORT_DIR}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(EXPORT_DIR))
    tokenizer.save_pretrained(str(EXPORT_DIR))

    # Save model metadata
    meta = {
        "version": "v2",
        "base_model": "nlpaueb/legal-bert-base-uncased",
        "training_time_seconds": round(elapsed, 1),
        "dataset": "leak-free policy-level splits",
        "test_metrics": metrics["test_metrics"],
    }
    with open(EXPORT_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Copy thresholds to model dir
    import shutil
    shutil.copy2(EVAL_DIR / "optimal_thresholds.json", EXPORT_DIR / "optimal_thresholds.json")

    print(f"\n{'='*60}")
    print(f"RETRAINED LegalBERT v2 RESULTS (leak-free)")
    print(f"{'='*60}")
    print(f"F1 Macro:  {f1_mac:.4f}")
    print(f"F1 Micro:  {f1_mic:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Exact Match: {exact:.4f}")
    print(f"Training time: {elapsed/60:.1f} min")
    print(f"{'='*60}")

    return metrics, tokenizer, model


def cross_validate(tokenizer):
    """Phase 3: 5-fold cross-validation on training data."""
    log.info("=" * 60)
    log.info("PHASE 3: 5-Fold Cross-Validation")
    log.info("=" * 60)

    train_texts, train_labels = load_data(DATA_DIR / "train.csv")

    # Use label powerset for stratification
    label_strings = ["".join(str(l) for l in row) for row in train_labels]
    label_codes = {s: i for i, s in enumerate(set(label_strings))}
    y_strat = np.array([label_codes[s] for s in label_strings])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_texts, y_strat)):
        log.info(f"\n--- Fold {fold+1}/5 ---")
        fold_train_texts = [train_texts[i] for i in tr_idx]
        fold_val_texts   = [train_texts[i] for i in va_idx]
        fold_train_labels = train_labels[tr_idx]
        fold_val_labels   = train_labels[va_idx]

        # Fresh model from base
        model = AutoModelForSequenceClassification.from_pretrained(
            str(BASE_MODEL), num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )

        # Freeze base, train only classifier head (fast CV)
        for param in model.base_model.parameters():
            param.requires_grad = False

        fold_train_ds = ClauseDataset(fold_train_texts, fold_train_labels, tokenizer)
        fold_val_ds   = ClauseDataset(fold_val_texts, fold_val_labels, tokenizer)

        pos_weight = compute_pos_weights(fold_train_labels)

        args = TrainingArguments(
            output_dir=str(BASE_DIR / f"cv_fold_{fold}"),
            num_train_epochs=3,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            learning_rate=5e-4,  # Higher LR for frozen base
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=False,
            seed=SEED,
            report_to="none",
            logging_steps=50,
        )

        trainer = WeightedTrainer(
            pos_weight=pos_weight,
            num_labels=NUM_LABELS,
            model=model,
            args=args,
            train_dataset=fold_train_ds,
            eval_dataset=fold_val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_result = trainer.evaluate()

        fold_metrics = {
            "fold": fold + 1,
            "f1_macro": eval_result["eval_f1_macro"],
            "f1_micro": eval_result["eval_f1_micro"],
            "precision_macro": eval_result["eval_precision_macro"],
            "recall_macro": eval_result["eval_recall_macro"],
        }
        fold_results.append(fold_metrics)
        log.info(f"  Fold {fold+1}: F1 Macro={fold_metrics['f1_macro']:.4f}, F1 Micro={fold_metrics['f1_micro']:.4f}")

        # Cleanup
        del model, trainer
        import shutil
        fold_dir = BASE_DIR / f"cv_fold_{fold}"
        if fold_dir.exists():
            shutil.rmtree(fold_dir)

    # Aggregate
    f1_macros = [f["f1_macro"] for f in fold_results]
    f1_micros = [f["f1_micro"] for f in fold_results]

    cv_report = {
        "method": "5-fold StratifiedKFold (frozen base, classifier head only)",
        "epochs_per_fold": 3,
        "folds": fold_results,
        "aggregate": {
            "mean_f1_macro": round(np.mean(f1_macros), 4),
            "std_f1_macro": round(np.std(f1_macros), 4),
            "mean_f1_micro": round(np.mean(f1_micros), 4),
            "std_f1_micro": round(np.std(f1_micros), 4),
        },
    }

    with open(EVAL_DIR / "cross_validation_metrics.json", "w") as f:
        json.dump(cv_report, f, indent=2)
    log.info(f"CV results saved to evaluation/cross_validation_metrics.json")

    print(f"\nCross-Validation: F1 Macro = {np.mean(f1_macros):.4f} +/- {np.std(f1_macros):.4f}")
    print(f"Cross-Validation: F1 Micro = {np.mean(f1_micros):.4f} +/- {np.std(f1_micros):.4f}")

    return cv_report


def main():
    # Phase: Retrain
    metrics, tokenizer, model = retrain()

    # Phase 3: Cross-validation
    cv = cross_validate(tokenizer)

    log.info("All phases complete!")


if __name__ == "__main__":
    main()
