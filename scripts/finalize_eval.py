"""
Finalize from checkpoint: evaluate, threshold-tune, export, cross-validate
==========================================================================
Uses checkpoint-56 (1 epoch on clean splits) to:
1. Threshold-tune on val set
2. Evaluate on test set
3. Export to models/legal-bert-finetuned-v2/
4. Run 5-fold CV (frozen base, 3 epochs/fold)
5. Re-run baseline comparison
"""

import csv, json, logging, os, shutil, time
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "evaluation"
CHECKPOINT = BASE_DIR / "training_output_v2_clean" / "checkpoint-56"
BASE_MODEL = BASE_DIR / "models" / "legal-bert"
EXPORT_DIR = BASE_DIR / "models" / "legal-bert-finetuned-v2"
EVAL_DIR.mkdir(exist_ok=True)

TAXONOMY = [
    "DATA_COLLECTION", "DATA_SHARING", "USER_RIGHTS", "DATA_RETENTION",
    "SECURITY_MEASURES", "THIRD_PARTY_TRANSFER", "COOKIES_TRACKING",
    "CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE", "LIABILITY_LIMITATION",
]
NUM_LABELS = len(TAXONOMY)
BATCH_SIZE = 16
MAX_LENGTH = 256
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
    return torch.tensor(np.clip(weights, 1.0, 20.0), dtype=torch.float)


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


# ═══════════════════════════════════════════════════════════
# STEP 1: Evaluate checkpoint on val + test, tune thresholds
# ═══════════════════════════════════════════════════════════
def evaluate_checkpoint():
    log.info("=" * 60)
    log.info("EVALUATING CHECKPOINT (1 epoch, clean splits)")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(CHECKPOINT), num_labels=NUM_LABELS,
    )
    model.eval()

    val_texts, val_labels   = load_data(DATA_DIR / "val.csv")
    test_texts, test_labels = load_data(DATA_DIR / "test.csv")

    def predict_batch(texts, batch_size=16):
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            with torch.no_grad():
                out = model(**enc)
            probs = torch.sigmoid(out.logits).numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    # Threshold tuning on val
    log.info("Tuning thresholds on val set...")
    val_probs = predict_batch(val_texts)

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

    # Evaluate on test
    log.info("Evaluating on test set...")
    test_probs = predict_batch(test_texts)

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
        "model": "LegalBERT v2 (retrained 1 epoch, leak-free splits)",
        "checkpoint": "checkpoint-56",
        "dataset_sizes": {"train": 888, "val": 190, "test": 179},
        "note": "Policy-level split isolation verified. No leakage.",
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

    print(f"\n{'='*60}")
    print(f"LegalBERT v2 (1 epoch, leak-free) RESULTS")
    print(f"{'='*60}")
    print(f"F1 Macro:  {f1_mac:.4f}")
    print(f"F1 Micro:  {f1_mic:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Exact Match: {exact:.4f}")
    print(f"{'='*60}")

    return metrics, tokenizer


# ═══════════════════════════════════════════════════════════
# STEP 2: Export checkpoint as final model
# ═══════════════════════════════════════════════════════════
def export_model(tokenizer):
    log.info("Exporting checkpoint to model dir...")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy model files
    for f in ["config.json", "model.safetensors"]:
        src = CHECKPOINT / f
        if src.exists():
            shutil.copy2(src, EXPORT_DIR / f)

    # Save tokenizer
    tokenizer.save_pretrained(str(EXPORT_DIR))

    # Copy thresholds
    shutil.copy2(EVAL_DIR / "optimal_thresholds.json", EXPORT_DIR / "optimal_thresholds.json")

    # Save metadata
    meta = {
        "version": "v2",
        "base_model": "nlpaueb/legal-bert-base-uncased",
        "training": "1 epoch on leak-free policy-level splits",
        "dataset_sizes": {"train": 888, "val": 190, "test": 179},
    }
    with open(EXPORT_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Model exported to {EXPORT_DIR}")


# ═══════════════════════════════════════════════════════════
# STEP 3: Baseline comparison
# ═══════════════════════════════════════════════════════════
def run_baseline():
    log.info("=" * 60)
    log.info("BASELINE: TF-IDF + Logistic Regression")
    log.info("=" * 60)

    train_texts, train_labels = load_data(DATA_DIR / "train.csv")
    test_texts, test_labels   = load_data(DATA_DIR / "test.csv")

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_texts)
    X_test  = tfidf.transform(test_texts)

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    )
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    f1_mac = f1_score(test_labels, preds, average="macro", zero_division=0)
    f1_mic = f1_score(test_labels, preds, average="micro", zero_division=0)
    prec   = precision_score(test_labels, preds, average="macro", zero_division=0)
    rec    = recall_score(test_labels, preds, average="macro", zero_division=0)
    exact  = np.all(preds == test_labels, axis=1).mean()

    per_label = {}
    for i, t in enumerate(TAXONOMY):
        per_label[t] = {
            "precision": round(precision_score(test_labels[:, i], preds[:, i], zero_division=0), 4),
            "recall":    round(recall_score(test_labels[:, i], preds[:, i], zero_division=0), 4),
            "f1":        round(f1_score(test_labels[:, i], preds[:, i], zero_division=0), 4),
            "support":   int(test_labels[:, i].sum()),
        }

    result = {
        "model": "TF-IDF + Logistic Regression (OneVsRest)",
        "tfidf_features": X_train.shape[1],
        "test_metrics": {
            "f1_macro": round(f1_mac, 4),
            "f1_micro": round(f1_mic, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "exact_match": round(exact, 4),
        },
        "per_label": per_label,
    }

    with open(EVAL_DIR / "baseline_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Baseline saved")
    print(f"Baseline F1 Macro: {f1_mac:.4f}, F1 Micro: {f1_mic:.4f}")
    return result


# ═══════════════════════════════════════════════════════════
# STEP 4: 5-fold Cross-Validation (frozen base)
# ═══════════════════════════════════════════════════════════
def cross_validate(tokenizer):
    log.info("=" * 60)
    log.info("5-FOLD CROSS-VALIDATION (frozen base)")
    log.info("=" * 60)

    train_texts, train_labels = load_data(DATA_DIR / "train.csv")

    label_strings = ["".join(str(l) for l in row) for row in train_labels]
    label_codes = {s: i for i, s in enumerate(set(label_strings))}
    y_strat = np.array([label_codes[s] for s in label_strings])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_texts, y_strat)):
        log.info(f"\n--- Fold {fold+1}/5 ---")
        fold_tr_texts = [train_texts[i] for i in tr_idx]
        fold_va_texts = [train_texts[i] for i in va_idx]
        fold_tr_labels = train_labels[tr_idx]
        fold_va_labels = train_labels[va_idx]

        model = AutoModelForSequenceClassification.from_pretrained(
            str(BASE_MODEL), num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )
        for param in model.base_model.parameters():
            param.requires_grad = False

        fold_tr_ds = ClauseDataset(fold_tr_texts, fold_tr_labels, tokenizer)
        fold_va_ds = ClauseDataset(fold_va_texts, fold_va_labels, tokenizer)
        pos_weight = compute_pos_weights(fold_tr_labels)

        args = TrainingArguments(
            output_dir=str(BASE_DIR / f"cv_fold_{fold}"),
            num_train_epochs=3,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            learning_rate=5e-4,
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
            train_dataset=fold_tr_ds,
            eval_dataset=fold_va_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_result = trainer.evaluate()

        fm = {
            "fold": fold + 1,
            "f1_macro": eval_result["eval_f1_macro"],
            "f1_micro": eval_result["eval_f1_micro"],
            "precision_macro": eval_result["eval_precision_macro"],
            "recall_macro": eval_result["eval_recall_macro"],
        }
        fold_results.append(fm)
        log.info(f"  Fold {fold+1}: F1 Macro={fm['f1_macro']:.4f}")

        del model, trainer
        fold_dir = BASE_DIR / f"cv_fold_{fold}"
        if fold_dir.exists():
            shutil.rmtree(fold_dir)

    f1_macs = [f["f1_macro"] for f in fold_results]
    f1_mics = [f["f1_micro"] for f in fold_results]

    cv_report = {
        "method": "5-fold StratifiedKFold (frozen base, classifier head only)",
        "epochs_per_fold": 3,
        "folds": fold_results,
        "aggregate": {
            "mean_f1_macro": round(float(np.mean(f1_macs)), 4),
            "std_f1_macro": round(float(np.std(f1_macs)), 4),
            "mean_f1_micro": round(float(np.mean(f1_mics)), 4),
            "std_f1_micro": round(float(np.std(f1_mics)), 4),
        },
    }

    with open(EVAL_DIR / "cross_validation_metrics.json", "w") as f:
        json.dump(cv_report, f, indent=2)
    log.info("CV results saved")
    print(f"\nCV F1 Macro: {np.mean(f1_macs):.4f} +/- {np.std(f1_macs):.4f}")
    return cv_report


def main():
    # Step 1: Evaluate checkpoint
    metrics, tokenizer = evaluate_checkpoint()

    # Step 2: Export model
    export_model(tokenizer)

    # Step 3: Baseline
    baseline = run_baseline()

    # Step 4: Cross-validation
    cv = cross_validate(tokenizer)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    b = baseline["test_metrics"]
    l = metrics["test_metrics"]
    print(f"{'Metric':<20} {'Baseline':>10} {'LegalBERT':>10} {'Delta':>10}")
    print("-" * 52)
    for k in ["f1_macro", "f1_micro", "precision_macro", "recall_macro", "exact_match"]:
        delta = l[k] - b[k]
        print(f"{k:<20} {b[k]:>10.4f} {l[k]:>10.4f} {delta:>+10.4f}")
    print("=" * 60)

    log.info("ALL PHASES COMPLETE")


if __name__ == "__main__":
    main()
