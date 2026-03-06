"""
Phase 4: Baseline Model Comparison — TF-IDF + Logistic Regression
Phase 5: Threshold Verification & Final LegalBERT Metrics
==========================================================
Outputs:
  - evaluation/baseline_metrics.json
  - evaluation/legalbert_final_metrics.json
"""

import csv, json, logging, sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "evaluation"
MODEL_DIR = BASE_DIR / "models" / "legal-bert-finetuned-v2"
EVAL_DIR.mkdir(exist_ok=True)

TAXONOMY = [
    "DATA_COLLECTION", "DATA_SHARING", "USER_RIGHTS", "DATA_RETENTION",
    "SECURITY_MEASURES", "THIRD_PARTY_TRANSFER", "COOKIES_TRACKING",
    "CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE", "LIABILITY_LIMITATION",
]


def load_data(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append([int(row.get(f"label_{t}", 0)) for t in TAXONOMY])
    return texts, np.array(labels)


# ═══════════════════════════════════════════════════════════
# PHASE 4 — Baseline: TF-IDF + Logistic Regression
# ═══════════════════════════════════════════════════════════
def run_baseline():
    log.info("=" * 60)
    log.info("PHASE 4: Baseline — TF-IDF + Logistic Regression")
    log.info("=" * 60)

    train_texts, train_labels = load_data(DATA_DIR / "train.csv")
    test_texts, test_labels   = load_data(DATA_DIR / "test.csv")

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_texts)
    X_test  = tfidf.transform(test_texts)

    log.info(f"TF-IDF: {X_train.shape[1]} features, train={X_train.shape[0]}, test={X_test.shape[0]}")

    # Multi-label classifier
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    )
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    # Metrics
    f1_mac = f1_score(test_labels, preds, average="macro", zero_division=0)
    f1_mic = f1_score(test_labels, preds, average="micro", zero_division=0)
    prec   = precision_score(test_labels, preds, average="macro", zero_division=0)
    rec    = recall_score(test_labels, preds, average="macro", zero_division=0)
    exact  = np.all(preds == test_labels, axis=1).mean()

    # Per-label metrics
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
        "ngram_range": [1, 2],
        "test_metrics": {
            "f1_macro": round(f1_mac, 4),
            "f1_micro": round(f1_mic, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "exact_match": round(exact, 4),
        },
        "per_label": per_label,
    }

    out = EVAL_DIR / "baseline_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info(f"Baseline saved to {out}")

    print(f"\nBaseline F1 Macro: {f1_mac:.4f}")
    print(f"Baseline F1 Micro: {f1_mic:.4f}")
    print(f"Baseline Exact Match: {exact:.4f}")

    return result


# ═══════════════════════════════════════════════════════════
# PHASE 5 — Threshold Verification & Final LegalBERT Metrics
# ═══════════════════════════════════════════════════════════
def run_threshold_verification():
    log.info("=" * 60)
    log.info("PHASE 5: Threshold Verification & Final Metrics")
    log.info("=" * 60)

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Load model
    log.info(f"Loading model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    # Load thresholds
    thresh_path = EVAL_DIR / "optimal_thresholds.json"
    with open(thresh_path, "r") as f:
        thresholds_data = json.load(f)
    thresholds = [thresholds_data[t]["threshold"] for t in TAXONOMY]
    log.info(f"Thresholds: {dict(zip(TAXONOMY, thresholds))}")

    # Load val and test data
    val_texts, val_labels   = load_data(DATA_DIR / "val.csv")
    test_texts, test_labels = load_data(DATA_DIR / "test.csv")

    def predict_batch(texts, batch_size=16):
        all_logits = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                out = model(**enc)
            probs = torch.sigmoid(out.logits).numpy()
            all_logits.append(probs)
        return np.vstack(all_logits)

    # Step 1: Re-derive thresholds from validation set
    log.info("Re-deriving optimal thresholds from validation set...")
    val_probs = predict_batch(val_texts)

    new_thresholds = []
    threshold_audit = {}
    for i, t in enumerate(TAXONOMY):
        best_t, best_f1 = 0.5, 0.0
        for th in np.arange(0.05, 0.96, 0.05):
            preds = (val_probs[:, i] >= th).astype(int)
            f1 = f1_score(val_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, th
        new_thresholds.append(round(best_t, 2))
        threshold_audit[t] = {
            "original_threshold": thresholds[i],
            "recomputed_threshold": round(best_t, 2),
            "val_f1": round(best_f1, 4),
            "match": bool(abs(thresholds[i] - best_t) < 0.06),
        }
        log.info(f"  {t}: original={thresholds[i]}, recomputed={best_t:.2f}, val_f1={best_f1:.4f}")

    # Step 2: Evaluate on test set with recomputed thresholds
    log.info("Evaluating test set with recomputed thresholds...")
    test_probs = predict_batch(test_texts)

    test_preds = np.zeros_like(test_probs, dtype=int)
    for i in range(len(TAXONOMY)):
        test_preds[:, i] = (test_probs[:, i] >= new_thresholds[i]).astype(int)

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
            "threshold": new_thresholds[i],
        }

    result = {
        "model": "LegalBERT v2 (fine-tuned)",
        "model_path": str(MODEL_DIR),
        "note": "Evaluated on regenerated test set with policy-level split isolation",
        "threshold_audit": threshold_audit,
        "test_metrics": {
            "f1_macro": round(f1_mac, 4),
            "f1_micro": round(f1_mic, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "exact_match": round(exact, 4),
        },
        "per_label": per_label,
    }

    out = EVAL_DIR / "legalbert_final_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info(f"Final metrics saved to {out}")

    # Also save updated thresholds
    updated_thresh = {t: {"threshold": new_thresholds[i], "val_f1": threshold_audit[t]["val_f1"]} for i, t in enumerate(TAXONOMY)}
    thresh_out = EVAL_DIR / "optimal_thresholds.json"
    with open(thresh_out, "w", encoding="utf-8") as f:
        json.dump(updated_thresh, f, indent=2)
    log.info(f"Updated thresholds saved to {thresh_out}")

    print(f"\nLegalBERT v2 (regenerated test set):")
    print(f"  F1 Macro: {f1_mac:.4f}")
    print(f"  F1 Micro: {f1_mic:.4f}")
    print(f"  Exact Match: {exact:.4f}")

    return result


def main():
    baseline = run_baseline()
    legalbert = run_threshold_verification()

    # Comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    b = baseline["test_metrics"]
    l = legalbert["test_metrics"]
    print(f"{'Metric':<20} {'Baseline':>10} {'LegalBERT':>10} {'Delta':>10}")
    print("-" * 52)
    for k in ["f1_macro", "f1_micro", "precision_macro", "recall_macro", "exact_match"]:
        delta = l[k] - b[k]
        print(f"{k:<20} {b[k]:>10.4f} {l[k]:>10.4f} {delta:>+10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
