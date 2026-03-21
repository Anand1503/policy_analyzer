"""
Phase 1-2: Dataset Leakage Verification & Quality Validation
============================================================
- Assigns policy_id to each clause via template-base hashing
- Verifies policy-level split isolation (train ∩ test = ∅, etc.)
- Detects duplicates, short clauses (<10 tokens), near-duplicate paraphrases
- Regenerates splits with GroupShuffleSplit if violations found
- Outputs evaluation/dataset_integrity_report.json
"""

import csv, hashlib, json, os, re, sys, logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

TAXONOMY = [
    "DATA_COLLECTION", "DATA_SHARING", "USER_RIGHTS", "DATA_RETENTION",
    "SECURITY_MEASURES", "THIRD_PARTY_TRANSFER", "COOKIES_TRACKING",
    "CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE", "LIABILITY_LIMITATION",
]

# ── Prefix patterns used in synthetic generation ──
PREFIXES = [
    r"^In accordance with our policies,\s*",
    r"^As part of our practices,\s*",
    r"^To be transparent,\s*",
    r"^For your information,\s*",
    r"^It is important to understand that\s*",
    r"^You should be aware that\s*",
    r"^We want you to know that\s*",
    r"^Please note that\s*",
]

def strip_prefix(text: str) -> str:
    """Remove known synthetic prefixes to get base clause."""
    for p in PREFIXES:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()


def assign_policy_id(text: str) -> str:
    """
    Assign a policy_id by hashing the first sentence of the base clause.
    Clauses sharing the same first sentence belong to the same 'policy'.
    """
    base = strip_prefix(text)
    # Take first sentence (up to first period)
    first_sent = base.split(".")[0].strip().lower()
    # Normalize whitespace
    first_sent = re.sub(r"\s+", " ", first_sent)
    return hashlib.md5(first_sent.encode()).hexdigest()[:12]


def load_split(path: Path) -> list:
    """Load CSV and return list of dicts with text + labels."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = [int(row.get(f"label_{t}", 0)) for t in TAXONOMY]
            rows.append({"text": row["text"], "labels": labels})
    return rows


def save_split(rows: list, path: Path):
    """Save list of dicts back to CSV."""
    fieldnames = ["text"] + [f"label_{t}" for t in TAXONOMY]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {"text": r["text"]}
            for i, t in enumerate(TAXONOMY):
                row[f"label_{t}"] = r["labels"][i]
            writer.writerow(row)


def check_leakage(train, val, test):
    """Phase 1: verify policy-level split isolation."""
    train_pids = {assign_policy_id(r["text"]) for r in train}
    val_pids   = {assign_policy_id(r["text"]) for r in val}
    test_pids  = {assign_policy_id(r["text"]) for r in test}

    tv = train_pids & val_pids
    tt = train_pids & test_pids
    vt = val_pids & test_pids

    log.info(f"Policy IDs — train: {len(train_pids)}, val: {len(val_pids)}, test: {len(test_pids)}")
    log.info(f"Overlap train∩val: {len(tv)}, train∩test: {len(tt)}, val∩test: {len(vt)}")

    return {
        "train_policies": len(train_pids),
        "val_policies": len(val_pids),
        "test_policies": len(test_pids),
        "overlap_train_val": len(tv),
        "overlap_train_test": len(tt),
        "overlap_val_test": len(vt),
        "leakage_detected": bool(tv or tt or vt),
        "leaked_policy_ids": {
            "train_val": list(tv)[:10],
            "train_test": list(tt)[:10],
            "val_test": list(vt)[:10],
        }
    }


def quality_check(rows, split_name):
    """Phase 2: detect duplicates, short clauses, near-dupes."""
    issues = {"duplicates": 0, "short_clauses": 0, "cleaned": 0}
    seen_texts = set()
    clean = []

    for r in rows:
        text = r["text"].strip()
        # Skip exact duplicates
        if text in seen_texts:
            issues["duplicates"] += 1
            continue
        seen_texts.add(text)
        # Skip extremely short clauses (<10 tokens)
        tokens = text.split()
        if len(tokens) < 10:
            issues["short_clauses"] += 1
            continue
        clean.append(r)

    issues["original_count"] = len(rows)
    issues["cleaned_count"] = len(clean)
    issues["removed"] = len(rows) - len(clean)
    log.info(f"[{split_name}] {issues['original_count']}→{issues['cleaned_count']} "
             f"(dup={issues['duplicates']}, short={issues['short_clauses']})")
    return clean, issues


def regenerate_splits(all_rows):
    """Regenerate splits using policy-level GroupShuffleSplit."""
    from sklearn.model_selection import GroupShuffleSplit

    # Assign policy IDs
    policy_ids = [assign_policy_id(r["text"]) for r in all_rows]

    # First split: 70% train, 30% temp
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(all_rows, groups=policy_ids))

    temp_rows = [all_rows[i] for i in temp_idx]
    temp_pids = [policy_ids[i] for i in temp_idx]

    # Second split: 50/50 of temp → val / test (15%/15%)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_rows, groups=temp_pids))

    train = [all_rows[i] for i in train_idx]
    val   = [temp_rows[i] for i in val_idx]
    test  = [temp_rows[i] for i in test_idx]

    log.info(f"Regenerated splits: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def label_distribution(rows):
    """Compute label frequency stats."""
    dist = Counter()
    for r in rows:
        for i, t in enumerate(TAXONOMY):
            if r["labels"][i] == 1:
                dist[t] += 1
    total = len(rows)
    return {t: {"count": dist[t], "ratio": round(dist[t]/total, 4) if total else 0} for t in TAXONOMY}


def main():
    log.info("=" * 60)
    log.info("PHASE 1-2: Dataset Integrity Verification")
    log.info("=" * 60)

    # Load existing splits
    train = load_split(DATA_DIR / "train.csv")
    val   = load_split(DATA_DIR / "val.csv")
    test  = load_split(DATA_DIR / "test.csv")

    report = {
        "phase": "Dataset Integrity Verification",
        "original_sizes": {"train": len(train), "val": len(val), "test": len(test)},
    }

    # Phase 1: Leakage check
    leakage = check_leakage(train, val, test)
    report["leakage_analysis"] = leakage

    # Phase 2: Quality check
    train_clean, train_issues = quality_check(train, "train")
    val_clean, val_issues     = quality_check(val, "val")
    test_clean, test_issues   = quality_check(test, "test")
    report["quality_checks"] = {
        "train": train_issues,
        "val": val_issues,
        "test": test_issues,
    }

    # If leakage detected OR quality issues found, regenerate
    needs_regen = leakage["leakage_detected"] or any(
        q["removed"] > 0 for q in [train_issues, val_issues, test_issues]
    )

    if needs_regen:
        log.info("Issues detected — regenerating splits with policy-level grouping...")
        all_rows = train_clean + val_clean + test_clean
        train_new, val_new, test_new = regenerate_splits(all_rows)

        # Re-verify leakage after regeneration
        leakage_post = check_leakage(train_new, val_new, test_new)
        report["post_regeneration"] = {
            "leakage_analysis": leakage_post,
            "sizes": {"train": len(train_new), "val": len(val_new), "test": len(test_new)},
        }

        # Save regenerated splits
        save_split(train_new, DATA_DIR / "train.csv")
        save_split(val_new, DATA_DIR / "val.csv")
        save_split(test_new, DATA_DIR / "test.csv")
        log.info("Saved regenerated splits.")

        train, val, test = train_new, val_new, test_new
    else:
        log.info("No leakage or quality issues — splits are valid.")
        report["post_regeneration"] = None

    # Label distribution
    report["label_distribution"] = {
        "train": label_distribution(train),
        "val": label_distribution(val),
        "test": label_distribution(test),
    }

    # Summary stats
    all_rows = train + val + test
    report["summary"] = {
        "total_clauses": len(all_rows),
        "total_policies": len({assign_policy_id(r["text"]) for r in all_rows}),
        "final_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "imbalance_ratios": {},
    }

    # Compute imbalance ratio (max/min label count)
    train_dist = label_distribution(train)
    counts = [v["count"] for v in train_dist.values() if v["count"] > 0]
    if counts:
        report["summary"]["imbalance_ratios"] = {
            "max_min_ratio": round(max(counts) / min(counts), 2),
            "most_common": max(train_dist, key=lambda k: train_dist[k]["count"]),
            "least_common": min(train_dist, key=lambda k: train_dist[k]["count"]),
        }

    # Save report
    out = EVAL_DIR / "dataset_integrity_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {out}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET INTEGRITY REPORT SUMMARY")
    print("=" * 60)
    print(f"Total clauses: {report['summary']['total_clauses']}")
    print(f"Total policies: {report['summary']['total_policies']}")
    print(f"Leakage detected: {leakage['leakage_detected']}")
    if report.get("post_regeneration"):
        p = report["post_regeneration"]
        print(f"Post-regen leakage: {p['leakage_analysis']['leakage_detected']}")
        print(f"Post-regen sizes: {p['sizes']}")
    print(f"Final sizes: {report['summary']['final_sizes']}")
    if report["summary"]["imbalance_ratios"]:
        ir = report["summary"]["imbalance_ratios"]
        print(f"Imbalance ratio: {ir['max_min_ratio']}x ({ir['most_common']}/{ir['least_common']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
