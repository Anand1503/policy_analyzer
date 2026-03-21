# Data Directory

This directory contains processed datasets and vector store data.
Data files are **not tracked by git**.

## Structure

```
data/
├── processed/          # Training/validation/test CSVs
│   ├── train.csv       # Training set (~70%)
│   ├── val.csv         # Validation set (~15%)
│   └── test.csv        # Test set (~15%)
├── vector_store/       # ChromaDB vector embeddings
└── README.md           # This file
```

## Generating Training Data

The training script generates the dataset automatically:

```bash
python scripts/train_legalbert_v2.py
```

This creates `data/processed/{train,val,test}.csv` with:
- 3000+ clause samples across 10 privacy policy categories
- Multi-label one-hot encoding
- Policy-level GroupShuffleSplit (no data leakage)

## Dataset Format

CSV with columns:
- `text` — Privacy policy clause text
- `label_DATA_COLLECTION` — Binary (0/1)
- `label_DATA_SHARING` — Binary (0/1)
- `label_USER_RIGHTS` — Binary (0/1)
- `label_DATA_RETENTION` — Binary (0/1)
- `label_SECURITY_MEASURES` — Binary (0/1)
- `label_THIRD_PARTY_TRANSFER` — Binary (0/1)
- `label_COOKIES_TRACKING` — Binary (0/1)
- `label_CHILDREN_PRIVACY` — Binary (0/1)
- `label_COMPLIANCE_REFERENCE` — Binary (0/1)
- `label_LIABILITY_LIMITATION` — Binary (0/1)

## Taxonomy (10 labels)

1. `DATA_COLLECTION` — Information gathering practices
2. `DATA_SHARING` — Data disclosure to third parties
3. `USER_RIGHTS` — User control and opt-out options
4. `DATA_RETENTION` — How long data is stored
5. `SECURITY_MEASURES` — Protection mechanisms
6. `THIRD_PARTY_TRANSFER` — Cross-border/processor transfers
7. `COOKIES_TRACKING` — Tracking technologies
8. `CHILDREN_PRIVACY` — Minor-specific protections
9. `COMPLIANCE_REFERENCE` — Regulatory references
10. `LIABILITY_LIMITATION` — Legal disclaimers
