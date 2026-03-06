# Data Directory

This directory contains processed datasets for model training and evaluation.

## Expected Files

- `processed/train.csv` — Training set
- `processed/val.csv` — Validation set
- `processed/test.csv` — Test set

## Setup

Run the OPP-115 data preparation script:

```bash
python scripts/prepare_opp115.py
```

This will generate the processed CSV files with the 10-label taxonomy mapping.
