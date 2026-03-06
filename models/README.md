# Local ML Models

All ML models are stored locally for offline development and deployment.

## Models

| Directory | Model | Size | Source |
|-----------|-------|------|--------|
| `legal-bert/` | nlpaueb/legal-bert-base-uncased | 418 MB | HuggingFace Hub |
| `t5-base/` | t5-base | 852 MB | HuggingFace Hub |
| `all-MiniLM-L6-v2/` | sentence-transformers/all-MiniLM-L6-v2 | 87 MB | HuggingFace Hub |

SpaCy `en_core_web_sm` is installed into the venv via `python -m spacy download en_core_web_sm`.

## Download Script

```bash
# From backend/ directory with venv activated:
python scripts/download_models.py
```

## Total Storage: ~1.36 GB
