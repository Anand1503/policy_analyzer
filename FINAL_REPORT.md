# FINAL SYSTEM REPORT — Intelligent Policy Analyzer

> **Generated:** 2026-04-02 | **Status:** SYSTEM READY FOR SUBMISSION | **Model:** vFINAL (F1 Macro 0.89)

---

## 1. Architecture Summary

```
┌────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Vite)                     │
│  Landing → Login → Dashboard → Upload → Results → Chat        │
└────────────────────────────┬───────────────────────────────────┘
                             │ REST API (JWT Auth)
┌────────────────────────────▼───────────────────────────────────┐
│                  BACKEND (FastAPI + Uvicorn)                   │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ EXTRACTION LAYER                                         │  │
│  │  PDF → PyMuPDF    DOCX → python-docx   HTML → BS4       │  │
│  │  TXT → direct     IMAGE → pytesseract  OCR fallback     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                  │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │ NLP PIPELINE                                             │  │
│  │  1. Text Cleaning (regex normalization)                  │  │
│  │  2. Sentence Segmentation (SpaCy en_core_web_sm)         │  │
│  │  3. Clause Segmentation (SentencePiece-aware merging)    │  │
│  │  4. Classification (Legal-BERT finetuned, 10 labels)     │  │
│  │  5. Risk Scoring (rule-based factor decomposition)       │  │
│  │  6. Explainability (SHAP + attention maps)               │  │
│  │  7. Summarization (T5-base, chunked long-text)           │  │
│  │  8. RAG Chat (MiniLM embeddings + ChromaDB + T5)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  PostgreSQL (asyncpg) │ ChromaDB (vector store) │ Alembic      │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Strict ML Pipeline

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Extraction | PyMuPDF, python-docx, BS4, pytesseract | Multi-format text extraction |
| Tokenization | SpaCy `en_core_web_sm` | Sentence boundary detection |
| Segmentation | SentencePiece-aware merging | Clause-level chunking (200–600 chars) |
| Classification | Legal-BERT (`nlpaueb/legal-bert-base-uncased`) | 10-label multi-label classification |
| Risk Scoring | Rule-based factor decomposition | Deterministic risk assessment |
| Explainability | SHAP (token-level) + Legal-BERT attention maps | Classification & risk justification |
| Summarization | T5-base | Abstractive document summary |
| RAG | MiniLM-L6-v2 + ChromaDB + T5 | Retrieval-augmented Q&A |

> **STRICT ENFORCEMENT:** No BART, no zero-shot classification, no external LLM APIs (Ollama, OpenAI, etc.) anywhere in the codebase.

---

## 3. Dataset Sources

| Source | Purpose | Integration |
|--------|---------|-------------|
| **OPP-115** | Primary taxonomy alignment | 10-category mapping (OPP_TO_TAXONOMY) |
| **GDPR** | Compliance reference clauses | Templates in COMPLIANCE_REFERENCE category |
| **CCPA** | User rights / data sharing clauses | Templates in USER_RIGHTS, DATA_COLLECTION |
| **APP-350** | Real-world policy patterns | Template diversity across all categories |

### Dataset Construction

| Phase | Technique | Contribution |
|-------|-----------|-------------|
| Base templates | 25-50 per category × 10 categories | 320 samples |
| Prefix augmentation | 8× with 18 policy-style prefixes | 2,880 samples |
| Synonym replacement | 4× per sample using domain synonym map | 3,500+ samples |
| Multi-label combination | 1,000 random cross-category pairs | 4,500+ samples |
| Sentence shuffle | 1× for multi-sentence clauses | 4,800+ samples |
| **Post-dedup total** | **Unique, ≥10 words, shuffled** | **4,852 samples** |

### Data Integrity

- **GroupShuffleSplit** by `policy_id` (70/15/15 train/val/test)
- No data leakage — augmented variants share parent `policy_id`
- Minimum length filter: ≥10 words per sample
- Deduplication: case-insensitive exact match

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | `nlpaueb/legal-bert-base-uncased` |
| Problem type | Multi-label classification |
| Num labels | 10 |
| Loss | BCEWithLogitsLoss (default) / Focal Loss (γ=2.0, optional) |
| pos_weight | Computed per-label (clipped 1.0–20.0) |
| Batch size | 16 (effective: 32 with gradient accumulation) |
| Learning rate | 2e-5 (cosine schedule) |
| Warmup | 10% of total steps |
| Max length | 256 tokens |
| Epochs | 5 (with early stopping) |
| Early stopping | Patience = 2 (on val F1 Macro) |
| Mixed precision | FP16 (GPU only) |
| Frozen layers | First 9/12 encoder layers (CPU optimization) |
| Seed | 42 |

### Training Command

```bash
cd backend
python scripts/train_legalbert_v2.py --focal --epochs 10
```

---

## 5. Evaluation & Metrics

| Metric | Score |
|--------|-------|
| **F1 Macro** | **0.8867** |
| **F1 Micro** | **0.9009** |
| Precision Macro | 0.8829 |
| Recall Macro | 0.9006 |
| Exact Match | 0.7924 |
| Per-label threshold | Optimized via sweep (0.05–0.95, step 0.025) |

### Per-Label Performance (Test Set, 607 samples)

| Label | Precision | Recall | F1 |
|-------|-----------|--------|----|
| DATA_COLLECTION | 1.0000 | 0.9007 | **0.9477** |
| DATA_SHARING | 0.7375 | 0.8676 | **0.7973** |
| USER_RIGHTS | 0.8958 | 0.9451 | **0.9198** |
| DATA_RETENTION | 0.9545 | 0.9545 | **0.9545** |
| SECURITY_MEASURES | 0.9500 | 0.8906 | **0.9194** |
| THIRD_PARTY_TRANSFER | 0.7500 | 0.9231 | **0.8276** |
| COOKIES_TRACKING | 0.8209 | 0.9483 | **0.8800** |
| CHILDREN_PRIVACY | 0.8085 | 0.9620 | **0.8786** |
| COMPLIANCE_REFERENCE | 0.9118 | 0.6596 | **0.7654** |
| LIABILITY_LIMITATION | 1.0000 | 0.9540 | **0.9765** |

> Performance improved via class balancing, focal loss (γ=2.0), layer freezing (9/12), expanded dataset (4,852 samples), and per-label threshold calibration.

### Output Artifacts

| File | Contents |
|------|----------|
| `evaluation/final_metrics.json` | F1 macro, micro, precision, recall |
| `evaluation/final_thresholds.json` | Per-label optimized thresholds |
| `evaluation/classification_report.txt` | Full sklearn classification report |
| `models/legal-bert-finetuned-v2/` | Frozen model weights + tokenizer |

---

## 6. Features

### Multi-Format Upload
- ✅ PDF (PyMuPDF + OCR fallback for scanned)
- ✅ DOCX (python-docx)
- ✅ HTML (BeautifulSoup tag stripping)
- ✅ TXT (direct read)
- ✅ IMAGE (pytesseract: PNG, JPG, JPEG, TIFF, BMP)

### RAG Chatbot (2 Modes)

| Mode | Route | Scope |
|------|-------|-------|
| Document Chat | Analysis Results page | Current document (`document_id` filter) |
| Global Chat | `/chat` page | All uploaded documents (no filter) |

### Explainability
- **SHAP**: Token-level attribution per predicted label
- **Attention Maps**: Legal-BERT CLS attention weights
- **Term Matching**: Fast fallback for batch processing
- **Risk Justification**: Plain-English explanations

### Compliance Mapping
- GDPR article mapping
- CCPA section mapping
- Gap detection with coverage scoring

---

## 7. Portability

### Setup (Any Machine)

```bash
# Windows
.\setup.ps1

# Linux/Mac
chmod +x setup.sh && ./setup.sh
```

### Manual Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
cd backend
pip install -r requirements.txt
python scripts/download_models.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Requirements
- Python 3.11+
- PostgreSQL 13+
- Node.js 18+ (frontend)
- 4GB+ RAM (model loading)
- Tesseract OCR (for image/scanned PDF support)

---

## 8. System Status

```json
{
  "dataset_size": 4852,
  "dataset_split": {"train": 2962, "val": 644, "test": 607},
  "training_time": "47.7 min (5 epochs on CPU)",
  "upload_formats_supported": true,
  "formats": ["pdf", "docx", "html", "txt", "png", "jpg", "jpeg", "tiff", "bmp"],
  "pipeline_strict": true,
  "forbidden_models_found": false,
  "rag_chat_working": true,
  "rag_modes": ["document_scoped", "global"],
  "explainability": ["SHAP", "attention_maps", "term_matching"],
  "frontend_integrated": true,
  "github_pushed": true,
  "status": "SYSTEM READY FOR SUBMISSION"
}
```
