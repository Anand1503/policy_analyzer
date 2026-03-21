# Intelligent Policy Analyzer

AI-powered privacy policy analysis system using Legal-BERT, T5, and RAG — fully local, no external APIs.

## Features

- **Document Upload** — PDF, DOCX, TXT, HTML with automatic text extraction (PyMuPDF + Tesseract OCR fallback)
- **Clause Extraction** — SpaCy + SentencePiece segmentation into semantic clause units
- **Clause Classification** — Fine-tuned Legal-BERT multi-label classifier (10 categories)
- **Risk Scoring** — Hybrid rule-based + ML risk assessment per clause
- **Explainability** — SHAP-based explanations for classification decisions
- **Compliance Mapping** — GDPR and CCPA automated compliance evaluation
- **Summarization** — T5-based simplified document summaries
- **RAG Chat** — Ask questions about uploaded documents with source citations
- **JWT Authentication** — Secure user accounts with rate limiting
- **Dashboard** — Interactive results with risk gauge, clause explorer, and recommendations

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.11+, Gunicorn |
| Database | PostgreSQL 16 + SQLAlchemy (async) + Alembic |
| ML/NLP | Legal-BERT, T5-base, SHAP, SpaCy |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Auth | JWT (python-jose + bcrypt) |
| Frontend | React 19 + Vite + CSS |
| DevOps | Docker + Docker Compose |

## Project Structure

```
policy_analyser/
├── backend/
│   ├── app/
│   │   ├── api/           # Auth, documents, analysis endpoints
│   │   ├── core/          # Config, security, database, rate limiting
│   │   ├── models/        # SQLAlchemy ORM models
│   │   ├── schemas/       # Pydantic request/response schemas
│   │   ├── services/      # Analysis pipeline service
│   │   ├── ml/            # Classifier, risk scorer, summarizer, RAG, explainability
│   │   ├── extractors/    # PDF, DOCX, HTML, TXT extractors
│   │   └── main.py        # FastAPI entry point
│   ├── scripts/           # Training, model download, setup scripts
│   ├── models/            # Local ML models (git-ignored)
│   ├── requirements.txt   # Pinned dependencies
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/    # Upload, Results, Chat, Navigation
│   │   ├── pages/         # Landing, Login, Dashboard, Analysis
│   │   ├── services/      # API client with JWT interceptor
│   │   └── context/       # Auth context
│   └── package.json
├── docker-compose.yml
├── setup.ps1              # Windows setup
├── setup.sh               # Linux/Mac setup
└── README.md
```

## Quick Start

### Automated Setup

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

**Linux/Mac:**
```bash
chmod +x setup.sh && ./setup.sh
```

This will: create venv → install dependencies → download ML models → setup directories.

### Manual Setup

#### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+

#### 1. Database
```bash
createdb policy_db
```

#### 2. Backend
```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
# Linux/Mac: source venv/bin/activate
pip install -r backend/requirements.txt
python backend/scripts/download_models.py   # ~1.4 GB models
cp backend/.env.example backend/.env        # Edit credentials
cd backend
alembic upgrade head
uvicorn app.main:app --port 8000 --reload
```

#### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

#### 4. Docker (Alternative)
```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register user |
| POST | `/api/v1/auth/login` | Login, get JWT |
| POST | `/api/v1/documents/upload` | Upload document |
| GET | `/api/v1/documents/` | List documents |
| POST | `/api/v1/analysis/analyze/{id}` | Trigger background analysis |
| POST | `/api/v1/analysis/run/{id}` | Run full pipeline (synchronous) |
| GET | `/api/v1/analysis/results/{id}` | Get analysis results |
| POST | `/api/v1/analysis/chat` | RAG Q&A with document scope |
| GET | `/api/v1/system/health` | System health check |

## ML Pipeline

```
Upload → Extract (PyMuPDF/OCR) → Clean → Segment (SpaCy)
    → Classify (Legal-BERT) → Score Risks → Explain (SHAP)
    → Summarize (T5) → Compliance (GDPR/CCPA)
    → Embed (MiniLM) → Store (ChromaDB) → Ready for Q&A
```

## Model Training

To retrain the Legal-BERT classifier:

```bash
cd backend
python scripts/train_legalbert_v2.py [--focal] [--fp16] [--epochs 10]
```

Target: F1 Macro ≥ 0.70 on test set.

## License

MIT — Final Year Project
