# Intelligent Policy Analyzer — Backend

FastAPI backend for AI-powered privacy policy analysis, featuring multi-label clause classification, risk scoring, explainability, summarization, and RAG-based question answering.

## Architecture

- **Framework**: FastAPI + SQLAlchemy + Alembic
- **Database**: PostgreSQL with multi-tenant RBAC
- **ML Pipeline**:
  - **Classification**: Legal-BERT (fine-tuned on OPP-115 taxonomy)
  - **Summarization**: T5-base
  - **Explainability**: SHAP token-level importance
  - **QA**: RAG pipeline (Sentence-Transformers + ChromaDB)
  - **NER**: SpaCy + custom legal entity recognition
  - **Segmentation**: SentencePiece clause segmentation

## Project Structure

```
app/            → FastAPI application (routes, services, ML modules)
alembic/        → Database migration scripts
scripts/        → Training, evaluation, and utility scripts
models/         → ML model weights (not tracked in git)
data/           → Datasets (not tracked in git)
evaluation/     → Evaluation reports (not tracked in git)
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Configure environment
cp .env.example .env
# Edit .env with your database credentials and model paths

# Run database migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload --port 8000
```

## ML Models (Local Storage)

Models are stored locally and not tracked in git. See `models/README.md` for download instructions.

Required models:
- `models/legal-bert/` — Base Legal-BERT
- `models/legal-bert-finetuned-v2/` — Fine-tuned classifier
- `models/t5-base/` — T5 summarizer
- `models/all-MiniLM-L6-v2/` — Sentence embeddings

## API Endpoints

- `GET /health/live` — Health check
- `POST /api/v1/auth/register` — User registration
- `POST /api/v1/auth/login` — Authentication
- `POST /api/v1/documents/upload` — Upload policy document
- `POST /api/v1/analysis/classify/{id}` — Classify document clauses
- `GET /api/v1/analysis/results/{id}` — Get analysis results
- `POST /api/v1/qa/ask` — RAG question answering
- `GET /docs` — Interactive API documentation (Swagger)

## License

Private — Final Year Project
