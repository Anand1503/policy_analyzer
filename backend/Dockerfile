# ── BUILD STAGE ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── PRODUCTION STAGE ─────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime deps: PostgreSQL client, curl for health checks, Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

COPY . .

# Create required directories
RUN mkdir -p uploads/raw_documents uploads/extracted_text data/vector_store

EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Production entrypoint with Gunicorn + Uvicorn workers
CMD ["python", "-m", "gunicorn", "app.main:app", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "2", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--timeout", "300", \
    "--graceful-timeout", "30", \
    "--access-logfile", "-", \
    "--error-logfile", "-"]
