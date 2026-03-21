#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Intelligent Policy Analyzer — Linux/Mac Setup
# ═══════════════════════════════════════════════════════════
#
# Usage: chmod +x setup.sh && ./setup.sh
#

set -e

echo ""
echo "═══════════════════════════════════════════════════"
echo " Intelligent Policy Analyzer — Setup"
echo "═══════════════════════════════════════════════════"
echo ""

# ──────────────────────────────────────────────────────────
# Step 1: Check Python
# ──────────────────────────────────────────────────────────
echo "[1/8] Checking Python installation..."
if command -v python3 &> /dev/null; then
    PY=$(python3 --version)
    echo "  OK: $PY"
    PYTHON=python3
elif command -v python &> /dev/null; then
    PY=$(python --version)
    echo "  OK: $PY"
    PYTHON=python
else
    echo "  FAIL: Python not found. Install Python 3.11+"
    exit 1
fi

# ──────────────────────────────────────────────────────────
# Step 2: Check PostgreSQL
# ──────────────────────────────────────────────────────────
echo "[2/8] Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "  OK: $(psql --version)"
else
    echo "  WARN: PostgreSQL not found. Install it or use Docker."
fi

# ──────────────────────────────────────────────────────────
# Step 3: Check Node.js
# ──────────────────────────────────────────────────────────
echo "[3/8] Checking Node.js..."
if command -v node &> /dev/null; then
    echo "  OK: Node.js $(node --version)"
else
    echo "  WARN: Node.js not found. Required for frontend."
fi

# ──────────────────────────────────────────────────────────
# Step 4: Create Python virtual environment
# ──────────────────────────────────────────────────────────
echo "[4/8] Setting up Python virtual environment..."
if [ -d "venv" ]; then
    echo "  SKIP: venv already exists"
else
    $PYTHON -m venv venv
    echo "  OK: Virtual environment created"
fi

# ──────────────────────────────────────────────────────────
# Step 5: Install Python dependencies
# ──────────────────────────────────────────────────────────
echo "[5/8] Installing Python dependencies..."
./venv/bin/pip install -r backend/requirements.txt --quiet
echo "  OK: Dependencies installed"

# ──────────────────────────────────────────────────────────
# Step 6: Download ML models
# ──────────────────────────────────────────────────────────
echo "[6/8] Downloading ML models (this may take 5-10 minutes)..."
./venv/bin/python backend/scripts/download_models.py
echo "  OK: All models downloaded"

# ──────────────────────────────────────────────────────────
# Step 7: Setup backend environment
# ──────────────────────────────────────────────────────────
echo "[7/8] Setting up backend environment..."

mkdir -p backend/uploads/raw_documents
mkdir -p backend/uploads/extracted_text
mkdir -p backend/data/vector_store

if [ ! -f "backend/.env" ] && [ -f "backend/.env.example" ]; then
    cp backend/.env.example backend/.env
    echo "  Created .env from template — edit with your DB credentials"
else
    echo "  OK: .env exists"
fi

# ──────────────────────────────────────────────────────────
# Step 8: Install frontend dependencies
# ──────────────────────────────────────────────────────────
echo "[8/8] Installing frontend dependencies..."
if [ -f "frontend/package.json" ]; then
    cd frontend && npm install --silent && cd ..
    echo "  OK: Frontend dependencies installed"
else
    echo "  SKIP: No frontend/package.json found"
fi

# ──────────────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo " Setup Complete!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Edit backend/.env with your PostgreSQL credentials"
echo "  2. Create database:  createdb policy_db"
echo "  3. Run migrations:   cd backend && ../venv/bin/alembic upgrade head"
echo "  4. Start backend:    cd backend && ../venv/bin/uvicorn app.main:app --reload"
echo "  5. Start frontend:   cd frontend && npm run dev"
echo ""
echo "API:       http://localhost:8000"
echo "API Docs:  http://localhost:8000/docs"
echo "Frontend:  http://localhost:5173"
echo ""
