# ═══════════════════════════════════════════════════════════
# Intelligent Policy Analyzer — Windows Setup
# ═══════════════════════════════════════════════════════════
#
# Usage: powershell -ExecutionPolicy Bypass -File setup.ps1
#

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Intelligent Policy Analyzer — Setup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ──────────────────────────────────────────────────────────
# Step 1: Check Python
# ──────────────────────────────────────────────────────────
Write-Host "[1/8] Checking Python installation..." -ForegroundColor Yellow
try {
    $pyVersion = python --version 2>&1
    Write-Host "  OK: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "  FAIL: Python not found. Install Python 3.11+" -ForegroundColor Red
    exit 1
}

# ──────────────────────────────────────────────────────────
# Step 2: Check PostgreSQL
# ──────────────────────────────────────────────────────────
Write-Host "[2/8] Checking PostgreSQL..." -ForegroundColor Yellow
try {
    $pgVersion = psql --version 2>&1
    Write-Host "  OK: $pgVersion" -ForegroundColor Green
} catch {
    Write-Host "  WARN: PostgreSQL not found. Install PostgreSQL 15+ or use Docker." -ForegroundColor DarkYellow
}

# ──────────────────────────────────────────────────────────
# Step 3: Check Node.js (for frontend)
# ──────────────────────────────────────────────────────────
Write-Host "[3/8] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  OK: Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  WARN: Node.js not found. Required for frontend." -ForegroundColor DarkYellow
}

# ──────────────────────────────────────────────────────────
# Step 4: Create Python virtual environment
# ──────────────────────────────────────────────────────────
Write-Host "[4/8] Setting up Python virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  SKIP: venv already exists" -ForegroundColor DarkYellow
} else {
    python -m venv venv
    Write-Host "  OK: Virtual environment created" -ForegroundColor Green
}

# ──────────────────────────────────────────────────────────
# Step 5: Install Python dependencies
# ──────────────────────────────────────────────────────────
Write-Host "[5/8] Installing Python dependencies..." -ForegroundColor Yellow
& .\venv\Scripts\pip.exe install -r backend\requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK: Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  FAIL: pip install failed" -ForegroundColor Red
    exit 1
}

# ──────────────────────────────────────────────────────────
# Step 6: Download ML models
# ──────────────────────────────────────────────────────────
Write-Host "[6/8] Downloading ML models (this may take 5-10 minutes)..." -ForegroundColor Yellow
& .\venv\Scripts\python.exe backend\scripts\download_models.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK: All models downloaded" -ForegroundColor Green
} else {
    Write-Host "  WARN: Model download had issues. Run manually: python backend\scripts\download_models.py" -ForegroundColor DarkYellow
}

# ──────────────────────────────────────────────────────────
# Step 7: Setup backend environment
# ──────────────────────────────────────────────────────────
Write-Host "[7/8] Setting up backend environment..." -ForegroundColor Yellow

# Create directories
$dirs = @(
    "backend\uploads\raw_documents",
    "backend\uploads\extracted_text",
    "backend\data\vector_store"
)
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Create .env from example if needed
if (!(Test-Path "backend\.env") -and (Test-Path "backend\.env.example")) {
    Copy-Item "backend\.env.example" "backend\.env"
    Write-Host "  Created .env from template — edit with your DB credentials" -ForegroundColor DarkYellow
} else {
    Write-Host "  OK: .env exists" -ForegroundColor Green
}

# ──────────────────────────────────────────────────────────
# Step 8: Install frontend dependencies
# ──────────────────────────────────────────────────────────
Write-Host "[8/8] Installing frontend dependencies..." -ForegroundColor Yellow
if (Test-Path "frontend\package.json") {
    Push-Location frontend
    npm install --silent 2>&1 | Out-Null
    Pop-Location
    Write-Host "  OK: Frontend dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  SKIP: No frontend/package.json found" -ForegroundColor DarkYellow
}

# ──────────────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Green
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit backend\.env with your PostgreSQL credentials" -ForegroundColor White
Write-Host "  2. Create database:  createdb policy_db" -ForegroundColor White
Write-Host "  3. Run migrations:   cd backend && ..\venv\Scripts\alembic.exe upgrade head" -ForegroundColor White
Write-Host "  4. Start backend:    ..\venv\Scripts\python.exe -m uvicorn app.main:app --reload" -ForegroundColor White
Write-Host "  5. Start frontend:   cd frontend && npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "API:       http://localhost:8000" -ForegroundColor Yellow
Write-Host "API Docs:  http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "Frontend:  http://localhost:5173" -ForegroundColor Yellow
Write-Host ""
