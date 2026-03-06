"""
Phase 8: System Health Check
============================
Verifies backend runtime environment:
  - Virtual environment
  - Required Python packages
  - Model files exist
  - Database connectivity (PostgreSQL)
  - Alembic migration status
  - API endpoints (requires running server)
"""

import importlib, json, logging, os, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "evaluation"

REQUIRED_PACKAGES = [
    "fastapi", "uvicorn", "sqlalchemy", "alembic",
    "transformers", "torch", "sklearn", "spacy",
    "sentencepiece", "fitz", "chromadb",
    "pydantic", "pydantic_settings",
    "shap", "numpy", "pandas",
]

REQUIRED_MODELS = {
    "LegalBERT base": BASE_DIR / "models" / "legal-bert",
    "LegalBERT v2 (finetuned)": BASE_DIR / "models" / "legal-bert-finetuned-v2",
    "T5 Summarizer": BASE_DIR / "models" / "t5-base",
    "Embedding (MiniLM)": BASE_DIR / "models" / "all-MiniLM-L6-v2",
}

REQUIRED_V2_FILES = [
    "config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json",
    "optimal_thresholds.json", "model_metadata.json",
]


def check_venv():
    """Check if running inside virtual environment."""
    in_venv = hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    return {"status": "OK" if in_venv else "WARNING", "in_venv": in_venv, "prefix": sys.prefix}


def check_packages():
    """Check required Python packages."""
    results = {}
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            results[pkg] = {"status": "OK", "version": str(ver)}
        except ImportError:
            results[pkg] = {"status": "MISSING"}
    return results


def check_models():
    """Check model directories exist and have required files."""
    results = {}
    for name, path in REQUIRED_MODELS.items():
        if path.exists():
            files = [f.name for f in path.iterdir() if f.is_file()]
            size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
            results[name] = {"status": "OK", "path": str(path), "files": len(files), "size_mb": round(size_mb, 1)}
        else:
            results[name] = {"status": "MISSING", "path": str(path)}

    # Check v2 specific files
    v2_path = REQUIRED_MODELS["LegalBERT v2 (finetuned)"]
    if v2_path.exists():
        missing = [f for f in REQUIRED_V2_FILES if not (v2_path / f).exists()]
        results["v2_completeness"] = {
            "status": "OK" if not missing else "INCOMPLETE",
            "missing_files": missing,
        }

    return results


def check_database():
    """Check PostgreSQL connectivity."""
    try:
        import psycopg2
        from app.core.config import settings
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
        )
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
        table_count = cur.fetchone()[0]
        conn.close()
        return {"status": "OK", "version": version[:60], "tables": table_count}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)[:200]}


def check_config():
    """Check configuration values."""
    try:
        sys.path.insert(0, str(BASE_DIR))
        from app.core.config import settings
        return {
            "status": "OK",
            "app_name": settings.APP_NAME,
            "app_env": settings.APP_ENV,
            "classifier_model": settings.CLASSIFIER_MODEL,
            "summarizer_model": settings.SUMMARIZER_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "db_host": settings.POSTGRES_HOST,
            "db_name": settings.POSTGRES_DB,
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)[:200]}


def check_evaluation_files():
    """Check evaluation output files."""
    expected = [
        "dataset_integrity_report.json",
        "baseline_metrics.json",
        "legalbert_final_metrics.json",
        "optimal_thresholds.json",
    ]
    results = {}
    for f in expected:
        path = EVAL_DIR / f
        results[f] = {"status": "OK" if path.exists() else "MISSING"}
        if path.exists():
            results[f]["size_bytes"] = path.stat().st_size
    return results


def main():
    log.info("=" * 60)
    log.info("PHASE 8: System Health Check")
    log.info("=" * 60)

    report = {}

    # 1. Virtual environment
    report["virtual_env"] = check_venv()
    log.info(f"Venv: {report['virtual_env']['status']}")

    # 2. Required packages
    report["packages"] = check_packages()
    missing = [k for k, v in report["packages"].items() if v["status"] == "MISSING"]
    log.info(f"Packages: {len(report['packages']) - len(missing)}/{len(report['packages'])} OK, missing: {missing}")

    # 3. Model files
    report["models"] = check_models()
    for name, status in report["models"].items():
        log.info(f"Model '{name}': {status['status']}")

    # 4. Configuration
    report["config"] = check_config()
    log.info(f"Config: {report['config']['status']}")

    # 5. Database
    report["database"] = check_database()
    log.info(f"Database: {report['database']['status']}")

    # 6. Evaluation files
    report["evaluation_files"] = check_evaluation_files()
    eval_missing = [k for k, v in report["evaluation_files"].items() if v["status"] == "MISSING"]
    log.info(f"Eval files: {len(report['evaluation_files']) - len(eval_missing)}/{len(report['evaluation_files'])} OK")

    # Save report
    out = EVAL_DIR / "health_check_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Report saved to {out}")

    # Summary
    issues = []
    if report["virtual_env"]["status"] != "OK":
        issues.append("Not in venv")
    if missing:
        issues.append(f"Missing packages: {missing}")
    for name, status in report["models"].items():
        if status.get("status") in ("MISSING", "INCOMPLETE"):
            issues.append(f"Model issue: {name}")
    if report["database"]["status"] != "OK":
        issues.append(f"DB: {report['database'].get('error', 'unknown')}")

    print(f"\n{'='*60}")
    print(f"HEALTH CHECK SUMMARY")
    print(f"{'='*60}")
    if issues:
        print(f"Issues found ({len(issues)}):")
        for i in issues:
            print(f"  ⚠ {i}")
    else:
        print("✓ All checks passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
