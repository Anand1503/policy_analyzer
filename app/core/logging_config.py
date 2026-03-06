"""
Logging configuration — centralized structured logging.
Maps to the 'Logging & Monitoring' component in the architecture's Backend Layer.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import settings


def setup_logging():
    """
    Configure application-wide logging with:
    - Console output (colored)  
    - File rotation (10MB max, 5 backups)
    - Structured format
    """
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)

    # Formatter
    fmt = "%(asctime)s [%(levelname)-8s] %(name)-25s │ %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # Root logger
    root = logging.getLogger()
    root.setLevel(log_level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logging.info("Logging initialized")


def get_logger(name: str) -> logging.Logger:
    """Create a named logger for a module."""
    return logging.getLogger(name)
