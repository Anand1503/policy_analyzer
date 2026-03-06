"""
Singleton model loader — loads Legal-BERT, T5, SpaCy, and embedding models once.
Thread-safe lazy-loading with GPU fallback, checksum tracking, and memory logging.

Models:
  - Classifier: nlpaueb/legal-bert-base-uncased (AutoModelForSequenceClassification)
  - Summarizer: t5-base (T5ForConditionalGeneration)
  - Embeddings: all-MiniLM-L6-v2 (via rag.py / sentence-transformers)
  - SpaCy: en_core_web_sm (tokenization + sentencizer)
"""

import hashlib
import logging
import os
import threading
import torch
from typing import Optional, Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from app.core.config import settings
from app.core.exceptions import PolicyAnalyzerError

logger = logging.getLogger(__name__)

_models = {}
_model_versions = {}
_model_checksums = {}
_lock = threading.Lock()


class ModelLoadError(PolicyAnalyzerError):
    """Raised when a model fails to load."""
    pass


# ═══════════════════════════════════════════════════════════════
# Memory Logging
# ═══════════════════════════════════════════════════════════════

def _log_gpu_memory(label: str):
    """Log GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        logger.info(f"[gpu_memory] {label}: allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")


def _log_cpu_memory(label: str):
    """Log CPU/process memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        logger.info(f"[cpu_memory] {label}: rss={mem.rss / (1024**2):.1f}MB, vms={mem.vms / (1024**2):.1f}MB")
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════
# Model Checksum
# ═══════════════════════════════════════════════════════════════

def _compute_model_checksum(model_name: str) -> dict:
    """Compute model version metadata for reproducibility."""
    checksum_info = {"model_name": model_name, "revision": "unknown", "sha256_partial": "unknown"}
    try:
        cache_dir = os.environ.get(
            "TRANSFORMERS_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        )
        model_safe_name = model_name.replace("/", "--")
        model_dir = os.path.join(cache_dir, f"models--{model_safe_name}")
        if os.path.isdir(model_dir):
            for root, dirs, files in os.walk(model_dir):
                if "config.json" in files:
                    config_path = os.path.join(root, "config.json")
                    with open(config_path, "rb") as f:
                        checksum_info["sha256_partial"] = hashlib.sha256(f.read()).hexdigest()[:16]
                    break
            refs_path = os.path.join(model_dir, "refs", "main")
            if os.path.isfile(refs_path):
                with open(refs_path, "r") as f:
                    checksum_info["revision"] = f.read().strip()[:12]
    except Exception as e:
        logger.debug(f"[model_loader] Checksum computation skipped: {e}")
    return checksum_info


def _detect_device() -> torch.device:
    """Detect best available device."""
    if torch.cuda.is_available():
        logger.info(f"[model_loader] GPU detected: {torch.cuda.get_device_name(0)}")
        _log_gpu_memory("before_model_load")
        return torch.device("cuda:0")
    logger.info("[model_loader] No GPU detected, using CPU")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════
# Legal-BERT Classifier (Sequence Classification)
# ═══════════════════════════════════════════════════════════════

def get_classifier() -> Tuple:
    """
    Legal-BERT sequence classification model + tokenizer.
    Returns: (model, tokenizer) tuple.
    Thread-safe lazy-loaded singleton.
    """
    if "classifier" not in _models:
        with _lock:
            if "classifier" not in _models:
                model_name = settings.CLASSIFIER_MODEL
                num_labels = settings.LEGAL_BERT_NUM_LABELS
                device = _detect_device()
                logger.info(f"[model_loader] Loading Legal-BERT classifier: {model_name} ({num_labels} labels, device={device})")
                _log_cpu_memory("before_classifier_load")

                try:
                    from app.core.metrics import track_model_load
                except ImportError:
                    track_model_load = lambda *a: None

                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=num_labels,
                        ignore_mismatched_sizes=True,  # Base model → classification head
                    )
                    model.to(device)
                    model.eval()

                    _models["classifier"] = (model, tokenizer)
                    _models["classifier_device"] = device
                    _model_versions["classifier"] = model_name
                    _model_checksums["classifier"] = _compute_model_checksum(model_name)
                    track_model_load(model_name, True)
                    _log_gpu_memory("after_classifier_load")
                    _log_cpu_memory("after_classifier_load")
                    logger.info(f"[model_loader] ✓ Legal-BERT classifier loaded: {model_name}")
                except Exception as e:
                    logger.error(f"[model_loader] Classifier load failed: {e}")
                    track_model_load(model_name, False)
                    # GPU fallback
                    if device.type == "cuda":
                        logger.warning("[model_loader] GPU fallback → retrying on CPU...")
                        torch.cuda.empty_cache()
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(model_name)
                            model = AutoModelForSequenceClassification.from_pretrained(
                                model_name, num_labels=num_labels, ignore_mismatched_sizes=True,
                            )
                            model.eval()
                            _models["classifier"] = (model, tokenizer)
                            _models["classifier_device"] = torch.device("cpu")
                            _model_versions["classifier"] = model_name
                            _model_checksums["classifier"] = _compute_model_checksum(model_name)
                            track_model_load(model_name, True)
                            logger.info(f"[model_loader] ✓ Classifier loaded on CPU: {model_name}")
                        except Exception as e2:
                            track_model_load(model_name, False)
                            raise ModelLoadError(f"Failed to load classifier on CPU: {e2}", step="model_load")
                    else:
                        raise ModelLoadError(f"Failed to load classifier: {e}", step="model_load")

    return _models["classifier"]


def get_classifier_device() -> torch.device:
    """Return the device the classifier is loaded on."""
    return _models.get("classifier_device", torch.device("cpu"))


# ═══════════════════════════════════════════════════════════════
# T5 Summarizer (Seq2Seq)
# ═══════════════════════════════════════════════════════════════

def get_summarizer() -> Tuple:
    """
    T5 summarization model + tokenizer.
    Returns: (model, tokenizer) tuple.
    Thread-safe lazy-loaded singleton.
    """
    if "summarizer" not in _models:
        with _lock:
            if "summarizer" not in _models:
                model_name = settings.SUMMARIZER_MODEL
                device = _detect_device()
                logger.info(f"[model_loader] Loading T5 summarizer: {model_name} (device={device})")

                try:
                    from app.core.metrics import track_model_load
                except ImportError:
                    track_model_load = lambda *a: None

                try:
                    tokenizer = T5Tokenizer.from_pretrained(model_name)
                    model = T5ForConditionalGeneration.from_pretrained(model_name)
                    model.to(device)
                    model.eval()

                    _models["summarizer"] = (model, tokenizer)
                    _models["summarizer_device"] = device
                    _model_versions["summarizer"] = model_name
                    _model_checksums["summarizer"] = _compute_model_checksum(model_name)
                    track_model_load(model_name, True)
                    _log_gpu_memory("after_summarizer_load")
                    _log_cpu_memory("after_summarizer_load")
                    logger.info(f"[model_loader] ✓ T5 summarizer loaded: {model_name}")
                except Exception as e:
                    logger.error(f"[model_loader] Summarizer load failed: {e}")
                    track_model_load(model_name, False)
                    if device.type == "cuda":
                        logger.warning("[model_loader] GPU fallback → retrying on CPU...")
                        torch.cuda.empty_cache()
                        try:
                            tokenizer = T5Tokenizer.from_pretrained(model_name)
                            model = T5ForConditionalGeneration.from_pretrained(model_name)
                            model.eval()
                            _models["summarizer"] = (model, tokenizer)
                            _models["summarizer_device"] = torch.device("cpu")
                            _model_versions["summarizer"] = model_name
                            _model_checksums["summarizer"] = _compute_model_checksum(model_name)
                            track_model_load(model_name, True)
                            logger.info(f"[model_loader] ✓ Summarizer loaded on CPU: {model_name}")
                        except Exception as e2:
                            track_model_load(model_name, False)
                            raise ModelLoadError(f"Failed to load summarizer on CPU: {e2}", step="model_load")
                    else:
                        raise ModelLoadError(f"Failed to load summarizer: {e}", step="model_load")

    return _models["summarizer"]


def get_summarizer_device() -> torch.device:
    """Return the device the summarizer is loaded on."""
    return _models.get("summarizer_device", torch.device("cpu"))


# ═══════════════════════════════════════════════════════════════
# SpaCy NLP Model
# ═══════════════════════════════════════════════════════════════

def get_spacy_nlp():
    """Lazy-loaded singleton SpaCy NLP model for tokenization and sentencization."""
    if "spacy_nlp" not in _models:
        with _lock:
            if "spacy_nlp" not in _models:
                import spacy
                model_name = settings.SPACY_MODEL
                logger.info(f"[model_loader] Loading SpaCy model: {model_name}")
                _models["spacy_nlp"] = spacy.load(model_name)
                _model_versions["spacy_nlp"] = model_name
                logger.info(f"[model_loader] ✓ SpaCy loaded: {model_name}")
    return _models["spacy_nlp"]


# ═══════════════════════════════════════════════════════════════
# Preload / Cleanup / Status
# ═══════════════════════════════════════════════════════════════

def preload_models():
    """Pre-load all models at startup."""
    logger.info("[model_loader] Pre-loading models...")
    _log_cpu_memory("before_preload")

    try:
        get_classifier()
        logger.info("[model_loader] ✓ Legal-BERT classifier pre-loaded")
    except Exception as e:
        logger.error(f"[model_loader] Classifier pre-load failed: {e}")

    try:
        get_summarizer()
        logger.info("[model_loader] ✓ T5 summarizer pre-loaded")
    except Exception as e:
        logger.error(f"[model_loader] Summarizer pre-load failed: {e}")

    try:
        get_spacy_nlp()
        logger.info("[model_loader] ✓ SpaCy pre-loaded")
    except Exception as e:
        logger.error(f"[model_loader] SpaCy pre-load failed: {e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log_gpu_memory("after_preload")

    _log_cpu_memory("after_preload")
    logger.info("[model_loader] Pre-load complete")


def cleanup_models():
    """Release models and GPU memory on shutdown."""
    global _models, _model_versions, _model_checksums
    with _lock:
        _models.clear()
        _model_versions.clear()
        _model_checksums.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[model_loader] Models released, GPU cache cleared")


def get_model_version(model_key: str) -> str:
    return _model_versions.get(model_key, "unknown")


def get_model_checksum(model_key: str) -> dict:
    """Get the checksum info for a loaded model."""
    return _model_checksums.get(model_key, {
        "model_name": "unknown", "revision": "unknown", "sha256_partial": "unknown",
    })


def is_model_loaded(model_key: str) -> bool:
    return model_key in _models


def get_models_status() -> dict:
    """Return status of all models — used by /health/ready."""
    return {
        "classifier": is_model_loaded("classifier"),
        "summarizer": is_model_loaded("summarizer"),
        "spacy_nlp": is_model_loaded("spacy_nlp"),
        "classifier_version": get_model_version("classifier"),
        "summarizer_version": get_model_version("summarizer"),
        "classifier_checksum": get_model_checksum("classifier"),
        "summarizer_checksum": get_model_checksum("summarizer"),
    }
