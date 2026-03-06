"""
Download all ML models to the local models/ directory.
Run this once after cloning or on a new deployment machine.

Usage: python scripts/download_models.py
"""

import os
import sys

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.environ["HF_HOME"] = BASE_DIR
os.environ["TRANSFORMERS_CACHE"] = BASE_DIR

def download_legal_bert():
    """Download Legal-BERT for classification."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    save_path = os.path.join(BASE_DIR, "legal-bert")
    os.makedirs(save_path, exist_ok=True)
    print(f"Downloading Legal-BERT to {save_path} ...")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    tokenizer.save_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased", num_labels=10, ignore_mismatched_sizes=True,
    )
    model.save_pretrained(save_path)
    print("  [OK] Legal-BERT saved")


def download_t5():
    """Download T5-base for summarization and RAG generation."""
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    save_path = os.path.join(BASE_DIR, "t5-base")
    os.makedirs(save_path, exist_ok=True)
    print(f"Downloading T5-base to {save_path} ...")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer.save_pretrained(save_path)
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.save_pretrained(save_path)
    print("  [OK] T5-base saved")


def download_embeddings():
    """Download sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer
    save_path = os.path.join(BASE_DIR, "all-MiniLM-L6-v2")
    os.makedirs(save_path, exist_ok=True)
    print(f"Downloading all-MiniLM-L6-v2 to {save_path} ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.save(save_path)
    print("  [OK] all-MiniLM-L6-v2 saved")


def download_spacy():
    """Download SpaCy English model."""
    import subprocess
    print("Downloading SpaCy en_core_web_sm ...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    print("  [OK] SpaCy en_core_web_sm installed")


if __name__ == "__main__":
    print("=" * 50)
    print("Downloading all ML models for offline use")
    print("=" * 50)
    download_legal_bert()
    download_t5()
    download_embeddings()
    download_spacy()
    print()
    print("=" * 50)
    print("All models downloaded. Total ~1.36 GB")
    print("=" * 50)
