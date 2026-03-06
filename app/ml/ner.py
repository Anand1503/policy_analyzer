"""
Named Entity Recognition (NER) for legal/privacy policy documents.

Supports two modes:
  1. extract_entities(full_text)       — document-level entity extraction
  2. extract_clause_entities(clause)   — per-clause entity extraction (Module 1)

Entity output format:
  {"text": "GDPR", "label": "REGULATION", "start": 42, "end": 46}
"""

import re
import logging
from typing import List, Dict

from app.core.exceptions import NERError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Pattern Definitions
# ═══════════════════════════════════════════════════════════════

PATTERNS: List[tuple] = [
    # ── Regulations ──────────────────────────────────────────
    (r"\bGDPR\b", "REGULATION"),
    (r"\bCCPA\b", "REGULATION"),
    (r"\bHIPAA\b", "REGULATION"),
    (r"\bCOPPA\b", "REGULATION"),
    (r"\bPIPEDA\b", "REGULATION"),
    (r"\bCalOPPA\b", "REGULATION"),
    (r"\bFERPA\b", "REGULATION"),
    (r"\bSOC\s*2\b", "REGULATION"),
    (r"\bISO\s*27001\b", "REGULATION"),
    (r"\bPCI[\s-]DSS\b", "REGULATION"),
    (r"\bDPA\s*2018\b", "REGULATION"),

    # ── Data Types ───────────────────────────────────────────
    (r"\bemail\s*address(?:es)?\b", "DATA_TYPE"),
    (r"\bIP\s*address(?:es)?\b", "DATA_TYPE"),
    (r"\bcookies?\b", "DATA_TYPE"),
    (r"\blocation\s*data\b", "DATA_TYPE"),
    (r"\bbiometric(?:\s*data)?\b", "DATA_TYPE"),
    (r"\bfinancial\s*(?:data|information)\b", "DATA_TYPE"),
    (r"\bhealth\s*(?:data|information)\b", "DATA_TYPE"),
    (r"\bdevice\s*(?:id|identifier)s?\b", "DATA_TYPE"),
    (r"\bSSN\b|social\s*security\s*(?:number)?", "DATA_TYPE"),
    (r"\bcredit\s*card\b", "DATA_TYPE"),
    (r"\bbrowsing\s*(?:history|data)\b", "DATA_TYPE"),
    (r"\bsearch\s*(?:history|queries)\b", "DATA_TYPE"),
    (r"\bpersonal\s*(?:data|information)\b", "DATA_TYPE"),
    (r"\bsensitive\s*(?:data|information)\b", "DATA_TYPE"),

    # ── Organizations ────────────────────────────────────────
    (r"\b(?:Google|Facebook|Meta|Apple|Amazon|Microsoft|Twitter|LinkedIn)\b", "ORGANIZATION"),
    (r"\b(?:Inc|LLC|Ltd|Corp|GmbH|S\.A\.|Pty|PLC)\b", "ORGANIZATION"),

    # ── Jurisdictions ────────────────────────────────────────
    (r"\b(?:California|European Union|EU|United States|United Kingdom|Canada|Australia|India|Germany|France)\b", "JURISDICTION"),
    (r"\b(?:Delaware|New York|Texas|Nevada|Virginia|Colorado|Connecticut)\b", "JURISDICTION"),

    # ── Legal References ─────────────────────────────────────
    (r"\bSection\s+\d+[\.\d]*\b", "LEGAL_REFERENCE"),
    (r"\bArticle\s+\d+[\.\d]*\b", "LEGAL_REFERENCE"),
    (r"\b(?:Title|Chapter|Part)\s+(?:I{1,3}V?|V?I{0,3}|\d+)\b", "LEGAL_REFERENCE"),
    (r"\b\d+\s*(?:U\.?S\.?C\.?|C\.?F\.?R\.?)\s*§?\s*\d+\b", "LEGAL_REFERENCE"),

    # ── Time Periods ─────────────────────────────────────────
    (r"\b\d+\s*(?:days?|weeks?|months?|years?)\b", "TIME_PERIOD"),
    (r"\b(?:indefinite|perpetual|unlimited)\s*(?:retention|period|storage|basis)\b", "TIME_PERIOD"),
    (r"\b(?:annual|quarterly|monthly|weekly|daily)\b", "TIME_PERIOD"),

    # ── Monetary Values ──────────────────────────────────────
    (r"\$\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|USD|EUR))?", "MONETARY_VALUE"),
    (r"\b€\s*[\d,]+(?:\.\d{2})?\b", "MONETARY_VALUE"),
    (r"\b\d[\d,]*\s*(?:dollars?|euros?|pounds?|USD|EUR|GBP)\b", "MONETARY_VALUE"),

    # ── Dates ────────────────────────────────────────────────
    (r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", "DATE"),
    (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "DATE"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "DATE"),
]

# Compile patterns once for performance
_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), label) for p, label in PATTERNS]


# ═══════════════════════════════════════════════════════════════
# Per-Clause Entity Extraction (Module 1 primary interface)
# ═══════════════════════════════════════════════════════════════

def extract_clause_entities(clause_text: str) -> List[Dict]:
    """
    Extract named entities from a single clause.
    
    Returns:
        [{"text": "GDPR", "label": "REGULATION", "start": 42, "end": 46}, ...]
    """
    try:
        entities = []
        seen_spans = set()

        for compiled, label in _COMPILED_PATTERNS:
            for match in compiled.finditer(clause_text):
                span = (match.start(), match.end())
                # Avoid overlapping entities
                if not _overlaps(span, seen_spans):
                    seen_spans.add(span)
                    entities.append({
                        "text": match.group().strip(),
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                    })

        # Sort by position
        entities.sort(key=lambda e: e["start"])
        return entities

    except Exception as e:
        raise NERError(
            f"NER failed on clause: {str(e)}",
            step="extract_clause_entities",
        )


def _overlaps(span: tuple, existing: set) -> bool:
    """Check if a span overlaps with any existing span."""
    s, e = span
    for es, ee in existing:
        if s < ee and e > es:
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# Document-Level Entity Extraction (legacy + API endpoint)
# ═══════════════════════════════════════════════════════════════

def extract_entities(text: str) -> Dict[str, List[Dict]]:
    """
    Extract named entities from full document text.
    Returns categorized entities grouped by label type.
    """
    try:
        raw = extract_clause_entities(text)

        grouped = {
            "regulations": [],
            "data_types": [],
            "organizations": [],
            "jurisdictions": [],
            "legal_references": [],
            "time_periods": [],
            "monetary_values": [],
            "dates": [],
        }

        label_map = {
            "REGULATION": "regulations",
            "DATA_TYPE": "data_types",
            "ORGANIZATION": "organizations",
            "JURISDICTION": "jurisdictions",
            "LEGAL_REFERENCE": "legal_references",
            "TIME_PERIOD": "time_periods",
            "MONETARY_VALUE": "monetary_values",
            "DATE": "dates",
        }

        # Deduplicate by (text, label)
        seen = set()
        for entity in raw:
            key = (entity["text"].lower(), entity["label"])
            if key not in seen:
                seen.add(key)
                group = label_map.get(entity["label"], "other")
                if group in grouped:
                    grouped[group].append(entity)

        total = sum(len(v) for v in grouped.values())
        logger.info(f"NER document-level: {total} unique entities extracted")
        return grouped

    except NERError:
        raise
    except Exception as e:
        raise NERError(f"Document-level NER failed: {str(e)}", step="extract_entities")
