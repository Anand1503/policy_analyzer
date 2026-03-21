"""
Compliance Engine — Pure computational module for regulatory gap detection.
Module 5: Maps classified clauses to regulatory articles, detects gaps, scores coverage.

Design:
  - Purely computational (no DB access)
  - Deterministic (no ML randomness)
  - Accepts pre-fetched data
  - Returns structured compliance report

Scoring Formula:
  compliance_score = (Σ article_weight × article_satisfaction) / (Σ article_weight) × 100

  Where article_satisfaction:
    1.0  = FULLY_SATISFIED   (label match + keyword match)
    0.5  = PARTIALLY_SATISFIED (label match OR keyword match, but not both)
    0.0  = MISSING            (neither match)
"""

import logging
from typing import Dict, List, Any, Optional

from app.core.regulatory_frameworks import get_framework, get_available_frameworks

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════

class ArticleFinding:
    """Result of evaluating a single regulatory article."""

    def __init__(
        self,
        article_id: str,
        title: str,
        requirement: str,
        status: str,  # "satisfied", "partial", "missing"
        label_coverage: float,
        keyword_coverage: float,
        entity_coverage: float,
        supporting_clauses: List[Dict[str, Any]],
        explanation: str,
        importance_weight: float = 1.0,
    ):
        self.article_id = article_id
        self.title = title
        self.requirement = requirement
        self.status = status
        self.label_coverage = label_coverage
        self.keyword_coverage = keyword_coverage
        self.entity_coverage = entity_coverage
        self.supporting_clauses = supporting_clauses
        self.explanation = explanation
        self.importance_weight = importance_weight

    def to_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "requirement": self.requirement,
            "status": self.status,
            "label_coverage": round(self.label_coverage, 3),
            "keyword_coverage": round(self.keyword_coverage, 3),
            "entity_coverage": round(self.entity_coverage, 3),
            "supporting_clauses": self.supporting_clauses,
            "explanation": self.explanation,
            "importance_weight": self.importance_weight,
        }


# ═══════════════════════════════════════════════════════════════
# Core Engine
# ═══════════════════════════════════════════════════════════════

def evaluate_compliance(
    framework_name: str,
    clauses: List[Dict[str, Any]],
    classifications: List[Dict[str, Any]],
    risk_scores: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a document against a regulatory framework.

    Args:
        framework_name: "GDPR", "CCPA", or custom framework name
        clauses: List of clause dicts [{clause_id, clause_index, clause_text, ...}]
        classifications: List of classification dicts [{clause_id, label, confidence_score}]
        risk_scores: List of risk dicts [{clause_id, risk_score, risk_level}]
        entities: List of entity dicts [{clause_id, entities: [...]}]
        custom_weights: Optional override for article importance weights

    Returns:
        Structured compliance report dict
    """
    framework = get_framework(framework_name)

    # ── Pre-index data by clause ID for O(1) lookup ──────────
    clause_map = {str(c["clause_id"]): c for c in clauses}
    label_index = _build_label_index(classifications)
    entity_index = _build_entity_index(entities)
    risk_index = {str(r["clause_id"]): r for r in risk_scores}

    # Collect all clause texts (lowered) for keyword matching
    all_texts = [(str(c["clause_id"]), c.get("clause_text", "").lower()) for c in clauses]

    # ── Evaluate each article ────────────────────────────────
    findings: List[ArticleFinding] = []

    for article_id, article in framework.items():
        weight = custom_weights.get(article_id, article.get("importance_weight", 1.0)) \
            if custom_weights else article.get("importance_weight", 1.0)

        finding = _evaluate_article(
            article_id=article_id,
            article=article,
            label_index=label_index,
            entity_index=entity_index,
            risk_index=risk_index,
            clause_map=clause_map,
            all_texts=all_texts,
            importance_weight=weight,
        )
        findings.append(finding)

    # ── Compute scores ───────────────────────────────────────
    satisfied = [f for f in findings if f.status == "satisfied"]
    partial = [f for f in findings if f.status == "partial"]
    missing = [f for f in findings if f.status == "missing"]

    total_weight = sum(f.importance_weight for f in findings) or 1.0
    weighted_score = sum(
        f.importance_weight * (
            1.0 if f.status == "satisfied" else
            0.5 if f.status == "partial" else
            0.0
        )
        for f in findings
    )
    compliance_score = round((weighted_score / total_weight) * 100, 1)
    coverage_percentage = round(
        ((len(satisfied) + 0.5 * len(partial)) / max(len(findings), 1)) * 100, 1
    )

    return {
        "framework": framework_name.upper(),
        "compliance_score": compliance_score,
        "coverage_percentage": coverage_percentage,
        "total_articles": len(findings),
        "satisfied_count": len(satisfied),
        "partial_count": len(partial),
        "missing_count": len(missing),
        "missing_requirements": [f.to_dict() for f in missing],
        "partial_requirements": [f.to_dict() for f in partial],
        "fully_satisfied": [f.to_dict() for f in satisfied],
    }


# ═══════════════════════════════════════════════════════════════
# Article Evaluation
# ═══════════════════════════════════════════════════════════════

def _evaluate_article(
    article_id: str,
    article: Dict[str, Any],
    label_index: Dict[str, List[Dict]],
    entity_index: Dict[str, List[str]],
    risk_index: Dict[str, Dict],
    clause_map: Dict[str, Dict],
    all_texts: List[tuple],
    importance_weight: float,
) -> ArticleFinding:
    """Evaluate a single regulatory article against clause data."""

    expected_labels = article.get("expected_labels", [])
    keywords = article.get("keywords", [])
    required_entities = article.get("required_entities", [])

    # ── 1. Label coverage ────────────────────────────────────
    # How many expected labels have at least one classified clause?
    matched_labels = set()
    label_clauses = {}  # label → clause_ids

    for label in expected_labels:
        if label in label_index:
            matched_labels.add(label)
            label_clauses[label] = [c["clause_id"] for c in label_index[label]]

    label_coverage = len(matched_labels) / max(len(expected_labels), 1)

    # ── 2. Keyword coverage ──────────────────────────────────
    # How many keywords appear in ANY clause text?
    matched_keywords = set()
    keyword_clause_ids = set()

    for keyword in keywords:
        kw_lower = keyword.lower()
        for clause_id, text in all_texts:
            if kw_lower in text:
                matched_keywords.add(keyword)
                keyword_clause_ids.add(clause_id)

    keyword_coverage = len(matched_keywords) / max(len(keywords), 1)

    # ── 3. Entity coverage ───────────────────────────────────
    # Were the required entity types found?
    matched_entities = set()
    all_entity_types = set()
    for ent_list in entity_index.values():
        all_entity_types.update(ent_list)

    for ent_type in required_entities:
        if ent_type in all_entity_types:
            matched_entities.add(ent_type)

    entity_coverage = len(matched_entities) / max(len(required_entities), 1) \
        if required_entities else 1.0  # No entity requirement = automatic pass

    # ── 4. Collect supporting clauses ────────────────────────
    supporting_clause_ids = set()
    for label, cids in label_clauses.items():
        supporting_clause_ids.update(cids)
    supporting_clause_ids.update(keyword_clause_ids)

    supporting_clauses = []
    for cid in supporting_clause_ids:
        cid_str = str(cid)
        clause = clause_map.get(cid_str, {})
        risk = risk_index.get(cid_str, {})
        supporting_clauses.append({
            "clause_id": cid_str,
            "clause_index": clause.get("clause_index", -1),
            "risk_score": risk.get("risk_score", 0.0),
            "risk_level": risk.get("risk_level", "unknown"),
            "matched_labels": [
                l for l in expected_labels
                if cid_str in [str(x) for x in label_clauses.get(l, [])]
            ],
        })

    # ── 5. Determine status ──────────────────────────────────
    # Combined score: 50% labels + 35% keywords + 15% entities
    combined_score = (label_coverage * 0.50) + (keyword_coverage * 0.35) + (entity_coverage * 0.15)

    if combined_score >= 0.65:
        status = "satisfied"
    elif combined_score >= 0.30:
        status = "partial"
    else:
        status = "missing"

    # ── 6. Generate explanation ──────────────────────────────
    explanation = _generate_article_explanation(
        article_id=article_id,
        article=article,
        status=status,
        label_coverage=label_coverage,
        keyword_coverage=keyword_coverage,
        entity_coverage=entity_coverage,
        matched_labels=matched_labels,
        expected_labels=expected_labels,
        matched_keywords=matched_keywords,
        keywords=keywords,
        supporting_count=len(supporting_clauses),
    )

    return ArticleFinding(
        article_id=article_id,
        title=article.get("title", ""),
        requirement=article.get("requirement", ""),
        status=status,
        label_coverage=label_coverage,
        keyword_coverage=keyword_coverage,
        entity_coverage=entity_coverage,
        supporting_clauses=supporting_clauses,
        explanation=explanation,
        importance_weight=importance_weight,
    )


# ═══════════════════════════════════════════════════════════════
# Explanation Generator
# ═══════════════════════════════════════════════════════════════

def _generate_article_explanation(
    article_id: str,
    article: Dict[str, Any],
    status: str,
    label_coverage: float,
    keyword_coverage: float,
    entity_coverage: float,
    matched_labels: set,
    expected_labels: List[str],
    matched_keywords: set,
    keywords: List[str],
    supporting_count: int,
) -> str:
    """Generate a human-readable explanation for an article finding."""

    title = article.get("title", article_id)
    status_desc = {
        "satisfied": "appears fully satisfied",
        "partial": "appears partially satisfied",
        "missing": "does not appear to be addressed",
    }

    parts = [f"{article_id} ({title}) {status_desc.get(status, status)}."]

    if status == "satisfied":
        parts.append(
            f"Found {supporting_count} supporting clause(s) with matching labels "
            f"({', '.join(sorted(matched_labels))}) and keywords."
        )
    elif status == "partial":
        missing_labels = set(expected_labels) - matched_labels
        missing_kws = set(keywords) - matched_keywords
        if missing_labels:
            parts.append(
                f"Missing classification labels: {', '.join(sorted(missing_labels))}."
            )
        if missing_kws and len(missing_kws) <= 5:
            parts.append(
                f"Missing keywords: {', '.join(sorted(missing_kws))}."
            )
        elif missing_kws:
            parts.append(f"Missing {len(missing_kws)} expected keywords.")
        parts.append(f"Label coverage: {label_coverage:.0%}, Keyword coverage: {keyword_coverage:.0%}.")
    else:  # missing
        parts.append(
            f"No clauses match expected labels ({', '.join(expected_labels)}) "
            f"or keywords. This requirement may need explicit policy language."
        )

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
# Indexing Helpers
# ═══════════════════════════════════════════════════════════════

def _build_label_index(classifications: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """Index classifications by label for O(1) lookup."""
    index: Dict[str, List[Dict]] = {}
    for c in classifications:
        label = c.get("label", "")
        if label not in index:
            index[label] = []
        index[label].append(c)
    return index


def _build_entity_index(entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Index entity types per clause for O(1) lookup."""
    index: Dict[str, List[str]] = {}
    for e in entities:
        clause_id = str(e.get("clause_id", ""))
        ent_list = e.get("entities", [])
        types = list(set(
            ent.get("type", "") if isinstance(ent, dict) else str(ent)
            for ent in ent_list
        ))
        index[clause_id] = types
    return index
