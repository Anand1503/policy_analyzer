"""
Executive Insight Engine — Deterministic executive-level document insights.
Module 6: Generates top risks, compliance gaps, and recommended actions.

Design:
  - Purely computational (no DB access)
  - Deterministic rules (no ML randomness, no speculative legal advice)
  - Based only on analyzed data (clause risks, compliance gaps, labels)
  - Produces structured executive summary

Output:
  {
      "overall_summary": "This policy presents a HIGH risk...",
      "risk_level": "high",
      "overall_risk_score": 0.72,
      "top_risks": [...],
      "compliance_gaps": [...],
      "recommendations": [...],
      "key_statistics": {...}
  }
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Risk Narrative Templates
# ═══════════════════════════════════════════════════════════════

_RISK_NARRATIVES = {
    "critical": (
        "This policy presents a CRITICAL risk level. "
        "Multiple clauses contain terms that significantly reduce user protections "
        "and create substantial legal exposure. Immediate remediation is strongly recommended."
    ),
    "high": (
        "This policy presents a HIGH risk level. "
        "Several clauses contain concerning terms related to data handling, third-party sharing, "
        "or limited user rights that warrant careful review before acceptance."
    ),
    "medium": (
        "This policy presents a MODERATE risk level. "
        "While most provisions are standard, some clauses contain terms that may limit "
        "user protections or create ambiguity in data handling practices."
    ),
    "low": (
        "This policy presents a LOW risk level. "
        "The document appears to follow standard privacy practices with adequate user protections. "
        "No critical gaps were identified, though periodic review is still recommended."
    ),
}

# Category-specific risk descriptors for Top Risks
_CATEGORY_RISK_DESCRIPTORS: Dict[str, str] = {
    "DATA_COLLECTION": "Extensive personal data collection practices",
    "DATA_SHARING": "Broad data sharing with external parties",
    "USER_RIGHTS": "Limited or unclear user rights provisions",
    "DATA_RETENTION": "Unclear or excessive data retention periods",
    "SECURITY_MEASURES": "Insufficient security commitments or vague protections",
    "THIRD_PARTY_TRANSFER": "Uncontrolled third-party data transfers",
    "COOKIES_TRACKING": "Extensive tracking and profiling mechanisms",
    "CHILDREN_PRIVACY": "Inadequate children's data protections",
    "COMPLIANCE_REFERENCE": "Vague or unverifiable compliance claims",
    "LIABILITY_LIMITATION": "Broad liability limitations reducing user recourse",
}

# Category-specific recommendation templates
_CATEGORY_RECOMMENDATIONS: Dict[str, str] = {
    "DATA_COLLECTION": "Review data collection scope and ensure data minimization principles are applied.",
    "DATA_SHARING": "Clarify data sharing recipients and implement data processing agreements.",
    "USER_RIGHTS": "Strengthen user rights language with clear exercise mechanisms and response timelines.",
    "DATA_RETENTION": "Define specific data retention periods and implement automatic deletion procedures.",
    "SECURITY_MEASURES": "Specify technical security measures including encryption standards and access controls.",
    "THIRD_PARTY_TRANSFER": "Limit third-party transfers and require standard contractual clauses.",
    "COOKIES_TRACKING": "Implement granular cookie consent mechanisms with clear opt-out options.",
    "CHILDREN_PRIVACY": "Add explicit age verification and parental consent requirements for minors' data.",
    "COMPLIANCE_REFERENCE": "Provide specific regulatory references and maintain verifiable compliance documentation.",
    "LIABILITY_LIMITATION": "Review liability clauses to ensure they do not override statutory consumer protections.",
}


# ═══════════════════════════════════════════════════════════════
# Core Engine
# ═══════════════════════════════════════════════════════════════

def generate_executive_insights(
    overall_risk_score: float,
    risk_level: str,
    clause_risks: List[Dict[str, Any]],
    compliance_gaps: Optional[List[Dict[str, Any]]] = None,
    compliance_score: Optional[float] = None,
    total_clauses: int = 0,
) -> Dict[str, Any]:
    """
    Generate executive-level insights from analyzed document data.

    Args:
        overall_risk_score: 0.0-1.0 aggregate risk score
        risk_level: "low" / "medium" / "high" / "critical"
        clause_risks: [{category, risk_score, risk_level, clause_id}]
        compliance_gaps: Optional [{article_id, title, status}] from compliance engine
        compliance_score: Optional 0-100 compliance score
        total_clauses: Total number of clauses analyzed

    Returns:
        Structured executive insight dict
    """
    level = risk_level.lower()

    # 1. Overall narrative
    overall_summary = _build_overall_summary(
        level, overall_risk_score, total_clauses, compliance_score,
    )

    # 2. Top risks (max 5, ordered by severity)
    top_risks = _identify_top_risks(clause_risks)

    # 3. Compliance gaps
    gap_items = _extract_compliance_gaps(compliance_gaps)

    # 4. Recommendations
    recommendations = _generate_recommendations(
        clause_risks, gap_items, level,
    )

    # 5. Key statistics
    key_statistics = _compute_statistics(
        clause_risks, total_clauses, compliance_score,
    )

    return {
        "overall_summary": overall_summary,
        "risk_level": level,
        "overall_risk_score": round(overall_risk_score, 3),
        "top_risks": top_risks,
        "compliance_gaps": gap_items,
        "recommendations": recommendations,
        "key_statistics": key_statistics,
    }


# ═══════════════════════════════════════════════════════════════
# Summary Builders
# ═══════════════════════════════════════════════════════════════

def _build_overall_summary(
    level: str,
    risk_score: float,
    total_clauses: int,
    compliance_score: Optional[float],
) -> str:
    """Build executive-level overall summary paragraph."""
    narrative = _RISK_NARRATIVES.get(level, _RISK_NARRATIVES["medium"])

    # Add compliance context if available
    if compliance_score is not None:
        if compliance_score >= 80:
            narrative += f" Compliance coverage is strong at {compliance_score:.0f}%."
        elif compliance_score >= 50:
            narrative += (
                f" Compliance coverage is moderate at {compliance_score:.0f}%, "
                f"indicating some regulatory gaps that should be addressed."
            )
        else:
            narrative += (
                f" Compliance coverage is low at {compliance_score:.0f}%, "
                f"indicating significant regulatory gaps requiring immediate attention."
            )

    narrative += f" Analysis covered {total_clauses} clauses."
    return narrative


def _identify_top_risks(
    clause_risks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Identify top 5 risk areas by aggregating clause-level risks per category.
    Deterministic: sorted by average risk score per category.
    """
    # Aggregate risks per category
    category_scores: Dict[str, List[float]] = {}
    for cr in clause_risks:
        cat = cr.get("category", "")
        score = cr.get("risk_score", 0.0)
        if cat:
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(score)

    # Compute average per category, sort by descending avg
    category_avg = []
    for cat, scores in category_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        count_high = sum(1 for s in scores if s >= 0.7)
        category_avg.append({
            "category": cat,
            "avg_score": avg,
            "max_score": max_score,
            "high_risk_count": count_high,
            "total_clauses": len(scores),
        })

    # Sort by: max_score desc → avg_score desc (deterministic)
    category_avg.sort(key=lambda x: (x["max_score"], x["avg_score"]), reverse=True)

    # Build top 5 risks
    top_risks = []
    for item in category_avg[:5]:
        cat = item["category"]
        descriptor = _CATEGORY_RISK_DESCRIPTORS.get(
            cat, f"Risk in '{cat}' category"
        )
        detail = (
            f"{descriptor}. "
            f"Found {item['total_clauses']} clause(s), "
            f"{item['high_risk_count']} high-risk (avg score: {item['avg_score']:.2f})."
        )
        top_risks.append({
            "category": cat,
            "description": detail,
            "severity": "critical" if item["max_score"] >= 0.85 else
                        "high" if item["max_score"] >= 0.7 else
                        "medium" if item["max_score"] >= 0.4 else "low",
            "avg_risk_score": round(item["avg_score"], 3),
        })

    return top_risks


def _extract_compliance_gaps(
    compliance_gaps: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    """Extract compliance gap items from compliance report data."""
    if not compliance_gaps:
        return []

    gaps = []
    for gap in compliance_gaps:
        article_id = gap.get("article_id", "")
        title = gap.get("title", "")
        status = gap.get("status", "missing")
        explanation = gap.get("explanation", "")

        gaps.append({
            "article": f"{article_id} — {title}",
            "status": status,
            "description": explanation[:200] if explanation else f"{article_id} ({title}) requires attention.",
        })

    return gaps


def _generate_recommendations(
    clause_risks: List[Dict[str, Any]],
    compliance_gaps: List[Dict[str, str]],
    risk_level: str,
) -> List[str]:
    """
    Generate actionable recommendations.
    Deterministic: based on high-risk categories + compliance gaps.
    Avoids speculative legal advice.
    """
    recommendations = []
    seen_categories = set()

    # From high-risk clauses
    high_risk = sorted(
        [cr for cr in clause_risks if cr.get("risk_score", 0) >= 0.6],
        key=lambda x: x.get("risk_score", 0),
        reverse=True,
    )

    for cr in high_risk:
        cat = cr.get("category", "")
        if cat and cat not in seen_categories:
            seen_categories.add(cat)
            rec = _CATEGORY_RECOMMENDATIONS.get(
                cat,
                f"Review clauses in the '{cat}' category for potential improvements.",
            )
            recommendations.append(rec)
        if len(recommendations) >= 5:
            break

    # From compliance gaps
    for gap in compliance_gaps[:3]:
        article = gap.get("article", "")
        if article:
            recommendations.append(
                f"Address compliance gap: {article}."
            )

    # Ensure at least one recommendation
    if not recommendations:
        if risk_level in ("high", "critical"):
            recommendations.append(
                "Conduct a thorough review of all high-risk clauses with legal counsel."
            )
        else:
            recommendations.append(
                "No critical issues identified. Maintain periodic policy review practices."
            )

    return recommendations[:8]  # Cap at 8 recommendations


def _compute_statistics(
    clause_risks: List[Dict[str, Any]],
    total_clauses: int,
    compliance_score: Optional[float],
) -> Dict[str, Any]:
    """Compute key statistics for the executive summary."""
    scores = [cr.get("risk_score", 0.0) for cr in clause_risks]

    high_risk_count = sum(1 for s in scores if s >= 0.7)
    medium_risk_count = sum(1 for s in scores if 0.4 <= s < 0.7)
    low_risk_count = sum(1 for s in scores if s < 0.4)

    # Category distribution
    categories = {}
    for cr in clause_risks:
        cat = cr.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_clauses_analyzed": total_clauses,
        "total_clauses_scored": len(scores),
        "high_risk_clauses": high_risk_count,
        "medium_risk_clauses": medium_risk_count,
        "low_risk_clauses": low_risk_count,
        "average_risk_score": round(sum(scores) / max(len(scores), 1), 3),
        "max_risk_score": round(max(scores) if scores else 0.0, 3),
        "compliance_score": compliance_score,
        "category_distribution": categories,
    }
