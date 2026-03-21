"""
Report Generator Service — generates compliance reports and audit documents.
Maps to the 'Licence Generator' block in the architecture diagram.
Produces structured risk/compliance reports for GDPR, CCPA, HIPAA.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ─── Compliance Frameworks ───────────────────────────────────
GDPR_REQUIREMENTS = [
    {"id": "GDPR-5", "name": "Lawfulness & Transparency", "categories": ["Consent", "Data Collection"]},
    {"id": "GDPR-6", "name": "Purpose Limitation", "categories": ["Data Collection", "Data Sharing"]},
    {"id": "GDPR-13", "name": "Right to Information", "categories": ["User Rights", "Compliance"]},
    {"id": "GDPR-15", "name": "Right of Access", "categories": ["User Rights"]},
    {"id": "GDPR-17", "name": "Right to Erasure", "categories": ["User Rights", "Data Retention"]},
    {"id": "GDPR-20", "name": "Data Portability", "categories": ["User Rights"]},
    {"id": "GDPR-25", "name": "Data Protection by Design", "categories": ["Security"]},
    {"id": "GDPR-32", "name": "Security of Processing", "categories": ["Security"]},
    {"id": "GDPR-33", "name": "Breach Notification", "categories": ["Security", "Compliance"]},
    {"id": "GDPR-35", "name": "Data Protection Impact", "categories": ["Compliance"]},
]

CCPA_REQUIREMENTS = [
    {"id": "CCPA-1798.100", "name": "Right to Know", "categories": ["User Rights", "Data Collection"]},
    {"id": "CCPA-1798.105", "name": "Right to Delete", "categories": ["User Rights", "Data Retention"]},
    {"id": "CCPA-1798.110", "name": "Right to Disclosure", "categories": ["Data Sharing", "Third-Party Access"]},
    {"id": "CCPA-1798.115", "name": "Right to Opt-Out of Sale", "categories": ["Data Sharing"]},
    {"id": "CCPA-1798.120", "name": "Non-Discrimination", "categories": ["User Rights"]},
    {"id": "CCPA-1798.140", "name": "Definition of Sale", "categories": ["Data Sharing", "Third-Party Access"]},
]


class ReportGenerator:
    """
    Generates structured compliance and risk reports.
    """

    @staticmethod
    def generate_compliance_report(clauses: list, framework: str = "GDPR") -> Dict:
        """
        Check clauses against a compliance framework (GDPR or CCPA).
        Returns a structured report with compliance status per requirement.
        """
        requirements = GDPR_REQUIREMENTS if framework.upper() == "GDPR" else CCPA_REQUIREMENTS
        report = {
            "framework": framework.upper(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_requirements": len(requirements),
            "requirements": [],
            "compliance_score": 0.0,
        }

        addressed_count = 0

        for req in requirements:
            # Find clauses that match this requirement's categories
            matching_clauses = [
                c for c in clauses
                if c.get("category") in req["categories"]
            ]

            if matching_clauses:
                addressed_count += 1
                # Average risk of matching clauses
                avg_risk = sum(c.get("risk_score", 0) for c in matching_clauses) / len(matching_clauses)
                status = "compliant" if avg_risk < 0.5 else "needs_review"
            else:
                status = "not_addressed"
                avg_risk = None

            report["requirements"].append({
                "id": req["id"],
                "name": req["name"],
                "status": status,
                "risk_score": round(avg_risk, 3) if avg_risk is not None else None,
                "matching_clauses": len(matching_clauses),
                "related_categories": req["categories"],
            })

        report["compliance_score"] = round(
            (addressed_count / len(requirements)) * 100, 1
        ) if requirements else 0

        return report

    @staticmethod
    def generate_risk_report(clauses: list, summary: str, overall_risk: str, overall_score: float) -> Dict:
        """
        Generate a structured risk assessment report.
        """
        high_risk = [c for c in clauses if c.get("risk_level") in ("high", "critical")]
        medium_risk = [c for c in clauses if c.get("risk_level") == "medium"]
        low_risk = [c for c in clauses if c.get("risk_level") == "low"]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "overall_risk": overall_risk,
            "overall_score": round(overall_score, 3),
            "breakdown": {
                "total_clauses": len(clauses),
                "high_risk": len(high_risk),
                "medium_risk": len(medium_risk),
                "low_risk": len(low_risk),
            },
            "high_risk_clauses": [
                {
                    "text": c["text"][:200],
                    "category": c.get("category"),
                    "risk_score": c.get("risk_score"),
                    "explanation": c.get("explanation"),
                }
                for c in high_risk
            ],
            "category_distribution": _count_categories(clauses),
        }

    @staticmethod
    def generate_executive_summary(clauses: list, summary: str, overall_risk: str, recommendations: list) -> str:
        """Generate a human-readable executive summary."""
        high = sum(1 for c in clauses if c.get("risk_level") in ("high", "critical"))
        total = len(clauses)

        lines = [
            f"## Executive Summary",
            f"",
            f"**Overall Risk Level:** {overall_risk.upper()}",
            f"",
            f"### Policy Overview",
            summary,
            f"",
            f"### Risk Assessment",
            f"- **Total clauses analyzed:** {total}",
            f"- **High/Critical risk clauses:** {high}",
            f"- **Risk ratio:** {(high/total*100):.1f}%" if total else "- N/A",
            f"",
            f"### Recommendations",
        ]
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)


def _count_categories(clauses: list) -> Dict[str, int]:
    counts = {}
    for c in clauses:
        cat = c.get("category", "Unknown")
        counts[cat] = counts.get(cat, 0) + 1
    return counts
