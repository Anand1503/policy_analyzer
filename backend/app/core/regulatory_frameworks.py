"""
Regulatory Framework Knowledge Base — structured article-to-label mappings.
Module 5: Compliance Mapping & Gap Detection.

Each framework is a dict of:
    article_id → {
        "title":             Human-readable article name
        "requirement":       Description of what is required
        "expected_labels":   Classification labels that satisfy this article
        "keywords":          Content keywords that indicate coverage
        "required_entities": NER entity types expected (optional)
        "importance_weight": Relative weight for scoring (1.0 = normal)
    }

Design:
  - Configurable & extensible — add new frameworks by defining a dict
  - Not hardcoded in service logic — engine reads from here
  - Supports custom framework upload (JSON-compatible structure)
"""

from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════
# GDPR — General Data Protection Regulation (EU)
# ═══════════════════════════════════════════════════════════════

GDPR: Dict[str, Dict[str, Any]] = {
    "Article 5": {
        "title": "Principles of Data Processing",
        "requirement": "Personal data must be processed lawfully, fairly, and transparently, collected for specified purposes, adequate, relevant, limited, accurate, and kept no longer than necessary.",
        "expected_labels": ["DATA_COLLECTION", "DATA_RETENTION"],
        "keywords": ["lawful", "fair", "transparent", "purpose limitation", "data minimization", "accuracy", "storage limitation"],
        "required_entities": [],
        "importance_weight": 1.2,
    },
    "Article 6": {
        "title": "Lawfulness of Processing",
        "requirement": "Processing must have a legal basis: consent, contract necessity, legal obligation, vital interests, public interest, or legitimate interests.",
        "expected_labels": ["DATA_COLLECTION", "COMPLIANCE_REFERENCE"],
        "keywords": ["consent", "legitimate interest", "contract", "legal obligation", "lawful basis", "legal basis"],
        "required_entities": ["REGULATION"],
        "importance_weight": 1.3,
    },
    "Article 7": {
        "title": "Conditions for Consent",
        "requirement": "Where processing is based on consent, the controller must be able to demonstrate consent. Consent must be freely given, specific, informed, and unambiguous.",
        "expected_labels": ["DATA_COLLECTION", "USER_RIGHTS"],
        "keywords": ["consent", "withdraw", "opt-in", "freely given", "informed consent"],
        "required_entities": [],
        "importance_weight": 1.1,
    },
    "Article 9": {
        "title": "Processing of Special Categories of Data",
        "requirement": "Processing of sensitive personal data (race, health, biometrics, etc.) is prohibited unless specific conditions are met.",
        "expected_labels": ["DATA_COLLECTION", "SECURITY_MEASURES"],
        "keywords": ["sensitive data", "biometric", "health data", "racial", "ethnic", "political", "religious", "genetic"],
        "required_entities": [],
        "importance_weight": 1.0,
    },
    "Article 12": {
        "title": "Transparent Information and Communication",
        "requirement": "Information must be provided in a concise, transparent, intelligible, and easily accessible form, using clear and plain language.",
        "expected_labels": ["DATA_COLLECTION"],
        "keywords": ["transparent", "clear", "plain language", "accessible", "concise"],
        "required_entities": [],
        "importance_weight": 0.9,
    },
    "Article 13": {
        "title": "Information to be Provided Where Data Is Collected",
        "requirement": "When collecting personal data, the controller must inform the data subject about identity, purposes, legal basis, recipients, retention period, and rights.",
        "expected_labels": ["DATA_COLLECTION", "USER_RIGHTS"],
        "keywords": ["identity", "contact", "purpose", "legal basis", "recipients", "retention period", "rights"],
        "required_entities": ["ORGANIZATION"],
        "importance_weight": 1.2,
    },
    "Article 14": {
        "title": "Information Where Data Not Obtained from Subject",
        "requirement": "When personal data has not been obtained from the data subject, the controller must provide information about the source and categories of data.",
        "expected_labels": ["DATA_COLLECTION", "THIRD_PARTY_TRANSFER"],
        "keywords": ["third party", "source", "categories of data", "obtained from"],
        "required_entities": [],
        "importance_weight": 0.8,
    },
    "Article 15": {
        "title": "Right of Access",
        "requirement": "Data subjects have the right to obtain confirmation of whether personal data is being processed and to access the data.",
        "expected_labels": ["USER_RIGHTS"],
        "keywords": ["access", "copy", "obtain", "right to access"],
        "required_entities": [],
        "importance_weight": 1.1,
    },
    "Article 16": {
        "title": "Right to Rectification",
        "requirement": "Data subjects have the right to obtain rectification of inaccurate personal data without undue delay.",
        "expected_labels": ["USER_RIGHTS"],
        "keywords": ["rectification", "correct", "inaccurate", "amend"],
        "required_entities": [],
        "importance_weight": 1.0,
    },
    "Article 17": {
        "title": "Right to Erasure (Right to Be Forgotten)",
        "requirement": "Data subjects have the right to obtain erasure of personal data without undue delay under certain conditions.",
        "expected_labels": ["USER_RIGHTS", "DATA_RETENTION"],
        "keywords": ["delete", "erase", "remove", "right to be forgotten", "erasure"],
        "required_entities": [],
        "importance_weight": 1.2,
    },
    "Article 18": {
        "title": "Right to Restriction of Processing",
        "requirement": "Data subjects have the right to obtain restriction of processing under certain conditions.",
        "expected_labels": ["USER_RIGHTS"],
        "keywords": ["restrict", "restriction", "suspend", "limit processing"],
        "required_entities": [],
        "importance_weight": 0.9,
    },
    "Article 20": {
        "title": "Right to Data Portability",
        "requirement": "Data subjects have the right to receive their personal data in a structured, commonly used, machine-readable format.",
        "expected_labels": ["USER_RIGHTS"],
        "keywords": ["portability", "transfer", "machine-readable", "structured format", "export"],
        "required_entities": [],
        "importance_weight": 1.0,
    },
    "Article 21": {
        "title": "Right to Object",
        "requirement": "Data subjects have the right to object to processing based on legitimate interests or direct marketing.",
        "expected_labels": ["USER_RIGHTS"],
        "keywords": ["object", "opt-out", "direct marketing", "profiling"],
        "required_entities": [],
        "importance_weight": 1.0,
    },
    "Article 25": {
        "title": "Data Protection by Design and Default",
        "requirement": "The controller must implement appropriate measures to ensure data protection principles are embedded from the design phase.",
        "expected_labels": ["SECURITY_MEASURES", "DATA_COLLECTION"],
        "keywords": ["privacy by design", "by default", "data protection", "appropriate measures"],
        "required_entities": [],
        "importance_weight": 1.0,
    },
    "Article 28": {
        "title": "Processor Obligations",
        "requirement": "Processing by a processor must be governed by a contract with specific data protection obligations.",
        "expected_labels": ["THIRD_PARTY_TRANSFER", "DATA_SHARING"],
        "keywords": ["processor", "sub-processor", "data processing agreement", "contract"],
        "required_entities": ["ORGANIZATION"],
        "importance_weight": 1.0,
    },
    "Article 30": {
        "title": "Records of Processing Activities",
        "requirement": "Each controller and processor must maintain records of processing activities under its responsibility.",
        "expected_labels": ["DATA_COLLECTION", "COMPLIANCE_REFERENCE"],
        "keywords": ["records", "processing activities", "documentation", "register"],
        "required_entities": [],
        "importance_weight": 0.8,
    },
    "Article 32": {
        "title": "Security of Processing",
        "requirement": "Controllers and processors must implement appropriate technical and organizational measures to ensure security.",
        "expected_labels": ["SECURITY_MEASURES"],
        "keywords": ["encryption", "pseudonymisation", "confidentiality", "integrity", "availability", "security measures", "technical measures"],
        "required_entities": [],
        "importance_weight": 1.2,
    },
    "Article 33": {
        "title": "Notification of Data Breach to Authority",
        "requirement": "In the case of a personal data breach, the controller must notify the supervisory authority within 72 hours.",
        "expected_labels": ["SECURITY_MEASURES", "COMPLIANCE_REFERENCE"],
        "keywords": ["breach", "notification", "72 hours", "supervisory authority", "data breach"],
        "required_entities": ["REGULATION"],
        "importance_weight": 1.1,
    },
    "Article 34": {
        "title": "Communication of Data Breach to Data Subject",
        "requirement": "When a data breach is likely to result in a high risk, the controller must communicate the breach to the data subject.",
        "expected_labels": ["SECURITY_MEASURES", "USER_RIGHTS"],
        "keywords": ["breach notification", "communicate", "high risk", "data subject"],
        "required_entities": [],
        "importance_weight": 1.0,
    },
    "Article 35": {
        "title": "Data Protection Impact Assessment",
        "requirement": "A DPIA is required when processing is likely to result in a high risk to the rights and freedoms of natural persons.",
        "expected_labels": ["COMPLIANCE_REFERENCE", "SECURITY_MEASURES"],
        "keywords": ["impact assessment", "DPIA", "risk assessment", "high risk"],
        "required_entities": [],
        "importance_weight": 0.9,
    },
    "Article 44": {
        "title": "General Principle for International Transfers",
        "requirement": "Any transfer of personal data to a third country must have adequate safeguards.",
        "expected_labels": ["THIRD_PARTY_TRANSFER", "DATA_SHARING"],
        "keywords": ["international transfer", "third country", "adequate safeguards", "cross-border"],
        "required_entities": ["LOCATION"],
        "importance_weight": 1.1,
    },
    "Article 46": {
        "title": "Transfers Subject to Appropriate Safeguards",
        "requirement": "Transfers to third countries may occur with appropriate safeguards like standard contractual clauses.",
        "expected_labels": ["THIRD_PARTY_TRANSFER"],
        "keywords": ["standard contractual clauses", "binding corporate rules", "safeguards", "adequacy"],
        "required_entities": [],
        "importance_weight": 0.9,
    },
}


# ═══════════════════════════════════════════════════════════════
# CCPA — California Consumer Privacy Act
# ═══════════════════════════════════════════════════════════════

CCPA: Dict[str, Dict[str, Any]] = {
    "Section 1798.100": {
        "title": "Right to Know What Information Is Being Collected",
        "requirement": "A consumer has the right to request that a business disclose what categories and specific pieces of personal information it has collected.",
        "expected_labels": ["DATA_COLLECTION", "USER_RIGHTS"],
        "keywords": ["categories", "personal information", "collected", "right to know", "disclose"],
        "required_entities": [],
        "importance_weight": 1.2,
    },
    "Section 1798.105": {
        "title": "Right to Delete Personal Information",
        "requirement": "A consumer has the right to request deletion of personal information collected by a business.",
        "expected_labels": ["USER_RIGHTS", "DATA_RETENTION"],
        "keywords": ["delete", "deletion", "remove", "erase"],
        "required_entities": [],
        "importance_weight": 1.2,
    },
    "Section 1798.110": {
        "title": "Right to Know About Personal Information Sold or Disclosed",
        "requirement": "A consumer has the right to request categories of personal information collected, sold, or disclosed for a business purpose.",
        "expected_labels": ["DATA_SHARING", "DATA_COLLECTION"],
        "keywords": ["sold", "disclosed", "business purpose", "categories"],
        "required_entities": [],
        "importance_weight": 1.1,
    },
    "Section 1798.115": {
        "title": "Right to Opt-Out of Sale of Personal Information",
        "requirement": "A consumer has the right to direct a business that sells personal information to stop selling.",
        "expected_labels": ["USER_RIGHTS", "DATA_SHARING"],
        "keywords": ["opt-out", "do not sell", "sale", "stop selling", "opt out"],
        "required_entities": [],
        "importance_weight": 1.3,
    },
    "Section 1798.120": {
        "title": "Right to Opt-Out — Prohibition on Selling",
        "requirement": "A business must not sell personal information of a consumer who has opted out.",
        "expected_labels": ["DATA_SHARING", "USER_RIGHTS"],
        "keywords": ["opt-out", "prohibit", "do not sell", "selling"],
        "required_entities": [],
        "importance_weight": 1.2,
    },
    "Section 1798.125": {
        "title": "Non-Discrimination",
        "requirement": "A business may not discriminate against a consumer for exercising their CCPA rights.",
        "expected_labels": ["USER_RIGHTS"],
        "keywords": ["non-discrimination", "equal service", "equal price", "penalize"],
        "required_entities": [],
        "importance_weight": 1.0,
    },
    "Section 1798.130": {
        "title": "Notice and Procedures for Requests",
        "requirement": "Business must provide at least two methods for consumers to submit requests and must respond within 45 days.",
        "expected_labels": ["USER_RIGHTS", "DATA_COLLECTION"],
        "keywords": ["request", "45 days", "toll-free", "verify", "contact"],
        "required_entities": ["ORGANIZATION"],
        "importance_weight": 1.0,
    },
    "Section 1798.135": {
        "title": "Do Not Sell My Personal Information Link",
        "requirement": "A business that sells personal information must provide a clear and conspicuous 'Do Not Sell My Personal Information' link.",
        "expected_labels": ["USER_RIGHTS", "DATA_SHARING"],
        "keywords": ["do not sell", "link", "homepage", "opt-out"],
        "required_entities": [],
        "importance_weight": 1.1,
    },
    "Section 1798.140": {
        "title": "Definitions — Personal Information",
        "requirement": "Defines personal information broadly including identifiers, biometrics, browsing history, geolocation, and inferences.",
        "expected_labels": ["DATA_COLLECTION"],
        "keywords": ["personal information", "identifiers", "biometric", "geolocation", "browsing history", "inferences"],
        "required_entities": [],
        "importance_weight": 0.8,
    },
    "Section 1798.145": {
        "title": "Exemptions",
        "requirement": "Defines exemptions including publicly available information, de-identified data, and employee data.",
        "expected_labels": ["COMPLIANCE_REFERENCE"],
        "keywords": ["exemption", "publicly available", "de-identified", "aggregate"],
        "required_entities": [],
        "importance_weight": 0.7,
    },
    "Section 1798.150": {
        "title": "Data Breach Private Right of Action",
        "requirement": "Consumers whose unencrypted personal information is breached may bring a civil action.",
        "expected_labels": ["SECURITY_MEASURES", "LIABILITY_LIMITATION"],
        "keywords": ["breach", "civil action", "damages", "unauthorized access", "unencrypted"],
        "required_entities": [],
        "importance_weight": 1.1,
    },
    "Section 1798.185": {
        "title": "Regulations — Children's Data",
        "requirement": "A business must not sell personal information of consumers under age 16 without affirmative authorization.",
        "expected_labels": ["CHILDREN_PRIVACY", "DATA_SHARING"],
        "keywords": ["children", "under 16", "under 13", "minor", "parental consent", "affirmative authorization"],
        "required_entities": [],
        "importance_weight": 1.2,
    },
}


# ═══════════════════════════════════════════════════════════════
# Framework Registry — extensible mapping
# ═══════════════════════════════════════════════════════════════

FRAMEWORK_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "GDPR": GDPR,
    "CCPA": CCPA,
}


def get_framework(name: str) -> Dict[str, Dict[str, Any]]:
    """Get a regulatory framework by name. Case-insensitive."""
    key = name.upper()
    if key not in FRAMEWORK_REGISTRY:
        raise ValueError(
            f"Unknown framework '{name}'. Available: {list(FRAMEWORK_REGISTRY.keys())}"
        )
    return FRAMEWORK_REGISTRY[key]


def get_available_frameworks() -> list:
    """List all registered framework names."""
    return list(FRAMEWORK_REGISTRY.keys())


def register_custom_framework(name: str, articles: Dict[str, Dict[str, Any]]) -> None:
    """
    Register a custom regulatory framework at runtime.
    Used for research experiments and custom compliance mapping.
    """
    FRAMEWORK_REGISTRY[name.upper()] = articles
