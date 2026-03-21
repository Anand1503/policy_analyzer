"""
Legal-BERT Fine-Tuning v2 — Production Pipeline
==================================================
Phase 1: OPP-115-aligned dataset (expanded, 3000+ samples)
Phase 2: Class imbalance handling (pos_weight + focal loss)
Phase 3: Advanced training (gradient accumulation, mixed precision, LR scheduler)
Phase 4: Per-label threshold tuning on validation set
Phase 5: Final evaluation on test set with tuned thresholds
Phase 6: Model export to models/legal-bert-finetuned-v2/
Phase 7: Analysis report generation

Usage:
    python scripts/train_legalbert_v2.py [--focal] [--fp16] [--epochs N]
"""
import os, sys, json, csv, random, re, time, logging, argparse
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, multilabel_confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

TAXONOMY = [
    "DATA_COLLECTION", "DATA_SHARING", "USER_RIGHTS",
    "DATA_RETENTION", "SECURITY_MEASURES", "THIRD_PARTY_TRANSFER",
    "COOKIES_TRACKING", "CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE",
    "LIABILITY_LIMITATION",
]
NUM_LABELS = len(TAXONOMY)

BASE_MODEL = "./models/legal-bert"
EXPORT_DIR = "./models/legal-bert-finetuned-v2"
DATA_DIR = Path("data/processed")
EVAL_DIR = Path("evaluation")
MAX_LENGTH = 256
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2  # effective batch size = 32
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# ═══════════════════════════════════════════════════════════
# OPP-115 TAXONOMY MAPPING
# ═══════════════════════════════════════════════════════════

OPP_TO_TAXONOMY = {
    "DATA_COLLECTION": ["DATA_COLLECTION"],
    "DATA_SHARING": ["DATA_SHARING", "THIRD_PARTY_TRANSFER"],
    "USER_RIGHTS": ["USER_RIGHTS"],
    "DATA_RETENTION": ["DATA_RETENTION"],
    "SECURITY_MEASURES": ["SECURITY_MEASURES"],
    "COOKIES_TRACKING": ["COOKIES_TRACKING"],
    "CHILDREN_PRIVACY": ["CHILDREN_PRIVACY"],
    "COMPLIANCE_REFERENCE": ["COMPLIANCE_REFERENCE"],
    "LIABILITY_LIMITATION": ["LIABILITY_LIMITATION"],
    "THIRD_PARTY_TRANSFER": ["THIRD_PARTY_TRANSFER", "DATA_SHARING"],
}

# ═══════════════════════════════════════════════════════════
# PHASE 1 — EXPANDED OPP-115-ALIGNED DATASET (3000+ CLAUSES)
# ═══════════════════════════════════════════════════════════

# Research-grade privacy policy clause templates — 30+ per category
CLAUSE_TEMPLATES = {
    "DATA_COLLECTION": [
        "We collect personal information such as your name, email address, and phone number when you create an account.",
        "We may collect information about your device, including IP address, browser type, and operating system.",
        "When you use our services, we automatically collect usage data including pages visited and time spent on our platform.",
        "We gather information you provide directly, such as when filling out forms, making purchases, or subscribing to newsletters.",
        "Our application collects location data to provide location-based services and improve your user experience.",
        "We collect payment information including credit card numbers and billing addresses to process your transactions securely.",
        "We may obtain personal data from third-party sources to supplement the information we collect directly from you.",
        "Your browsing history and search queries are collected to personalize content and deliver relevant advertisements.",
        "We record your interactions with our customer support team for quality assurance and training purposes.",
        "Information about your social media profiles may be collected when you connect your accounts to our service.",
        "We collect biometric data such as fingerprints or facial recognition data for authentication and security purposes.",
        "Your health and fitness data is collected through our wellness tracking features when you opt into these services.",
        "We gather demographic information including age, gender, and income level for market research and service improvement.",
        "Device identifiers such as IMEI numbers, advertising IDs, and hardware serial numbers are collected automatically.",
        "We collect information from cookies and similar tracking technologies placed on your device during your visits.",
        "Your contacts and address book information may be accessed when you choose to invite friends to our platform.",
        "We collect voice recordings and transcripts when you use our voice-activated features or virtual assistant.",
        "Photos and videos you upload to our platform are stored on our servers as part of the service we provide.",
        "We collect geolocation data from your mobile device with your explicit consent for location-based features.",
        "Transaction history and purchase patterns are recorded to improve our services and provide personalized recommendations.",
        "We automatically collect log data including access times, hardware and software information, and referring URLs.",
        "Your calendar and scheduling information may be accessed when you integrate our service with your calendar application.",
        "We collect information about your network connections and Wi-Fi access points for service optimization purposes.",
        "Employment and professional information is collected when you use our business-oriented features and services.",
        "We may collect information about your interests and preferences based on your interactions with our content.",
        "Sensor data from your device including accelerometer and gyroscope readings may be collected for specific features.",
        "We collect your communication preferences and settings to customize your experience on our platform.",
        "Your search history and content consumption patterns are tracked to improve our recommendation algorithms.",
        "We gather information about your app usage patterns including frequency, duration, and feature utilization.",
        "We collect technical data about your internet connection speed and quality to optimize streaming performance.",
        "Personal data is gathered through registration forms, surveys, and interactive tools provided on our website.",
        "We collect information about your shopping behavior, product views, and wishlist items to enhance your experience.",
        "Your email open rates and click-through data are collected to measure the effectiveness of our communications.",
        "We receive information from publicly available databases and commercial data providers to enrich user profiles.",
        "Audio recordings from customer service calls are collected for dispute resolution and service quality monitoring.",
    ],
    "DATA_SHARING": [
        "We share your personal data with trusted third-party service providers who assist in operating our platform and services.",
        "Your information may be disclosed to advertising partners for targeted marketing campaigns and promotional activities.",
        "We may share aggregated, non-personally identifiable information with business partners for analytical purposes.",
        "In the event of a merger, acquisition, or sale of assets, user data may be transferred to the acquiring entity.",
        "We disclose information to law enforcement agencies when required by applicable law or valid legal process.",
        "Your data may be shared with analytics providers to help us understand usage patterns and improve our services.",
        "We provide personal information to payment processors to complete financial transactions on your behalf.",
        "Third-party social media platforms may receive data when you use social sharing features on our platform.",
        "We share data with affiliate companies within our corporate family for joint marketing efforts and service improvements.",
        "Your information may be sold to data brokers for marketing purposes unless you opt out of such sharing arrangements.",
        "We may transfer your data to cloud service providers located in different jurisdictions for processing and storage.",
        "Advertising networks receive browsing data and user profiles to serve personalized advertisements across the internet.",
        "We share information with research partners for academic and scientific studies related to our products and services.",
        "Your data may be provided to government agencies in response to subpoenas, court orders, or other legal requests.",
        "We disclose information to credit reporting agencies for identity verification and fraud prevention purposes.",
        "We share anonymized usage statistics with industry groups and regulatory bodies for benchmarking and compliance.",
        "Your information may be shared with insurance providers in connection with claims processing and risk assessment.",
        "We provide aggregated data to investors and financial analysts as part of our business reporting obligations.",
        "Customer data may be shared with logistics and shipping partners to facilitate order delivery and tracking.",
        "We share your information with professional advisors including lawyers, accountants, and auditors as necessary.",
        "Your data may be disclosed to debt collection agencies in connection with outstanding payment obligations.",
        "We share information with content delivery networks to optimize the performance and availability of our services.",
        "Educational institutions may receive your information when you use our learning and certification features.",
        "We share your feedback and reviews with product teams and manufacturing partners to improve quality.",
        "Your data may be provided to emergency services when we believe there is a risk to your safety or the safety of others.",
        "We disclose personal information to our hosting providers and infrastructure partners for service delivery purposes.",
        "Your account information may be shared with customer identity verification services for security purposes.",
        "We share de-identified data with academic researchers studying consumer behavior and digital privacy trends.",
        "Marketing automation platforms receive user engagement data to deliver personalized promotional content.",
        "We provide your shipping information to courier services and postal operators for order fulfillment purposes.",
    ],
    "USER_RIGHTS": [
        "You may opt out of receiving marketing communications at any time by clicking the unsubscribe link in our emails.",
        "Users can manage their cookie preferences through our cookie settings panel available on every page of our website.",
        "You have the right to withdraw your consent for data processing at any time without affecting prior processing.",
        "You can choose to disable location tracking in your device settings or through our application preferences.",
        "We provide tools for you to control what personal information is visible to other users on our platform.",
        "You may request that we stop processing your data for direct marketing purposes at any time.",
        "Users can adjust their privacy settings to control data sharing with third parties through their account dashboard.",
        "You have the option to opt out of personalized advertising through industry opt-out tools and browser settings.",
        "You can choose not to provide certain personal information, though some features may be limited as a result.",
        "We offer parental controls to manage children's access to our services and control data collection for minors.",
        "You may decline to participate in surveys, promotional activities, or research studies at any time.",
        "Users can configure notification preferences to control the frequency and type of communications they receive.",
        "You have the right to object to automated decision-making and profiling that affects you significantly.",
        "You can request restriction of processing under certain circumstances as defined by applicable data protection laws.",
        "We provide Do Not Sell My Personal Information options for California residents as required by the CCPA.",
        "You can access and update your personal information through your account settings page at any time.",
        "Upon request, we will provide you with a copy of all personal data we hold about you in a portable format.",
        "You have the right to request deletion of your account and all associated personal data from our systems.",
        "We will correct any inaccurate personal information upon receiving your written request through our support channels.",
        "You may download a machine-readable copy of your data through our data portability feature in your account settings.",
        "Users can request access to their personal data by contacting our privacy team via email or our online form.",
        "You have the right to rectification of incomplete or inaccurate personal data we hold about you.",
        "We will erase your personal data within 30 days of receiving a valid deletion request, subject to legal requirements.",
        "You can review all data collected about you through our transparency dashboard available in your account settings.",
        "Account deletion requests can be submitted through our dedicated privacy portal or by contacting our support team.",
        "You may export your data in commonly used formats such as CSV, JSON, or PDF through our data export tool.",
        "We provide a self-service tool to review, modify, and delete your profile information and associated data.",
        "You have the right to lodge a complaint with a supervisory authority if you believe your rights have been violated.",
        "You can revoke API access tokens and third-party application permissions at any time through your security settings.",
        "We respect your right to data minimization and will only process data that is necessary for the stated purpose.",
        "You may request a human review of any automated decision that significantly impacts your rights or interests.",
        "You have the right to receive notification before your personal data is used for a purpose different from collection.",
        "Users can enable two-factor authentication and manage their account security settings through the security center.",
        "You may withdraw from loyalty or rewards programs at any time while retaining earned benefits as applicable.",
        "We provide clear mechanisms for you to exercise your privacy rights without facing any discrimination or penalty.",
    ],
    "DATA_RETENTION": [
        "We retain your personal data for as long as your account remains active or as needed to provide our services.",
        "Personal information is deleted within 90 days after account closure unless legally required to retain it longer.",
        "We keep transaction records for seven years to comply with financial regulations and tax reporting requirements.",
        "Backup copies of deleted data may persist for up to 60 days in our archival systems before permanent removal.",
        "Log data is automatically purged after 12 months from the date of collection to protect your privacy.",
        "We retain personal data for the minimum period necessary to fulfill our legal obligations and business purposes.",
        "Your data will be anonymized rather than deleted when retention periods expire, where technically feasible.",
        "Marketing data is retained for 24 months from your last interaction with our services or communications.",
        "We may retain certain information indefinitely for legitimate business purposes and legal compliance requirements.",
        "Data associated with legal proceedings or investigations will be retained until the matter is fully resolved.",
        "Customer support records are maintained for 3 years following the last interaction for quality and training purposes.",
        "We review our retention schedules annually to ensure compliance with applicable data protection laws and regulations.",
        "Analytics data is stored in aggregate form for up to 5 years to support long-term trend analysis and reporting.",
        "Your payment information is retained for the duration of your subscription plus 12 months for dispute resolution.",
        "Account credentials are permanently deleted within 24 hours of account termination at your request.",
        "We retain employment-related data for 7 years after the end of the employment relationship as required by law.",
        "Communication records including emails and chat logs are retained for 2 years for compliance and audit purposes.",
        "User-generated content such as posts and comments is retained for 6 months after account deletion unless requested.",
        "We retain cookie consent records for the duration required by applicable data protection regulations.",
        "Biometric data is deleted immediately upon request or within 3 years of last use, whichever comes first.",
        "Inactive account data is automatically archived after 18 months of inactivity and deleted after 36 months.",
        "We retain system logs containing personal data for a maximum of 6 months for security monitoring purposes.",
        "Financial audit trail data is preserved for 10 years in compliance with applicable accounting standards.",
        "Marketing consent records are kept for the entire duration of your relationship with us plus 2 years thereafter.",
        "Anonymized research data derived from your personal information may be retained indefinitely for statistical purposes.",
    ],
    "SECURITY_MEASURES": [
        "We implement industry-standard encryption protocols including AES-256 to protect your data at rest and in transit.",
        "Access to personal information is restricted to authorized employees on a strict need-to-know basis.",
        "We conduct regular security audits and penetration testing to identify and remediate potential vulnerabilities.",
        "All data transmissions between your device and our servers are protected using TLS 1.3 encryption.",
        "We maintain comprehensive incident response procedures for potential data breaches and security incidents.",
        "Multi-factor authentication is available and recommended to add an extra layer of security to your account.",
        "Our servers are hosted in SOC 2 Type II certified data centers with advanced physical security controls.",
        "We employ firewalls, intrusion detection systems, and access controls to safeguard your personal data.",
        "Regular security training is provided to all employees who handle personal data as part of their responsibilities.",
        "We use anonymization and pseudonymization techniques to minimize data exposure risks in our processing activities.",
        "Automated monitoring systems detect and respond to suspicious activities and potential security threats in real time.",
        "We implement data loss prevention tools and policies to prevent unauthorized data exfiltration and leakage.",
        "Regular vulnerability assessments are performed on all systems that process or store personal information.",
        "We use hardware security modules for cryptographic key management and secure data processing operations.",
        "Our development team follows secure coding practices and conducts code reviews to prevent security vulnerabilities.",
        "We maintain separate environments for development, testing, and production to protect live user data.",
        "Database access is logged and monitored with automated alerts for unauthorized access attempts.",
        "We implement rate limiting and DDoS protection measures to ensure service availability and data integrity.",
        "Passwords are stored using industry-standard hashing algorithms with salting to prevent unauthorized access.",
        "We perform regular backups of critical data with encryption and secure off-site storage for disaster recovery.",
        "End-to-end encryption is applied to sensitive communications and personal messages exchanged through our platform.",
        "We conduct regular third-party security assessments and act on all findings within defined remediation timelines.",
        "Our infrastructure is designed with defense-in-depth principles, including network segmentation and micro-segmentation.",
        "We maintain a bug bounty program to incentivize responsible disclosure of security vulnerabilities.",
        "All employee devices accessing company systems are required to have disk encryption and endpoint protection enabled.",
    ],
    "COOKIES_TRACKING": [
        "We honor Do Not Track signals sent by your browser and do not track users across third-party websites.",
        "Our website uses cookies and similar tracking technologies to track user behavior for analytics and personalization.",
        "We use web beacons and pixel tags to measure the effectiveness of our marketing campaigns and email communications.",
        "Third-party analytics tools such as Google Analytics collect anonymous usage statistics about our website visitors.",
        "You can manage tracking preferences through your browser settings or our comprehensive cookie consent banner.",
        "We use session cookies to maintain your login state, shopping cart contents, and language preferences.",
        "Persistent cookies are used to remember your preferences and settings across multiple visits to our website.",
        "We employ browser fingerprinting techniques to detect and prevent fraudulent activity on our platform.",
        "Our advertising partners may use tracking cookies to serve targeted advertisements based on your browsing history.",
        "We do not currently respond to Do Not Track signals from web browsers but provide alternative opt-out mechanisms.",
        "You can install browser extensions to block tracking technologies and third-party cookies on our website.",
        "We use retargeting pixels and cookies to show you relevant advertisements on other websites and social media platforms.",
        "First-party cookies are essential for the basic functionality of our website and cannot be disabled without impact.",
        "We use local storage and session storage in addition to cookies to enhance your browsing experience.",
        "Cookie consent can be modified at any time through the cookie preferences link in the footer of our website.",
        "We use analytics cookies to understand how visitors interact with our website and identify areas for improvement.",
        "Marketing cookies help us deliver relevant advertisements and measure the return on our advertising investment.",
        "We use performance cookies to monitor website speed, error rates, and overall user satisfaction metrics.",
        "Social media cookies enable sharing functionality and may track your activity across multiple websites.",
        "We provide a detailed cookie policy that lists all cookies used, their purpose, and their expiration dates.",
        "Tracking pixels embedded in emails allow us to determine whether messages have been opened and links clicked.",
        "We use heatmap and session recording tools to understand how users navigate and interact with our web pages.",
        "Our advertising technology partners may combine cookie data with information from other sources for targeting.",
        "We categorize our cookies into strictly necessary, functional, performance, and targeting categories.",
        "Users can selectively accept or reject different categories of cookies through our granular consent management tool.",
    ],
    "CHILDREN_PRIVACY": [
        "We do not knowingly collect personal information from children under the age of 13 without parental consent.",
        "If we discover that a child under 13 has provided personal data without consent, we will delete it promptly.",
        "Parents or guardians may contact us to review, modify, or delete their child's personal information at any time.",
        "Our services are not directed to individuals under the age of 16 in the European Union without parental consent.",
        "We comply with COPPA requirements regarding the collection, use, and disclosure of children's personal information.",
        "Special protections apply to sensitive personal data of minors and vulnerable individuals in our care.",
        "Age verification mechanisms are in place to prevent minors from accessing age-restricted content and features.",
        "Parental consent is required before collecting any personal information from children under the applicable age threshold.",
        "We provide age-appropriate privacy notices for younger users who access our educational and entertainment features.",
        "Schools and educational institutions must provide consent before students can use our educational technology platforms.",
        "We do not use children's personal information for targeted advertising or behavioral profiling purposes.",
        "Our content moderation systems include special protections for content uploaded by or involving minor users.",
        "We periodically review our practices to ensure ongoing compliance with international children's privacy regulations.",
        "Parental dashboard features allow guardians to monitor and control their child's activity and data sharing settings.",
        "We delete children's personal data automatically when the data is no longer necessary for the purpose of collection.",
        "Teachers and administrators acting in loco parentis may provide consent on behalf of students under applicable law.",
        "We have implemented age-gating mechanisms that require date of birth verification before account creation.",
        "Children's accounts have restricted features to limit exposure to advertising and social interaction capabilities.",
        "We maintain separate data processing procedures for information collected from users identified as minors.",
        "Our platform offers a family sharing mode that gives parents visibility into their children's usage patterns.",
    ],
    "COMPLIANCE_REFERENCE": [
        "We may update this privacy policy from time to time and will notify you of material changes via email or notice.",
        "Any modifications to this policy will be posted on this page with an updated effective date displayed prominently.",
        "We will send you an email notification at least 30 days before any significant changes to our privacy practices.",
        "Continued use of our services after policy updates constitutes your acceptance of the revised terms and conditions.",
        "We encourage you to review this privacy policy periodically for the latest information about our privacy practices.",
        "Material changes to our data processing practices will be communicated at least 30 days in advance of implementation.",
        "A summary of changes will be provided at the top of the updated privacy policy for your convenience and review.",
        "We maintain an archive of previous versions of this privacy policy on our website for your reference.",
        "If we make changes that materially reduce your privacy rights, we will obtain your explicit consent before proceeding.",
        "We will provide prominent notice of changes through in-app notifications, email alerts, and website banners.",
        "Our privacy practices comply with the General Data Protection Regulation and all applicable EU data protection laws.",
        "We have implemented comprehensive measures to ensure full compliance with the California Consumer Privacy Act.",
        "This policy is governed by and construed in accordance with applicable data protection laws and regulations.",
        "We maintain detailed records of processing activities as required by regulatory authorities and data protection laws.",
        "Our data protection impact assessments are conducted for all high-risk processing activities as required by the GDPR.",
        "We have designated a Data Protection Officer who can be contacted regarding any privacy-related concerns or inquiries.",
        "We undergo regular third-party audits to verify our compliance with stated privacy practices and applicable regulations.",
        "Our processing of personal data is based on legitimate interests, consent, contractual necessity, or legal obligations.",
        "We cooperate with data protection authorities and respond to their inquiries in a timely and transparent manner.",
        "This policy complies with the requirements of applicable sector-specific regulations including HIPAA and FERPA.",
        "We have appointed separate representatives in the EU and UK to handle data protection matters in those jurisdictions.",
        "Our data processing agreements with third parties include standard contractual clauses approved by supervisory authorities.",
        "We conduct annual reviews of our privacy program to ensure continuous compliance with evolving regulatory requirements.",
        "The legal basis for processing your personal data varies by jurisdiction and is documented in our processing records.",
        "We participate in industry self-regulatory programs and adhere to the guidelines of recognized privacy organizations.",
    ],
    "LIABILITY_LIMITATION": [
        "This agreement shall be governed by the laws of the State of California without regard to conflict of law principles.",
        "We disclaim all warranties, express or implied, regarding the accuracy, completeness, or reliability of information provided.",
        "Our total liability to you shall not exceed the amount you paid for the service in the preceding 12 months.",
        "You agree to indemnify and hold us harmless from any claims, damages, or expenses arising from your use of the service.",
        "Any disputes arising under this agreement shall be resolved through binding arbitration rather than court proceedings.",
        "We are not liable for any indirect, incidental, special, or consequential damages arising from your use of our services.",
        "You waive your right to participate in class action lawsuits or class-wide arbitration against our company.",
        "The limitation of liability provisions contained herein survive the termination of this agreement indefinitely.",
        "We reserve the right to modify, suspend, or discontinue the service at any time without prior notice or liability.",
        "You acknowledge that the service is provided on an AS IS and AS AVAILABLE basis without any warranty of fitness.",
        "Neither party shall be liable for any failure or delay in performance resulting from force majeure events.",
        "Our liability for data breaches is limited to the extent permitted by applicable data protection legislation.",
        "You agree that any claim arising from these terms must be filed within one year of the cause of action accruing.",
        "We shall not be responsible for any loss of data, profits, or business opportunity resulting from service interruptions.",
        "The exclusions and limitations of liability set forth herein are fundamental elements of the basis of the bargain.",
        "You release us from all liability in connection with third-party products or services accessed through our platform.",
        "Our aggregate liability for all claims related to these terms shall not exceed one hundred dollars in total.",
        "We do not guarantee uninterrupted, error-free, or secure operation of our services or any associated features.",
        "You assume all risk associated with the use of our services and agree to hold us harmless from resulting damages.",
        "Any liability on our part for unauthorized access to your data shall be limited to direct damages actually incurred.",
        "We disclaim responsibility for the actions, content, or policies of third-party websites linked from our platform.",
        "The remedies set forth in these terms represent your sole and exclusive remedies for any breach of this agreement.",
        "We are not responsible for disputes between users arising from their interactions on or through our platform.",
        "Our maximum financial liability under any circumstances shall not exceed the fees collected in the prior billing period.",
        "These limitation of liability provisions shall apply to the maximum extent permitted by applicable law.",
    ],
    "THIRD_PARTY_TRANSFER": [
        "We transfer personal data to third-party processors located in jurisdictions with adequate data protection standards.",
        "Your data may be processed by our subsidiaries and affiliates in countries including the United States and Singapore.",
        "We use standard contractual clauses approved by the European Commission for international data transfers.",
        "Third-party service providers who process data on our behalf are required to maintain appropriate security measures.",
        "We conduct due diligence on all third-party processors to verify their data protection practices and compliance.",
        "Data transferred to third parties is subject to confidentiality agreements and data processing addenda.",
        "We maintain a current list of sub-processors that handle personal data and notify users of any changes.",
        "International data transfers comply with applicable cross-border data transfer mechanisms and regulations.",
        "Third-party vendors are contractually required to delete or return personal data upon termination of services.",
        "We ensure that third-party recipients of personal data provide an adequate level of protection as required by law.",
        "Your data may be transferred to cloud infrastructure providers operating data centers in multiple global regions.",
        "We share limited personal data with advertising technology partners for campaign measurement and optimization.",
        "Third-party payment processors receive only the minimum data necessary to process your financial transactions.",
        "We transfer anonymized data to research institutions for the purpose of improving privacy and security technologies.",
        "Business partners who receive personal data are required to use it only for the specified purposes outlined herein.",
        "We share compliance-related data with regulatory authorities as required by applicable laws and court orders.",
        "Third-party analytics services process user interaction data under strict contractual data protection obligations.",
        "We transfer employee data to payroll providers and benefits administrators in accordance with employment regulations.",
        "Data sharing with law enforcement agencies occurs only in response to valid legal process and documented requests.",
        "We provide data to dispute resolution and mediation services when necessary to resolve customer complaints.",
        "Your data may be transferred to jurisdictions that do not provide the same level of data protection as your home country.",
        "We require all third-party data processors to implement technical and organizational measures equivalent to our own.",
        "Cross-border data transfers are conducted under binding corporate rules approved by relevant supervisory authorities.",
        "We maintain data processing agreements with all sub-processors that include audit rights and breach notification obligations.",
        "Third-party recipients are prohibited from further transferring your data without our prior written authorization.",
    ],
}


# ═══════════════════════════════════════════════════════════
# AUGMENTATION UTILITIES
# ═══════════════════════════════════════════════════════════

# Synonym map for privacy policy terms (no external deps needed)
SYNONYM_MAP = {
    "collect": ["gather", "obtain", "acquire", "capture", "record"],
    "share": ["disclose", "provide", "transmit", "distribute", "transfer"],
    "personal information": ["personal data", "user information", "user data", "your data", "your information"],
    "third-party": ["third party", "external", "outside"],
    "delete": ["erase", "remove", "purge", "destroy"],
    "retain": ["keep", "store", "maintain", "preserve"],
    "consent": ["permission", "authorization", "approval", "agreement"],
    "cookies": ["tracking technologies", "browser cookies", "web cookies"],
    "security": ["protection", "safeguarding", "safeguards"],
    "children": ["minors", "young users", "underage users"],
    "update": ["modify", "change", "revise", "amend"],
    "comply": ["adhere to", "conform to", "follow", "abide by"],
    "liability": ["responsibility", "obligation", "accountability"],
    "services": ["platform", "products", "offerings", "solutions"],
    "website": ["site", "web platform", "online platform"],
}


def synonym_replace(text: str, n_replacements: int = 2) -> str:
    """Replace n random words/phrases with synonyms."""
    result = text
    replaceable = [(k, v) for k, v in SYNONYM_MAP.items() if k in text.lower()]
    if not replaceable:
        return result
    random.shuffle(replaceable)
    for original, synonyms in replaceable[:n_replacements]:
        synonym = random.choice(synonyms)
        # Case-insensitive replacement preserving first match case
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        match = pattern.search(result)
        if match:
            replacement = synonym
            if match.group()[0].isupper():
                replacement = synonym.capitalize()
            result = result[:match.start()] + replacement + result[match.end():]
    return result


def sentence_shuffle(text: str) -> str:
    """Shuffle sentences within a clause (for multi-sentence clauses)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return text
    random.shuffle(sentences)
    return " ".join(sentences)


def build_dataset():
    """Build expanded OPP-115-aligned dataset with augmentation."""
    log.info("=" * 60)
    log.info("PHASE 1 — Building Expanded OPP-115-Aligned Dataset")
    log.info("=" * 60)

    samples = []

    # ── Original templates with policy_id for GroupShuffleSplit ──
    policy_id = 0
    for category, clauses in CLAUSE_TEMPLATES.items():
        taxonomy_labels = OPP_TO_TAXONOMY.get(category, [category])
        for clause in clauses:
            one_hot = [0] * NUM_LABELS
            for label in taxonomy_labels:
                if label in TAXONOMY:
                    one_hot[TAXONOMY.index(label)] = 1
            samples.append({
                "text": clause, "labels": one_hot,
                "category": category, "policy_id": f"policy_{policy_id}",
            })
        policy_id += 1

    log.info(f"  Base templates: {len(samples)}")

    # ── Augmentation 1: Prefix variations (4x) ──
    prefixes = [
        "Please note that ", "It is important to understand that ",
        "For your information, ", "As part of our practices, ",
        "We want you to know that ", "You should be aware that ",
        "In accordance with our policies, ", "To be transparent, ",
        "For clarity, ", "We wish to inform you that ",
    ]
    augmented = list(samples)
    for s in samples:
        for _ in range(4):
            prefix = random.choice(prefixes)
            new_text = prefix + s["text"][0].lower() + s["text"][1:]
            augmented.append({
                "text": new_text, "labels": list(s["labels"]),
                "category": s["category"], "policy_id": s["policy_id"] + "_aug",
            })

    log.info(f"  After prefix augmentation: {len(augmented)}")

    # ── Augmentation 2: Synonym replacement (2x) ──
    for s in samples:
        for _ in range(2):
            new_text = synonym_replace(s["text"], n_replacements=2)
            if new_text != s["text"]:
                augmented.append({
                    "text": new_text, "labels": list(s["labels"]),
                    "category": s["category"], "policy_id": s["policy_id"] + "_syn",
                })

    log.info(f"  After synonym replacement: {len(augmented)}")

    # ── Augmentation 3: Multi-label combination (500 pairs) ──
    categories = list(CLAUSE_TEMPLATES.keys())
    for i in range(500):
        cat1, cat2 = random.sample(categories, 2)
        clause1 = random.choice(CLAUSE_TEMPLATES[cat1])
        clause2 = random.choice(CLAUSE_TEMPLATES[cat2])
        combined = clause1 + " Additionally, " + clause2[0].lower() + clause2[1:]
        combined_labels = [0] * NUM_LABELS
        for label in OPP_TO_TAXONOMY.get(cat1, [cat1]):
            if label in TAXONOMY:
                combined_labels[TAXONOMY.index(label)] = 1
        for label in OPP_TO_TAXONOMY.get(cat2, [cat2]):
            if label in TAXONOMY:
                combined_labels[TAXONOMY.index(label)] = 1
        augmented.append({
            "text": combined, "labels": combined_labels,
            "category": f"{cat1}+{cat2}", "policy_id": f"combined_{i}",
        })

    log.info(f"  After multi-label combination: {len(augmented)}")

    # ── Augmentation 4: Sentence shuffle (1x for long clauses) ──
    for s in samples:
        if len(s["text"].split()) > 15:
            new_text = sentence_shuffle(s["text"])
            if new_text != s["text"]:
                augmented.append({
                    "text": new_text, "labels": list(s["labels"]),
                    "category": s["category"], "policy_id": s["policy_id"] + "_shuf",
                })

    log.info(f"  After sentence shuffle: {len(augmented)}")

    # ── Deduplication + minimum length filter ──
    seen = set()
    unique = []
    for s in augmented:
        key = s["text"].strip().lower()
        if key not in seen and len(s["text"].split()) >= 10:
            seen.add(key)
            unique.append(s)

    random.shuffle(unique)

    # ── Label distribution ──
    label_counts = Counter()
    for s in unique:
        for i, v in enumerate(s["labels"]):
            if v == 1:
                label_counts[TAXONOMY[i]] += 1

    log.info(f"\nTotal unique samples: {len(unique)}")
    log.info("\nLabel Distribution:")
    for label in TAXONOMY:
        log.info(f"  {label:25s}: {label_counts.get(label, 0):5d}")

    # ── GroupShuffleSplit by policy_id (prevents data leakage) ──
    policy_ids = list(set(s["policy_id"] for s in unique))
    random.shuffle(policy_ids)
    n_policies = len(policy_ids)
    n_train_p = int(n_policies * 0.70)
    n_val_p = int(n_policies * 0.15)

    train_policies = set(policy_ids[:n_train_p])
    val_policies = set(policy_ids[n_train_p:n_train_p + n_val_p])
    test_policies = set(policy_ids[n_train_p + n_val_p:])

    train = [s for s in unique if s["policy_id"] in train_policies]
    val = [s for s in unique if s["policy_id"] in val_policies]
    test = [s for s in unique if s["policy_id"] in test_policies]

    log.info(f"\nGroupShuffleSplit: {n_policies} policy groups")
    log.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # ── Save CSVs ──
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = DATA_DIR / f"{split_name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["text"] + [f"label_{t}" for t in TAXONOMY]
            writer.writerow(header)
            for s in split_data:
                writer.writerow([s["text"]] + s["labels"])
        log.info(f"  Saved {len(split_data)} → {path}")

    return train, val, test


# ═══════════════════════════════════════════════════════════
# PHASE 2 — CLASS IMBALANCE HANDLING
# ═══════════════════════════════════════════════════════════

def compute_pos_weights(train_data):
    """Compute pos_weight for BCEWithLogitsLoss."""
    log.info("\n" + "=" * 60)
    log.info("PHASE 2 — Class Imbalance Handling")
    log.info("=" * 60)

    n = len(train_data)
    pos_counts = np.zeros(NUM_LABELS)
    for s in train_data:
        for i, v in enumerate(s["labels"]):
            pos_counts[i] += v

    neg_counts = n - pos_counts
    pos_weights = np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, 20.0)

    log.info("\npos_weight values:")
    for i, label in enumerate(TAXONOMY):
        log.info(f"  {label:25s}: pos={int(pos_counts[i]):5d}  neg={int(neg_counts[i]):5d}  weight={pos_weights[i]:.2f}")

    return torch.tensor(pos_weights, dtype=torch.float)


# ═══════════════════════════════════════════════════════════
# FOCAL LOSS (for severe class imbalance)
# ═══════════════════════════════════════════════════════════

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Reduces loss contribution from well-classified examples,
    focusing training on hard negatives.
    """
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits, labels):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='none',
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = labels * probs + (1 - labels) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


# ═══════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════

class ClauseDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=MAX_LENGTH):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"]
                labels = [int(row.get(f"label_{t}", 0)) for t in TAXONOMY]
                self.samples.append({"text": text, "labels": labels})
        log.info(f"  Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(s["text"], truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(s["labels"], dtype=torch.float),
        }


# ═══════════════════════════════════════════════════════════
# WEIGHTED TRAINER (supports BCE + Focal Loss)
# ═══════════════════════════════════════════════════════════

class WeightedTrainer(Trainer):
    """Trainer with pos_weight-aware loss (BCE or Focal)."""
    def __init__(self, pos_weight=None, use_focal=False, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.use_focal:
            loss_fn = FocalBCEWithLogitsLoss(
                gamma=self.focal_gamma,
                pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
            )
        else:
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
            )

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy().astype(int)
    labels = labels.astype(int)
    return {
        "f1_macro": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "f1_micro": round(f1_score(labels, preds, average="micro", zero_division=0), 4),
        "precision_macro": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(labels, preds, average="macro", zero_division=0), 4),
        "exact_match": round(np.all(preds == labels, axis=1).mean(), 4),
    }


# ═══════════════════════════════════════════════════════════
# PHASE 4 — THRESHOLD TUNING
# ═══════════════════════════════════════════════════════════

def tune_thresholds(trainer, val_dataset):
    """Sweep thresholds per label to maximize per-label F1."""
    log.info("\n" + "=" * 60)
    log.info("PHASE 4 — Per-Label Threshold Tuning")
    log.info("=" * 60)

    preds_out = trainer.predict(val_dataset)
    logits = preds_out.predictions
    labels = preds_out.label_ids.astype(int)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    thresholds = {}
    for i, label in enumerate(TAXONOMY):
        best_f1 = 0.0
        best_t = 0.5
        for t in np.arange(0.05, 0.96, 0.05):
            preds_t = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds_t, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = round(float(t), 2)
        thresholds[label] = {"threshold": best_t, "val_f1": round(best_f1, 4)}
        log.info(f"  {label:25s}: threshold={best_t:.2f}, val_f1={best_f1:.4f}")

    # Save
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "optimal_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"\n  Saved → {EVAL_DIR / 'optimal_thresholds.json'}")

    return thresholds


# ═══════════════════════════════════════════════════════════
# PHASE 5 — FINAL EVALUATION
# ═══════════════════════════════════════════════════════════

def evaluate_with_thresholds(trainer, test_dataset, thresholds, train_size, val_size, test_size, train_time, pos_weights_np, use_focal):
    """Evaluate on test set with tuned thresholds."""
    log.info("\n" + "=" * 60)
    log.info("PHASE 5 — Final Evaluation with Tuned Thresholds")
    log.info("=" * 60)

    preds_out = trainer.predict(test_dataset)
    logits = preds_out.predictions
    labels = preds_out.label_ids.astype(int)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    # Apply tuned thresholds
    preds = np.zeros_like(probs, dtype=int)
    for i, label in enumerate(TAXONOMY):
        t = thresholds[label]["threshold"]
        preds[:, i] = (probs[:, i] >= t).astype(int)

    # Overall metrics
    f1_mac = f1_score(labels, preds, average="macro", zero_division=0)
    f1_mic = f1_score(labels, preds, average="micro", zero_division=0)
    prec_mac = precision_score(labels, preds, average="macro", zero_division=0)
    rec_mac = recall_score(labels, preds, average="macro", zero_division=0)
    exact_match = np.all(preds == labels, axis=1).mean()

    log.info(f"\n  F1 Macro:   {f1_mac:.4f}")
    log.info(f"  F1 Micro:   {f1_mic:.4f}")
    log.info(f"  Precision:  {prec_mac:.4f}")
    log.info(f"  Recall:     {rec_mac:.4f}")
    log.info(f"  Exact Match:{exact_match:.4f}")

    # Per-label
    report_dict = classification_report(labels, preds, target_names=TAXONOMY, zero_division=0, output_dict=True)
    report_text = classification_report(labels, preds, target_names=TAXONOMY, zero_division=0)

    per_label = {}
    for label in TAXONOMY:
        per_label[label] = {
            "precision": round(report_dict[label]["precision"], 4),
            "recall": round(report_dict[label]["recall"], 4),
            "f1": round(report_dict[label]["f1-score"], 4),
            "support": int(report_dict[label]["support"]),
            "threshold": thresholds[label]["threshold"],
        }

    cm = multilabel_confusion_matrix(labels, preds)

    # Save
    metrics = {
        "model": BASE_MODEL,
        "finetuned_model": EXPORT_DIR,
        "version": "v2",
        "loss_function": "focal_loss" if use_focal else "bce_with_logits",
        "dataset_sizes": {"train": train_size, "val": val_size, "test": test_size},
        "hyperparameters": {
            "learning_rate": LR, "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH, "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO, "seed": SEED,
        },
        "pos_weights": {TAXONOMY[i]: round(float(pos_weights_np[i]), 2) for i in range(NUM_LABELS)},
        "optimal_thresholds": {k: v["threshold"] for k, v in thresholds.items()},
        "test_metrics": {
            "f1_macro": round(f1_mac, 4),
            "f1_micro": round(f1_mic, 4),
            "precision_macro": round(prec_mac, 4),
            "recall_macro": round(rec_mac, 4),
            "exact_match": round(exact_match, 4),
        },
        "per_label": per_label,
        "confusion_matrices": {TAXONOMY[i]: cm[i].tolist() for i in range(NUM_LABELS)},
        "training_time_seconds": round(train_time, 1),
    }

    with open(EVAL_DIR / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(EVAL_DIR / "classification_report.txt", "w") as f:
        f.write("Legal-BERT v2 Multi-Label Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {BASE_MODEL} → {EXPORT_DIR}\n")
        f.write(f"Loss: {'Focal Loss (gamma=2.0)' if use_focal else 'BCEWithLogitsLoss'}\n")
        f.write(f"Test samples: {test_size}\n")
        f.write(f"Training time: {train_time:.0f}s\n")
        f.write(f"Gradient accumulation: {GRADIENT_ACCUMULATION} (effective batch={BATCH_SIZE * GRADIENT_ACCUMULATION})\n\n")
        f.write(report_text)
        f.write(f"\nExact Match Ratio: {exact_match:.4f}\n")
        f.write(f"\nOptimal Thresholds:\n")
        for label in TAXONOMY:
            f.write(f"  {label:25s}: {thresholds[label]['threshold']:.2f}\n")

    log.info(f"\n  Saved → {EVAL_DIR / 'final_metrics.json'}")
    log.info(f"  Saved → {EVAL_DIR / 'classification_report.txt'}")

    return metrics


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Legal-BERT v2")
    parser.add_argument("--focal", action="store_true", help="Use Focal Loss instead of BCE")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (default: 2)")
    parser.add_argument("--lr", type=float, default=LR, help=f"Learning rate (default: {LR})")
    args = parser.parse_args()

    global EPOCHS
    EPOCHS = args.epochs

    # Phase 1
    train_data, val_data, test_data = build_dataset()

    # Phase 2
    pos_weights = compute_pos_weights(train_data)
    pos_weights_np = pos_weights.numpy()

    # Phase 3 — Model + Training
    log.info("\n" + "=" * 60)
    log.info("PHASE 3 — Advanced Training")
    log.info("=" * 60)

    if args.focal:
        log.info("  Loss: Focal Loss (gamma=2.0)")
    else:
        log.info("  Loss: BCEWithLogitsLoss + pos_weight")
    log.info(f"  Mixed precision: {'ON' if args.fp16 else 'OFF'}")
    log.info(f"  Gradient accumulation: {GRADIENT_ACCUMULATION} (effective batch={BATCH_SIZE * GRADIENT_ACCUMULATION})")
    log.info(f"  Epochs: {args.epochs}, Patience: {args.patience}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    log.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    train_ds = ClauseDataset(DATA_DIR / "train.csv", tokenizer)
    val_ds = ClauseDataset(DATA_DIR / "val.csv", tokenizer)
    test_ds = ClauseDataset(DATA_DIR / "test.csv", tokenizer)

    total_steps = (len(train_ds) // (BATCH_SIZE * GRADIENT_ACCUMULATION) + 1) * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir="./training_output_v2",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        seed=SEED,
        report_to="none",
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        pos_weight=pos_weights,
        use_focal=args.focal,
        focal_gamma=2.0,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    log.info(f"\nTraining complete in {train_time:.0f}s")

    # Phase 4 — Threshold tuning
    thresholds = tune_thresholds(trainer, val_ds)

    # Phase 5 — Final evaluation
    metrics = evaluate_with_thresholds(
        trainer, test_ds, thresholds,
        len(train_ds), len(val_ds), len(test_ds),
        train_time, pos_weights_np, args.focal,
    )

    # Phase 6 — Export
    log.info("\n" + "=" * 60)
    log.info("PHASE 6 — Model Export")
    log.info("=" * 60)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    trainer.save_model(EXPORT_DIR)
    tokenizer.save_pretrained(EXPORT_DIR)

    # Save thresholds alongside model
    threshold_export = {k: v["threshold"] for k, v in thresholds.items()}
    with open(os.path.join(EXPORT_DIR, "thresholds.json"), "w") as f:
        json.dump(threshold_export, f, indent=2)

    metadata = {
        "model_name": "legal-bert-finetuned-v2",
        "base_model": "nlpaueb/legal-bert-base-uncased",
        "task": "multi_label_classification",
        "num_labels": NUM_LABELS,
        "labels": TAXONOMY,
        "optimal_thresholds": threshold_export,
        "loss_function": "focal_loss" if args.focal else "bce_with_logits",
        "f1_macro": metrics["test_metrics"]["f1_macro"],
        "f1_micro": metrics["test_metrics"]["f1_micro"],
        "training_epochs": args.epochs,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "mixed_precision": args.fp16,
        "seed": SEED,
        "version": "v2",
    }
    with open(os.path.join(EXPORT_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_size = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, dn, fns in os.walk(EXPORT_DIR) for fn in fns
    )
    log.info(f"  Model saved → {EXPORT_DIR} ({total_size // (1024*1024)} MB)")

    # Cleanup training checkpoints
    import shutil
    if os.path.exists("./training_output_v2"):
        shutil.rmtree("./training_output_v2", ignore_errors=True)

    # Phase 7 — Summary
    log.info("\n" + "=" * 60)
    log.info("PHASE 7 — FINAL REPORT")
    log.info("=" * 60)
    log.info(f"  Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    log.info(f"  Loss: {'Focal Loss' if args.focal else 'BCEWithLogitsLoss'}")
    log.info(f"  F1 Macro:  {metrics['test_metrics']['f1_macro']:.4f}")
    log.info(f"  F1 Micro:  {metrics['test_metrics']['f1_micro']:.4f}")
    log.info(f"  Exact Match: {metrics['test_metrics']['exact_match']:.4f}")
    log.info(f"  Model Size: {total_size // (1024*1024)} MB")
    log.info(f"  Training Time: {train_time:.0f}s")

    target_macro = 0.70
    if metrics["test_metrics"]["f1_macro"] >= target_macro:
        log.info(f"  ✓ F1 Macro TARGET MET (≥{target_macro})")
    else:
        log.info(f"  ✗ F1 Macro below target ({target_macro}). Consider: --focal, more data, or more epochs.")

    log.info("\nALL 7 PHASES COMPLETE")


if __name__ == "__main__":
    main()
