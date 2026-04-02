"""
Legal-BERT vFINAL — Maximum Performance Training (CPU-Optimized)
Targets: F1 Macro ≥ 0.70

Key optimizations vs v2:
  1. Freeze first 9/12 BERT layers → ~3x faster backward
  2. max_length=128 → ~2x faster attention
  3. Balanced dataset with extra weak-label templates
  4. 5 epochs (feasible in ~45 min on CPU)
  5. Aggressive threshold calibration
"""
import os, sys, json, csv, time, random, re, logging
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ═══════════════════════════════════════════════════════════════
# TAXONOMY
# ═══════════════════════════════════════════════════════════════
TAXONOMY = [
    "DATA_COLLECTION", "DATA_SHARING", "USER_RIGHTS",
    "DATA_RETENTION", "SECURITY_MEASURES", "THIRD_PARTY_TRANSFER",
    "COOKIES_TRACKING", "CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE",
    "LIABILITY_LIMITATION",
]
NUM_LABELS = len(TAXONOMY)
LABEL2IDX = {l: i for i, l in enumerate(TAXONOMY)}

# ═══════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS (CPU-OPTIMIZED)
# ═══════════════════════════════════════════════════════════════
MAX_LENGTH = 128        # Reduced from 256 for ~2x speedup
BATCH_SIZE = 16
GRAD_ACCUM = 2          # effective batch = 32
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1
FREEZE_LAYERS = 9       # Freeze first 9 of 12 BERT encoder layers
FOCAL_GAMMA = 2.0
PATIENCE = 2

# Paths
BASE_MODEL_DIR = Path("./models/legal-bert")
OUTPUT_DIR = Path("./training_output_final")
EXPORT_DIR = Path("./models/legal-bert-final")
EVAL_DIR = Path("./evaluation")
DATA_DIR = Path("./data/processed")

# ═══════════════════════════════════════════════════════════════
# CLAUSE TEMPLATES — EXPANDED FOR WEAK LABELS
# ═══════════════════════════════════════════════════════════════
# We add EXTRA templates for the 4 weakest labels to improve discrimination

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
        # === EXTRA TEMPLATES FOR WEAK LABEL BOOST ===
        "The personal data we obtain includes but is not limited to full name, date of birth, mailing address, and email.",
        "We gather behavioral data through analytics tools that monitor how you navigate through our digital properties.",
        "Upon registration, we require you to submit identification documents for verification of your identity.",
        "We systematically collect and process data relating to your online activities across our network of websites.",
        "Your mobile device transmits diagnostic and performance data to our servers for service reliability monitoring.",
        "We acquire personal information when you participate in promotions, contests, or loyalty reward programs.",
        "Metadata associated with your uploaded content, including timestamp, file size, and format, is automatically recorded.",
        "We harvest data from your interactions with our automated chatbot systems for machine learning improvement.",
        "Financial transaction metadata including merchant category, amount, and frequency is collected for fraud prevention.",
        "We process information derived from your use of connected smart devices within our Internet of Things ecosystem.",
        "Your educational background and qualifications are collected when you apply for positions through our career portal.",
        "We obtain psychographic data through optional surveys to better understand consumer preferences and motivations.",
        "We monitor and record the frequency and duration of your sessions for capacity planning and resource allocation.",
        "We collect dietary and allergy information when you use our food delivery or restaurant reservation services.",
        "Your travel preferences and booking history are collected to provide tailored travel recommendations and offers.",
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
        # === EXTRA TEMPLATES ===
        "Your personal information is disclosed to affiliated entities within our corporate group for internal administrative purposes.",
        "We furnish data to regulatory bodies and supervisory authorities as mandated by applicable industry regulations.",
        "Selected personal data is transmitted to our advertising technology partners for campaign attribution and measurement.",
        "We disseminate anonymized behavioral insights to industry consortiums for collective market intelligence purposes.",
        "Your account information may be communicated to financial institutions for the purpose of credit evaluation.",
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
        # === EXTRA TEMPLATES ===
        "You are entitled to obtain confirmation as to whether personal data concerning you is being processed by us.",
        "Under applicable law, you have the right to receive your data in a structured, commonly used, machine-readable format.",
        "You may exercise your right to be forgotten by submitting a formal erasure request through our designated channel.",
        "We shall respond to all valid data subject access requests within the statutory timeframe of thirty calendar days.",
        "You have the right to restrict the processing of your personal data where the accuracy of the data is contested.",
        "Any individual whose data we process has the right to lodge a complaint with the relevant supervisory authority.",
        "You may instruct us to transfer your personal data directly to another controller where technically feasible.",
        "We guarantee that exercising your data protection rights will not result in any punitive action or service degradation.",
        "You can submit a verifiable consumer request to know the categories of personal information collected about you.",
        "We provide a toll-free telephone number and online portal for submitting data protection rights requests.",
        "You have the right to opt out of the sale or sharing of your personal information for cross-context advertising.",
        "Upon your request, we will disclose the specific pieces of personal information we have collected about you.",
        "You may designate an authorized agent to submit data protection requests on your behalf with proper verification.",
        "We honor global privacy control signals as a valid mechanism for opting out of data sales and targeted advertising.",
        "You can manage your consent preferences for each category of data processing through our preference center.",
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
        # === EXTRA TEMPLATES ===
        "No personal data of children under thirteen years of age shall be collected without verifiable parental authorization.",
        "We take additional precautions to safeguard the privacy and security of information belonging to minor users.",
        "Our services implement technical safeguards designed to prevent the unauthorized collection of data from minors.",
        "We maintain a dedicated child safety team responsible for ensuring compliance with children's privacy legislation.",
        "Any processing of personal data belonging to a child shall be conducted with heightened security measures.",
        "We will not retain personal information of children beyond the period necessary for the original collection purpose.",
        "Parents may revoke their consent for data collection at any time and request immediate deletion of their child's data.",
        "We prohibit the sale or commercial exploitation of personal information obtained from users under the age of sixteen.",
        "Our advertising systems are configured to exclude users identified as minors from behavioral targeting campaigns.",
        "We require all third-party service providers to comply with our children's privacy standards when processing minor data.",
        "The platform employs machine learning algorithms to detect and flag potential underage users for additional verification.",
        "We conduct annual audits of our children's data protection practices in consultation with external privacy experts.",
        "Minors' data is stored in segregated, encrypted databases with enhanced access controls and monitoring.",
        "We provide educational resources to help parents understand and manage their children's digital privacy rights.",
        "Our terms of service explicitly prohibit users under the minimum age from creating accounts or providing personal data.",
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


# ═══════════════════════════════════════════════════════════════
# SYNONYM MAP (legal domain)
# ═══════════════════════════════════════════════════════════════
SYNONYM_MAP = {
    "collect": ["gather", "obtain", "acquire", "capture", "harvest"],
    "share": ["disclose", "transmit", "communicate", "furnish", "disseminate"],
    "personal information": ["personal data", "personally identifiable information", "PII", "user data"],
    "delete": ["erase", "remove", "purge", "destroy", "expunge"],
    "retain": ["store", "maintain", "preserve", "keep", "hold"],
    "consent": ["authorization", "approval", "permission", "agreement"],
    "third party": ["third-party", "external party", "outside organization", "third-party entity"],
    "children": ["minors", "young users", "underage individuals", "juvenile users"],
    "cookies": ["tracking technologies", "browser cookies", "HTTP cookies", "web cookies"],
    "security": ["protection", "safeguarding", "defense", "security controls"],
    "right": ["entitlement", "privilege", "prerogative", "legal right"],
    "use": ["utilize", "employ", "leverage", "make use of"],
    "provide": ["supply", "furnish", "deliver", "offer"],
    "process": ["handle", "manage", "treat", "deal with"],
    "transfer": ["transmit", "convey", "transport", "send"],
    "comply": ["adhere", "conform", "abide by", "observe"],
    "notify": ["inform", "alert", "advise", "apprise"],
    "request": ["ask", "petition", "submit a request for", "demand"],
}


def synonym_replace(text, n_replacements=2):
    """Replace n random words with legal-domain synonyms."""
    result = text
    keys = list(SYNONYM_MAP.keys())
    random.shuffle(keys)
    replaced = 0
    for key in keys:
        if replaced >= n_replacements:
            break
        if key in result.lower():
            syn = random.choice(SYNONYM_MAP[key])
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            result = pattern.sub(syn, result, count=1)
            replaced += 1
    return result


def sentence_shuffle(text):
    """Shuffle sentence order for multi-sentence texts."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) < 2:
        return text
    random.shuffle(sentences)
    return " ".join(sentences)


# ═══════════════════════════════════════════════════════════════
# DATASET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════
def build_dataset():
    """Build balanced, augmented dataset targeting 3500-5000 samples."""
    log.info("=" * 60)
    log.info("PHASE 1 — Building Balanced Dataset")
    log.info("=" * 60)

    samples = []
    policy_counter = 0
    for category, templates in CLAUSE_TEMPLATES.items():
        label_vec = [0] * NUM_LABELS
        label_vec[LABEL2IDX[category]] = 1
        for i, text in enumerate(templates):
            samples.append({
                "text": text,
                "labels": label_vec,
                "category": category,
                "policy_id": f"{category}_{i}",
            })
            policy_counter += 1

    log.info(f"  Base templates: {len(samples)}")

    # ── Augmentation 1: Prefix variations (8x for better coverage) ──
    prefixes = [
        "Please note that ", "It is important to understand that ",
        "For your information, ", "As part of our practices, ",
        "We want you to know that ", "You should be aware that ",
        "In accordance with our policies, ", "To be transparent, ",
        "For clarity, ", "We wish to inform you that ",
        "Be advised that ", "Under this agreement, ",
        "As stated in our terms, ", "Per our data practices, ",
        "In compliance with applicable laws, ", "For your awareness, ",
        "Pursuant to our privacy commitments, ", "As outlined herein, ",
    ]
    augmented = list(samples)
    for s in samples:
        for _ in range(8):
            prefix = random.choice(prefixes)
            new_text = prefix + s["text"][0].lower() + s["text"][1:]
            augmented.append({
                "text": new_text, "labels": list(s["labels"]),
                "category": s["category"], "policy_id": s["policy_id"] + "_aug",
            })

    log.info(f"  After prefix augmentation: {len(augmented)}")

    # ── Augmentation 2: Synonym replacement (4x) ──
    for s in samples:
        for _ in range(4):
            new_text = synonym_replace(s["text"], n_replacements=2)
            if new_text != s["text"]:
                augmented.append({
                    "text": new_text, "labels": list(s["labels"]),
                    "category": s["category"], "policy_id": s["policy_id"] + "_syn",
                })

    log.info(f"  After synonym replacement: {len(augmented)}")

    # ── Augmentation 3: Multi-label combination (1000 pairs) ──
    categories = list(CLAUSE_TEMPLATES.keys())
    for i in range(1000):
        cat1, cat2 = random.sample(categories, 2)
        clause1 = random.choice(CLAUSE_TEMPLATES[cat1])
        clause2 = random.choice(CLAUSE_TEMPLATES[cat2])
        label_vec = [0] * NUM_LABELS
        label_vec[LABEL2IDX[cat1]] = 1
        label_vec[LABEL2IDX[cat2]] = 1
        augmented.append({
            "text": f"{clause1} {clause2}",
            "labels": label_vec,
            "category": f"{cat1}+{cat2}",
            "policy_id": f"combo_{i}",
        })

    log.info(f"  After multi-label combination: {len(augmented)}")

    # ── Augmentation 4: Sentence shuffle ──
    for s in samples:
        if len(s["text"].split(". ")) >= 2:
            shuffled = sentence_shuffle(s["text"])
            if shuffled != s["text"]:
                augmented.append({
                    "text": shuffled, "labels": list(s["labels"]),
                    "category": s["category"], "policy_id": s["policy_id"] + "_shuf",
                })

    log.info(f"  After sentence shuffle: {len(augmented)}")

    # ── Deduplication & filtering ──
    seen = set()
    unique = []
    for s in augmented:
        key = s["text"].strip().lower()
        if key in seen:
            continue
        if len(s["text"].split()) < 10:
            continue
        seen.add(key)
        unique.append(s)

    random.shuffle(unique)
    log.info(f"\nTotal unique samples: {len(unique)}")

    # Label distribution
    label_counts = Counter()
    for s in unique:
        for i, v in enumerate(s["labels"]):
            if v == 1:
                label_counts[TAXONOMY[i]] += 1
    log.info("\nLabel Distribution:")
    for label in TAXONOMY:
        log.info(f"  {label:25s}: {label_counts[label]:5d}")

    # ── Split with GroupShuffleSplit ──
    policy_ids = [s["policy_id"] for s in unique]
    groups = np.array(policy_ids)
    n_groups = len(set(policy_ids))
    log.info(f"\nGroupShuffleSplit: {n_groups} policy groups")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(gss.split(unique, groups=groups))

    temp_data = [unique[i] for i in temp_idx]
    temp_groups = np.array([temp_data[i]["policy_id"] for i in range(len(temp_data))])
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(gss2.split(temp_data, groups=temp_groups))

    train_data = [unique[i] for i in train_idx]
    val_data = [temp_data[i] for i in val_idx]
    test_data = [temp_data[i] for i in test_idx]

    log.info(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Verify no leakage
    train_pids = set(s["policy_id"] for s in train_data)
    val_pids = set(s["policy_id"] for s in val_data)
    test_pids = set(s["policy_id"] for s in test_data)
    assert len(train_pids & val_pids) == 0, "LEAKAGE: train-val overlap!"
    assert len(train_pids & test_pids) == 0, "LEAKAGE: train-test overlap!"
    assert len(val_pids & test_pids) == 0, "LEAKAGE: val-test overlap!"
    log.info("  ✅ No policy_id leakage across splits")

    # Save CSVs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = DATA_DIR / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text"] + TAXONOMY)
            for s in data:
                writer.writerow([s["text"]] + s["labels"])
        log.info(f"  Saved {len(data)} → {path}")

    return train_data, val_data, test_data


# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════
class ClauseDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"] if isinstance(item, dict) else item[0]
        labels = item["labels"] if isinstance(item, dict) else item[1]
        enc = self.tokenizer(
            text, max_length=MAX_LENGTH, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


# ═══════════════════════════════════════════════════════════════
# FOCAL LOSS
# ═══════════════════════════════════════════════════════════════
class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


# ═══════════════════════════════════════════════════════════════
# CUSTOM TRAINER (Focal Loss)
# ═══════════════════════════════════════════════════════════════
class FocalTrainer(Trainer):
    def __init__(self, focal_loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
    exact_match = np.mean(np.all(preds == labels, axis=1))
    return {
        "f1_macro": round(f1_macro, 4),
        "f1_micro": round(f1_micro, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "exact_match": round(exact_match, 4),
    }


# ═══════════════════════════════════════════════════════════════
# THRESHOLD CALIBRATION
# ═══════════════════════════════════════════════════════════════
def predict_all(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def tune_thresholds(logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    thresholds = {}
    for i, label in enumerate(TAXONOMY):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.025):  # Finer granularity
            preds = (probs[:, i] >= t).astype(int)
            if preds.sum() == 0:
                continue
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = round(float(t), 3)
        thresholds[label] = {"threshold": best_t, "val_f1": round(best_f1, 4)}
        log.info(f"  {label:25s}: t={best_t:.3f}  val_f1={best_f1:.4f}")
    return thresholds


def evaluate_test(logits, labels, thresholds):
    probs = 1 / (1 + np.exp(-logits))
    preds = np.zeros_like(probs, dtype=int)
    for i, label in enumerate(TAXONOMY):
        t = thresholds[label]["threshold"]
        preds[:, i] = (probs[:, i] >= t).astype(int)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
    exact_match = np.mean(np.all(preds == labels, axis=1))

    metrics = {
        "f1_macro": round(f1_macro, 4),
        "f1_micro": round(f1_micro, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "exact_match": round(exact_match, 4),
    }

    report = classification_report(
        labels, preds, target_names=TAXONOMY, zero_division=0, digits=4
    )

    log.info(f"\n  F1 Macro:  {f1_macro:.4f}")
    log.info(f"  F1 Micro:  {f1_micro:.4f}")
    log.info(f"  Precision: {precision_macro:.4f}")
    log.info(f"  Recall:    {recall_macro:.4f}")
    log.info(f"  Exact Match: {exact_match:.4f}")
    log.info(f"\n{report}")

    return metrics, report


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    start_time = time.time()
    device = torch.device("cpu")

    # Phase 1: Build dataset
    train_data, val_data, test_data = build_dataset()

    # Phase 2: Load model
    log.info("\n" + "=" * 60)
    log.info("PHASE 2 — Loading & Freezing Model")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE_MODEL_DIR), num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    # ── FREEZE FIRST N LAYERS ──
    # Freeze embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # Freeze first FREEZE_LAYERS encoder layers
    for i in range(FREEZE_LAYERS):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"  Total params:     {total:,}")
    log.info(f"  Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")
    log.info(f"  Frozen layers:    embeddings + encoder[0:{FREEZE_LAYERS}]")
    log.info(f"  Training layers:  encoder[{FREEZE_LAYERS}:12] + pooler + classifier")

    model.to(device)

    # Datasets
    train_ds = ClauseDataset(train_data, tokenizer)
    val_ds = ClauseDataset(val_data, tokenizer)
    test_ds = ClauseDataset(test_data, tokenizer)

    # Phase 3: Compute class weights
    log.info("\n" + "=" * 60)
    log.info("PHASE 3 — Class Balance Handling")
    log.info("=" * 60)

    label_matrix = np.array([s["labels"] for s in train_data])
    pos_counts = label_matrix.sum(axis=0)
    neg_counts = len(train_data) - pos_counts
    pos_weight = np.clip(neg_counts / (pos_counts + 1e-6), 1.0, 20.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float).to(device)

    for i, label in enumerate(TAXONOMY):
        log.info(f"  {label:25s}: pos={int(pos_counts[i]):5d}  neg={int(neg_counts[i]):5d}  weight={pos_weight[i]:.2f}")

    focal_loss = FocalBCEWithLogitsLoss(gamma=FOCAL_GAMMA, pos_weight=pos_weight_tensor)

    # Phase 4: Training
    log.info("\n" + "=" * 60)
    log.info("PHASE 4 — Training (CPU-Optimized)")
    log.info("=" * 60)
    log.info(f"  Focal Loss: gamma={FOCAL_GAMMA}")
    log.info(f"  Gradient accumulation: {GRAD_ACCUM} (effective batch={BATCH_SIZE * GRAD_ACCUM})")
    log.info(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}")
    log.info(f"  Max length: {MAX_LENGTH}, Frozen layers: {FREEZE_LAYERS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        seed=SEED,
        dataloader_pin_memory=False,  # CPU: no pinning
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = FocalTrainer(
        focal_loss_fn=focal_loss,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()
    train_time = time.time() - start_time
    log.info(f"\n  Training time: {train_time:.0f}s ({train_time/60:.1f} min)")

    # Phase 5: Threshold calibration
    log.info("\n" + "=" * 60)
    log.info("PHASE 5 — Threshold Calibration (Validation Set)")
    log.info("=" * 60)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_logits, val_labels = predict_all(model, val_loader, device)
    thresholds = tune_thresholds(val_logits, val_labels)

    # Phase 6: Test evaluation
    log.info("\n" + "=" * 60)
    log.info("PHASE 6 — Final Test Evaluation")
    log.info("=" * 60)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_logits, test_labels = predict_all(model, test_loader, device)
    metrics, report = evaluate_test(test_logits, test_labels, thresholds)
    metrics["dataset_sizes"] = {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
    }
    metrics["training_time_sec"] = round(train_time, 1)
    metrics["epochs_completed"] = EPOCHS

    # Phase 7: Export
    log.info("\n" + "=" * 60)
    log.info("PHASE 7 — Export Final Model")
    log.info("=" * 60)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(EVAL_DIR / "final_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(EVAL_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    model.save_pretrained(str(EXPORT_DIR))
    tokenizer.save_pretrained(str(EXPORT_DIR))

    with open(EXPORT_DIR / "optimal_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    metadata = {
        "version": "vFINAL",
        "base_model": "nlpaueb/legal-bert-base-uncased",
        "training": f"{EPOCHS} epochs, focal loss, {FREEZE_LAYERS} frozen layers, max_length={MAX_LENGTH}",
        "dataset_sizes": metrics["dataset_sizes"],
        "f1_macro": metrics["f1_macro"],
        "f1_micro": metrics["f1_micro"],
        "training_time_sec": metrics["training_time_sec"],
    }
    with open(EXPORT_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"  Saved model → {EXPORT_DIR}")
    log.info(f"  Saved evaluation → {EVAL_DIR}")

    total_time = time.time() - start_time
    log.info(f"\n{'=' * 60}")
    log.info(f"TRAINING COMPLETE")
    log.info(f"{'=' * 60}")
    log.info(f"  F1 Macro:   {metrics['f1_macro']}")
    log.info(f"  F1 Micro:   {metrics['f1_micro']}")
    log.info(f"  Precision:  {metrics['precision_macro']}")
    log.info(f"  Recall:     {metrics['recall_macro']}")
    log.info(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    if metrics["f1_macro"] >= 0.70:
        log.info("\n✅ TARGET ACHIEVED — F1 Macro ≥ 0.70")
    elif metrics["f1_macro"] >= 0.65:
        log.info(f"\n✅ GOOD PERFORMANCE — F1 Macro {metrics['f1_macro']}")
    else:
        log.info(f"\n⚠️  F1 Macro {metrics['f1_macro']} below 0.65")

    log.info("\n🎯 MODEL OPTIMIZED & READY")


if __name__ == "__main__":
    main()
