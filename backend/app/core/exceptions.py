"""
Custom exception hierarchy for all pipeline modules.
Module 1: Document Processing
Module 2: Classification
Module 3: Risk Scoring
"""


class PolicyAnalyzerError(Exception):
    """Base exception for all application errors."""
    def __init__(self, message: str, document_id: str = None, step: str = None):
        self.document_id = document_id
        self.step = step
        super().__init__(message)


# ─── Module 1: Document Processing ──────────────────────────

class DocumentProcessingError(PolicyAnalyzerError):
    """Raised when the overall document processing pipeline fails."""
    pass


class FileValidationError(PolicyAnalyzerError):
    """Raised when file validation fails (type, size, format)."""
    pass


class ExtractionError(PolicyAnalyzerError):
    """Raised when text extraction from a document fails."""
    pass


class ClauseSegmentationError(PolicyAnalyzerError):
    """Raised when clause splitting/segmentation fails."""
    pass


class NERError(PolicyAnalyzerError):
    """Raised when Named Entity Recognition fails."""
    pass


class EmbeddingError(PolicyAnalyzerError):
    """Raised when embedding generation or ChromaDB storage fails."""
    pass


# ─── Module 2: Classification ───────────────────────────────

class ClassificationError(PolicyAnalyzerError):
    """Raised when clause classification fails."""
    pass


class ModelLoadError(PolicyAnalyzerError):
    """Raised when an ML model fails to load."""
    pass


class InvalidDocumentStateError(PolicyAnalyzerError):
    """Raised when a document is not in the required state for an operation."""
    def __init__(self, message: str, document_id: str = None,
                 current_status: str = None, required_status: str = None):
        self.current_status = current_status
        self.required_status = required_status
        super().__init__(message, document_id=document_id, step="state_check")


# ─── Module 3: Risk Scoring ─────────────────────────────────

class RiskComputationError(PolicyAnalyzerError):
    """Raised when risk score computation fails."""
    pass


class MissingClassificationError(PolicyAnalyzerError):
    """Raised when risk scoring is attempted without classification data."""
    def __init__(self, message: str, document_id: str = None):
        super().__init__(message, document_id=document_id, step="risk_prereq")


# ─── Module 4: Explainability ───────────────────────────────

class ExplainabilityError(PolicyAnalyzerError):
    """Raised when explanation generation fails."""
    pass


# ─── Module 5: Compliance ──────────────────────────────────

class ComplianceError(PolicyAnalyzerError):
    """Raised when compliance evaluation fails."""
    pass


# ─── General ────────────────────────────────────────────────

class SummarizationError(PolicyAnalyzerError):
    """Raised when document summarization fails."""
    pass
