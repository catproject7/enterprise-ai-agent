"""Exceptions raised during document ingestion."""


class DocumentIngestionError(Exception):
    """Base exception for document ingestion failures."""


class UnsupportedFileTypeError(DocumentIngestionError):
    """Raised when a file extension has no registered loader."""


class EmptyDocumentError(DocumentIngestionError):
    """Raised when a file can be read but contains no extractable content."""


class DocumentParsingError(DocumentIngestionError):
    """Raised when a supported file cannot be decoded or parsed."""
