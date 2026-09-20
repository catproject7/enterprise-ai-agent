"""File loaders and the document ingestion entry point."""

from abc import ABC, abstractmethod
from pathlib import Path

from pypdf import PdfReader

from .document import Document, DocumentMetadata, DocumentType
from .exceptions import (
    DocumentIngestionError,
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)


class BaseLoader(ABC):
    """Contract for loading one supported document format."""

    file_type: DocumentType
    extensions: frozenset[str]

    @abstractmethod
    def load(self, path: Path) -> Document:
        """Load and normalize a document from disk."""


def _build_metadata(path: Path, file_type: DocumentType) -> DocumentMetadata:
    return DocumentMetadata(
        source=path.expanduser().resolve().as_posix(),
        file_name=path.name,
        file_type=file_type,
        file_size=path.stat().st_size,
    )


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParsingError(f"Unable to decode text file: {path}") from exc
    except OSError as exc:
        raise DocumentParsingError(f"Unable to read text file: {path}") from exc


def _require_non_empty_content(content: str, path: Path) -> None:
    if not content.strip():
        raise EmptyDocumentError(f"No extractable content in document: {path}")


class _TextLoaderBase(BaseLoader):
    def load(self, path: Path) -> Document:
        content = _read_text_file(path)
        _require_non_empty_content(content, path)
        return Document(
            content=content,
            metadata=_build_metadata(path, self.file_type),
        )


class TextLoader(_TextLoaderBase):
    """Loader for UTF-8 and UTF-8 BOM text files."""

    file_type = DocumentType.TEXT
    extensions = frozenset({".txt"})


class MarkdownLoader(_TextLoaderBase):
    """Loader that preserves Markdown source text."""

    file_type = DocumentType.MARKDOWN
    extensions = frozenset({".md", ".markdown"})


class PdfLoader(BaseLoader):
    """Loader for text-based PDFs using pypdf."""

    file_type = DocumentType.PDF
    extensions = frozenset({".pdf"})

    def load(self, path: Path) -> Document:
        metadata = _build_metadata(path, self.file_type)

        try:
            reader = PdfReader(str(path))
            pages = (page.extract_text() or "" for page in reader.pages)
            content = "\n\n".join(pages).strip()
        except Exception as exc:
            raise DocumentParsingError(f"Unable to parse PDF: {path}") from exc

        _require_non_empty_content(content, path)
        return Document(content=content, metadata=metadata)


_LOADERS: dict[str, BaseLoader] = {
    extension: loader
    for loader in (PdfLoader(), MarkdownLoader(), TextLoader())
    for extension in loader.extensions
}


def load_document(path: str | Path) -> Document:
    """Load a supported file into a unified Document."""

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(source)

    if not source.is_file():
        raise DocumentIngestionError(f"Document source is not a file: {source}")

    suffix = source.suffix.lower()

    try:
        loader = _LOADERS[suffix]
    except KeyError as exc:
        supported = ", ".join(sorted(_LOADERS))
        raise UnsupportedFileTypeError(
            f"Unsupported file extension: {suffix or '<none>'}; supported: {supported}"
        ) from exc

    return loader.load(source)
