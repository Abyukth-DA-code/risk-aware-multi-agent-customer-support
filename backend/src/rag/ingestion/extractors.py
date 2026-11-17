"""
Multi-Format Document Text Extractors (PDF, DOCX, TXT, CSV)
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
"""

from abc import ABC, abstractmethod
import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TextExtractionError(ValueError):
    """Raised when document extraction fails due to formatting, corruption, or empty content."""
    pass


class ExtractedDocument:
    """Represents successfully extracted text with structured page/section metadata."""

    def __init__(self, filename: str, full_text: str, pages: List[Dict[str, Any]]):
        self.filename = filename
        self.full_text = full_text
        self.pages = pages  # list of {"page_number": int, "text": str}
        self.char_count = len(full_text)
        self.word_count = len(full_text.split())

    def validate_quality(self, min_chars: int = 50) -> None:
        """
        Enforces extraction quality:
        - Must meet minimum character length.
        - Must contain alphanumeric content (not just whitespace or random control characters).
        """
        clean = self.full_text.strip()
        if not clean:
            raise TextExtractionError(f"File '{self.filename}' is completely empty after extraction.")

        if len(clean) < min_chars:
            raise TextExtractionError(
                f"File '{self.filename}' content too short ({len(clean)} characters). "
                f"Minimum required is {min_chars} characters of substantive policy text."
            )

        alnum_count = sum(1 for c in clean if c.isalnum())
        if alnum_count / len(clean) < 0.25:
            raise TextExtractionError(
                f"File '{self.filename}' contains insufficient substantive text "
                f"(mostly whitespace or special symbols)."
            )


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        pass


class PDFExtractor(BaseExtractor):
    def extract(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        try:
            import pypdf
        except ImportError as exc:
            raise RuntimeError("pypdf is not installed. Run 'pip install pypdf'.") from exc

        try:
            stream = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(stream)

            if reader.is_encrypted:
                raise TextExtractionError(f"PDF file '{filename}' is password-protected or encrypted.")

            pages = []
            full_texts = []
            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                pages.append({"page_number": idx, "text": page_text})
                if page_text:
                    full_texts.append(page_text)

            combined_text = "\n\n".join(full_texts).strip()
            doc = ExtractedDocument(filename=filename, full_text=combined_text, pages=pages)
            doc.validate_quality()
            return doc
        except TextExtractionError:
            raise
        except Exception as exc:
            raise TextExtractionError(f"Failed to parse PDF '{filename}': {str(exc)}") from exc


class DocxExtractor(BaseExtractor):
    def extract(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        try:
            import docx
        except ImportError as exc:
            raise RuntimeError("python-docx is not installed. Run 'pip install python-docx'.") from exc

        try:
            stream = io.BytesIO(file_bytes)
            doc = docx.Document(stream)

            paragraphs = []
            for p in doc.paragraphs:
                txt = p.text.strip()
                if txt:
                    paragraphs.append(txt)

            # Also extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_txt = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_txt:
                        paragraphs.append(row_txt)

            combined = "\n\n".join(paragraphs).strip()
            pages = [{"page_number": 1, "text": combined}]
            extracted = ExtractedDocument(filename=filename, full_text=combined, pages=pages)
            extracted.validate_quality()
            return extracted
        except TextExtractionError:
            raise
        except Exception as exc:
            raise TextExtractionError(f"Failed to parse Word DOCX '{filename}': {str(exc)}") from exc


class TextExtractor(BaseExtractor):
    def extract(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        text = None
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text = file_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise TextExtractionError(f"Unable to decode text file '{filename}' with standard character encodings.")

        clean = text.strip()
        doc = ExtractedDocument(filename=filename, full_text=clean, pages=[{"page_number": 1, "text": clean}])
        doc.validate_quality()
        return doc


class CSVExtractor(BaseExtractor):
    def extract(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        text = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = file_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise TextExtractionError(f"Unable to decode CSV file '{filename}'.")

        try:
            stream = io.StringIO(text)
            reader = csv.reader(stream)
            rows = list(reader)

            if not rows or len(rows) < 2:
                raise TextExtractionError(f"CSV file '{filename}' must contain at least a header and 1 data row.")

            headers = [h.strip() for h in rows[0]]
            rendered_rows = []
            for r_idx, row in enumerate(rows[1:], start=2):
                if not any(c.strip() for c in row):
                    continue
                row_str = ", ".join(f"{h}: {val.strip()}" for h, val in zip(headers, row) if val.strip())
                if row_str:
                    rendered_rows.append(row_str)

            combined = "\n".join(rendered_rows).strip()
            doc = ExtractedDocument(
                filename=filename,
                full_text=combined,
                pages=[{"page_number": 1, "text": combined}],
            )
            doc.validate_quality()
            return doc
        except TextExtractionError:
            raise
        except Exception as exc:
            raise TextExtractionError(f"Failed to parse CSV '{filename}': {str(exc)}") from exc


def get_extractor(filename: str) -> BaseExtractor:
    """Returns the dedicated extractor for supported file types."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return PDFExtractor()
    elif lower.endswith(".docx"):
        return DocxExtractor()
    elif lower.endswith(".txt"):
        return TextExtractor()
    elif lower.endswith(".csv"):
        return CSVExtractor()
    else:
        ext = Path(filename).suffix or "unknown"
        raise TextExtractionError(
            f"Unsupported file format '{ext}'. Allowed formats are: .pdf, .docx, .txt, .csv"
        )
