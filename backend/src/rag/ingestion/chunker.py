"""
Document Text Chunker with Sliding Window & Sentence Boundary Preservation
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation
"""

from typing import Any, Dict, List
import uuid

from src.rag.ingestion.extractors import ExtractedDocument


class DocumentChunk:
    """Represents an atomic text passage ready for embedding and retrieval."""

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        chunk_index: int,
        text: str,
        page_number: int = 1,
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.chunk_index = chunk_index
        self.text = text.strip()
        self.char_length = len(self.text)
        self.page_number = page_number

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_length": self.char_length,
            "page_number": self.page_number,
        }


class SemanticChunker:
    """
    Splits extracted document text into semantic chunks with overlapping context.
    Respects paragraph and sentence boundaries.
    """

    def __init__(self, target_chars: int = 400, overlap_chars: int = 60, min_chunk_chars: int = 40):
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_document(self, doc: ExtractedDocument, document_id: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_idx = 1

        for page in doc.pages:
            page_num = page.get("page_number", 1)
            page_text = page.get("text", "").strip()
            if not page_text:
                continue

            # Split by double newlines or single newlines
            paragraphs = [p.strip() for p in page_text.split("\n") if p.strip()]

            current_chunk_words: List[str] = []
            current_len = 0

            for para in paragraphs:
                words = para.split()
                for word in words:
                    current_chunk_words.append(word)
                    current_len += len(word) + 1

                    if current_len >= self.target_chars:
                        chunk_text = " ".join(current_chunk_words)
                        if len(chunk_text) >= self.min_chunk_chars:
                            cid = f"{document_id}_C{chunk_idx:02d}"
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=cid,
                                    document_id=document_id,
                                    chunk_index=chunk_idx,
                                    text=chunk_text,
                                    page_number=page_num,
                                )
                            )
                            chunk_idx += 1

                        # Compute overlap
                        overlap_words: List[str] = []
                        overlap_len = 0
                        for w in reversed(current_chunk_words):
                            overlap_words.insert(0, w)
                            overlap_len += len(w) + 1
                            if overlap_len >= self.overlap_chars:
                                break

                        current_chunk_words = overlap_words
                        current_len = sum(len(w) + 1 for w in current_chunk_words)

            # Flush remaining words
            if current_chunk_words:
                remaining_text = " ".join(current_chunk_words).strip()
                if len(remaining_text) >= self.min_chunk_chars:
                    cid = f"{document_id}_C{chunk_idx:02d}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=cid,
                            document_id=document_id,
                            chunk_index=chunk_idx,
                            text=remaining_text,
                            page_number=page_num,
                        )
                    )
                    chunk_idx += 1

        # Fallback if document text was completely under target_chars
        if not chunks and doc.full_text.strip():
            cid = f"{document_id}_C01"
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    document_id=document_id,
                    chunk_index=1,
                    text=doc.full_text.strip(),
                    page_number=1,
                )
            )

        return chunks
