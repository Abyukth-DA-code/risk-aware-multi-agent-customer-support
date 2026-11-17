"""
BM25Okapi Lexical Index for Customer Support RAG.

Provides deterministic BM25Okapi chunk retrieval over data/processed/rag/chunks.csv.
Pure Python/NumPy implementation with zero external dependencies beyond standard library and pandas/numpy.
Supports domain filtering, query tokenization, title enrichment, and rank generation.
"""

from collections import Counter
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "rag" / "chunks.csv"


class BM25OkapiIndex:
    """
    BM25Okapi retrieval engine for knowledge-base chunks.

    Formula:
        IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1.0)
        TF_comp = (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl)))
        Score(D, Q) = sum_{q in Q} IDF(q) * TF_comp
    """

    def __init__(
        self,
        chunks_path: Optional[Path] = None,
        k1: float = 1.5,
        b: float = 0.75,
        include_title: bool = True,
    ):
        self.chunks_path = Path(chunks_path) if chunks_path else CHUNKS_PATH
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")

        self.k1 = float(k1)
        self.b = float(b)
        self.include_title = bool(include_title)

        self._build_index()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words."""
        if not text:
            return []
        return re.findall(r"\b\w+\b", str(text).lower())

    def _build_index(self):
        df = pd.read_csv(self.chunks_path).fillna("")
        self.records = []
        self.corpus_tokens = []
        self.doc_lens = []

        for _, row in df.iterrows():
            title = str(row.get("Title", "")).strip()
            chunk_text = str(row.get("Chunk_Text", "")).strip()

            if self.include_title and title:
                full_searchable = f"{title} {chunk_text}"
            else:
                full_searchable = chunk_text

            tokens = self.tokenize(full_searchable)
            self.corpus_tokens.append(tokens)
            self.doc_lens.append(len(tokens))

            self.records.append({
                "chunk_id": str(row["Chunk_ID"]).strip(),
                "document_id": str(row["Document_ID"]).strip(),
                "domain": str(row.get("Domain", "")).strip(),
                "document_type": str(row.get("Document_Type", "")).strip(),
                "title": title,
                "authority": str(row.get("Authority", "")).strip(),
                "version": str(row.get("Version", "")).strip(),
                "freshness_status": str(row.get("Freshness_Status", "")).strip(),
                "retrieval_priority": int(row.get("Retrieval_Priority", 1)),
                "chunk_index": int(row.get("Chunk_Index", 1)),
                "chunk_text": chunk_text,
            })

        self.N = len(self.records)
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N > 0 else 0.0

        # Term document frequency: count of documents containing term t
        df_counter = Counter()
        for tokens in self.corpus_tokens:
            unique_terms = set(tokens)
            df_counter.update(unique_terms)

        self.df = dict(df_counter)
        self.idf = {}
        for term, n_t in self.df.items():
            # Standard Lucene/BM25Okapi smooth IDF formula
            self.idf[term] = math.log(((self.N - n_t + 0.5) / (n_t + 0.5)) + 1.0)

    def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Score all chunks against query and return top_k matches.

        Args:
            query: User search query string.
            domain: Optional domain filter (e.g. 'E-commerce').
            top_k: Number of chunks to return.
            min_score: Minimum BM25 score required for inclusion. Defaults to 0.0
                       (only strictly positive scores returned).
        """
        query = str(query).strip()
        if not query:
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        scored_candidates = []
        for idx, rec in enumerate(self.records):
            if domain and rec["domain"] != domain:
                continue

            doc_len = self.doc_lens[idx]
            if doc_len == 0:
                continue

            # Compute term frequencies in document
            doc_tokens = self.corpus_tokens[idx]
            tf_map = Counter(doc_tokens)

            score = 0.0
            for term in q_tokens:
                if term in tf_map:
                    f_td = tf_map[term]
                    idf_t = self.idf.get(term, 0.0)
                    denom = f_td + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf_t * (f_td * (self.k1 + 1.0)) / denom

            if score > min_score:
                item = dict(rec)
                item["bm25_score"] = float(score)
                scored_candidates.append(item)

        # Sort descending by score; break ties deterministically by retrieval_priority asc, chunk_id asc
        scored_candidates.sort(
            key=lambda x: (-x["bm25_score"], x["retrieval_priority"], x["chunk_id"])
        )

        return scored_candidates[:top_k]
