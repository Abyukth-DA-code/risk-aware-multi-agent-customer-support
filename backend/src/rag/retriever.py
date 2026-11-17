
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
_cand_vector = [
    ROOT / "vector_store" / "rag_chroma",
    Path(__file__).resolve().parents[3] / "backend" / "vector_store" / "rag_chroma",
    Path(__file__).resolve().parents[3] / "vector_store" / "rag_chroma",
    Path("backend/vector_store/rag_chroma"),
    Path("vector_store/rag_chroma"),
]
VECTOR_DIR = next((c for c in _cand_vector if c.exists()), _cand_vector[0])
COLLECTION_NAME = "capstone_knowledge_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RAGRetriever:
    _shared_model = None

    def __init__(self, client=None):
        if not VECTOR_DIR.exists():
            raise FileNotFoundError(
                "Vector store not found. Run src/rag/ingest.py first."
            )

        if RAGRetriever._shared_model is None:
            try:
                RAGRetriever._shared_model = SentenceTransformer(
                    EMBEDDING_MODEL, local_files_only=True
                )
            except Exception:
                RAGRetriever._shared_model = SentenceTransformer(EMBEDDING_MODEL)

        self.model = RAGRetriever._shared_model
        self.client = client or chromadb.PersistentClient(path=str(VECTOR_DIR))

        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME
            )
        except Exception as exc:
            if client is not None:
                self.collection = None
            else:
                raise RuntimeError(
                    "RAG collection not found. Run src/rag/ingest.py first."
                ) from exc

        if self.collection is not None and self.collection.count() == 0 and client is None:
            raise RuntimeError("RAG collection is empty.")

        self._bm25_index = None
        self.last_query_embedding = None

    @property
    def bm25(self):
        if self._bm25_index is None:
            try:
                from src.rag.bm25 import BM25OkapiIndex
            except ImportError:
                from bm25 import BM25OkapiIndex
            self._bm25_index = BM25OkapiIndex()
        return self._bm25_index

    def retrieve(
        self,
        query,
        domain=None,
        top_k=5,
        distinct_documents=False,
        candidate_pool_size=None,
        candidate_multiplier=4,
        hybrid=False,
        rrf_k=60,
        workspace_id=None,
    ):
        query = str(query).strip()

        if not query:
            raise ValueError("Query must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Workspace-Aware Collection Resolution
        # If workspace_id is provided, query strictly that workspace's dedicated collection.
        # NEVER silently fall back to the global demo collection!
        if workspace_id:
            coll_name = f"workspace_{workspace_id}"
            try:
                active_collection = self.client.get_collection(name=coll_name)
            except Exception:
                return []
            available = active_collection.count()
            if available == 0:
                return []
            where = {"workspace_id": workspace_id}
            # For tenant workspaces, pure dense distinct-document retrieval is required
            distinct_documents = True
            hybrid = False
        else:
            active_collection = self.collection
            where = {"Domain": domain} if domain else None
            available = active_collection.count()
            if available == 0:
                return []

        # Candidate pool size calculation
        if distinct_documents or hybrid:
            if candidate_pool_size is not None:
                fetch_k = max(top_k, int(candidate_pool_size))
            else:
                fetch_k = max(top_k, int(top_k * candidate_multiplier))
        else:
            fetch_k = top_k

        fetch_k = min(fetch_k, available)

        # -------------------------------------------------------------
        # PURE DENSE RETRIEVAL (Original raw-chunk and distinct modes)
        # -------------------------------------------------------------
        if not hybrid:
            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            self.last_query_embedding = query_embedding[0] if query_embedding else None

            result = active_collection.query(
                query_embeddings=query_embedding,
                n_results=fetch_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            ids = result["ids"][0] if result and result["ids"] else []

            candidates = []
            for i, chunk_id in enumerate(ids):
                metadata = result["metadatas"][0][i] or {}
                distance = float(result["distances"][0][i])
                doc_id = str(metadata.get("document_id") or metadata.get("Document_ID") or "").strip()
                title = str(metadata.get("filename") or metadata.get("Title") or "Policy Document").strip()
                dom_val = str(metadata.get("domain") or metadata.get("Domain") or "").strip()
                authority = str(metadata.get("authority") or metadata.get("Authority") or "High").strip()
                freshness = str(metadata.get("freshness_status") or metadata.get("Freshness_Status") or "Current").strip()
                doc_type = str(metadata.get("category_key") or metadata.get("Document_Type") or "Policy").strip()
                priority = int(metadata.get("retrieval_priority") or metadata.get("Retrieval_Priority", 1))

                chunk_dict = {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "domain": dom_val,
                    "title": title,
                    "authority": authority,
                    "freshness_status": freshness,
                    "document_type": doc_type,
                    "retrieval_priority": priority,
                    "chunk_text": result["documents"][0][i] if result["documents"] else "",
                    "cosine_distance": distance,
                    "similarity": 1.0 - distance,
                }
                if workspace_id is not None or "workspace_id" in metadata:
                    chunk_dict["workspace_id"] = metadata.get("workspace_id", workspace_id)
                candidates.append(chunk_dict)

            # Prioritize canonical documents (priority 1) over outdated/test documents (priority 3), then cosine distance
            candidates.sort(key=lambda x: (x["retrieval_priority"], x["cosine_distance"], x["chunk_id"]))

            items = []
            seen_docs = set()
            for cand in candidates:
                if distinct_documents:
                    if cand["document_id"] in seen_docs:
                        continue
                    seen_docs.add(cand["document_id"])
                items.append(cand)
                if len(items) == top_k:
                    break

            return items

        # -------------------------------------------------------------
        # HYBRID LEXICAL-DENSE RETRIEVAL (BM25 + Dense + RRF)
        # -------------------------------------------------------------
        # 1. Fetch dense candidates
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        self.last_query_embedding = query_embedding[0] if query_embedding else None

        dense_result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        dense_ids = dense_result["ids"][0] if dense_result and dense_result["ids"] else []
        dense_ranks = {}
        dense_info = {}

        for rank, cid in enumerate(dense_ids, start=1):
            meta = dense_result["metadatas"][0][rank - 1] or {}
            dist = float(dense_result["distances"][0][rank - 1])
            text = dense_result["documents"][0][rank - 1] if dense_result["documents"] else ""
            dense_ranks[cid] = rank
            dense_info[cid] = {
                "metadata": meta,
                "distance": dist,
                "text": text,
            }

        # 2. Fetch BM25 candidates
        lex_candidates = self.bm25.retrieve(
            query=query,
            domain=domain,
            top_k=fetch_k,
        )

        lex_ranks = {}
        lex_scores = {}
        lex_info = {}

        for rank, item in enumerate(lex_candidates, start=1):
            cid = item["chunk_id"]
            lex_ranks[cid] = rank
            lex_scores[cid] = float(item["bm25_score"])
            lex_info[cid] = item

        # 3. Reciprocal Rank Fusion (RRF)
        all_candidate_ids = list(dict.fromkeys(list(dense_ranks.keys()) + list(lex_ranks.keys())))

        scored_candidates = []
        for cid in all_candidate_ids:
            r_dense = dense_ranks.get(cid)
            r_lex = lex_ranks.get(cid)

            rrf_dense = (1.0 / (rrf_k + r_dense)) if r_dense is not None else 0.0
            rrf_lex = (1.0 / (rrf_k + r_lex)) if r_lex is not None else 0.0
            rrf_score = rrf_dense + rrf_lex

            # Retrieve metadata
            if cid in dense_info:
                meta = dense_info[cid]["metadata"]
                dist = dense_info[cid]["distance"]
                sim = 1.0 - dist
                text = dense_info[cid]["text"]
            else:
                lex_item = lex_info[cid]
                meta = {
                    "Document_ID": lex_item["document_id"],
                    "Domain": lex_item["domain"],
                    "Title": lex_item["title"],
                    "Authority": lex_item["authority"],
                    "Freshness_Status": lex_item["freshness_status"],
                    "Document_Type": lex_item["document_type"],
                    "Retrieval_Priority": lex_item["retrieval_priority"],
                }
                dist = 1.0
                sim = 0.0
                text = lex_item["chunk_text"]

            priority = int(meta.get("Retrieval_Priority", 1))

            scored_candidates.append({
                "chunk_id": cid,
                "document_id": str(meta.get("Document_ID", "")).strip(),
                "domain": meta.get("Domain", ""),
                "title": meta.get("Title", ""),
                "authority": meta.get("Authority", ""),
                "freshness_status": meta.get("Freshness_Status", ""),
                "document_type": meta.get("Document_Type", ""),
                "retrieval_priority": priority,
                "chunk_text": text,
                "cosine_distance": dist,
                "similarity": sim,
                "rrf_score": rrf_score,
                "bm25_score": lex_scores.get(cid, 0.0),
                "dense_rank": r_dense,
                "lexical_rank": r_lex,
            })

        # Sort descending by RRF score; break ties by retrieval priority, then chunk_id
        scored_candidates.sort(
            key=lambda x: (-x["rrf_score"], x["retrieval_priority"], x["chunk_id"])
        )

        # 4. Deduplication and Selection
        items = []
        seen_docs = set()
        for cand in scored_candidates:
            doc_id = cand["document_id"]
            if distinct_documents:
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)

            items.append(cand)
            if len(items) == top_k:
                break

        return items


if __name__ == "__main__":
    retriever = RAGRetriever()

    query = "How can I check my order status?"
    results = retriever.retrieve(
        query=query,
        domain="E-commerce",
        top_k=5,
    )

    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")

    for rank, item in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print("Chunk ID:", item["chunk_id"])
        print("Document ID:", item["document_id"])
        print("Title:", item["title"])
        print("Domain:", item["domain"])
        print("Authority:", item["authority"])
        print("Freshness:", item["freshness_status"])
        print("Similarity:", round(item["similarity"], 4))
        print("Text:", item["chunk_text"][:500])
