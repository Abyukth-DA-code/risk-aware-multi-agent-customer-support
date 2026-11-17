
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
_cand_rag = [
    Path(__file__).resolve().parents[3] / "datasets" / "processed" / "rag",
    ROOT / "datasets" / "processed" / "rag",
    Path(__file__).resolve().parents[3] / "data" / "processed" / "rag",
    ROOT / "data" / "processed" / "rag",
    Path("datasets/processed/rag"),
    Path("data/processed/rag"),
]
RAG_DIR = next((c for c in _cand_rag if c.exists()), _cand_rag[0])
CHUNKS_FILE = RAG_DIR / "chunks.csv"
DOCS_FILE = RAG_DIR / "documents.csv"

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
BATCH_SIZE = 32


def main():
    if not CHUNKS_FILE.exists() or not DOCS_FILE.exists():
        raise FileNotFoundError("Required RAG CSV files are missing.")

    chunks = pd.read_csv(CHUNKS_FILE).fillna("")
    docs = pd.read_csv(DOCS_FILE).fillna("")

    required = {
        "Chunk_ID", "Document_ID", "Domain", "Document_Type",
        "Title", "Authority", "Version", "Freshness_Status",
        "Retrieval_Priority", "Chunk_Index", "Chunk_Text",
    }
    missing = required - set(chunks.columns)
    if missing:
        raise ValueError(f"Missing chunk columns: {sorted(missing)}")

    if chunks["Chunk_ID"].duplicated().any():
        raise ValueError("Duplicate Chunk_ID values detected.")

    if chunks["Chunk_ID"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty Chunk_ID values detected.")

    if chunks["Chunk_Text"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty chunk text detected.")

    if docs["Document_ID"].duplicated().any():
        raise ValueError("Duplicate Document_ID values detected.")

    document_ids = set(docs["Document_ID"].astype(str))
    missing_docs = set(chunks["Document_ID"].astype(str)) - document_ids
    if missing_docs:
        raise ValueError(
            f"Chunks reference missing documents: {sorted(missing_docs)}"
        )

    linked = chunks.merge(
        docs[["Document_ID", "Domain"]],
        on="Document_ID",
        suffixes=("_chunk", "_document"),
        validate="many_to_one",
    )
    mismatches = linked[
        linked["Domain_chunk"] != linked["Domain_document"]
    ]
    if not mismatches.empty:
        raise ValueError("Chunk/document domain mismatch detected.")

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert makes repeated runs idempotent for the same Chunk_IDs.
    # It does not delete unrelated IDs already in the collection.
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks.iloc[start:start + BATCH_SIZE]

        texts = batch["Chunk_Text"].astype(str).tolist()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = batch["Chunk_ID"].astype(str).tolist()
        metadatas = []

        for _, row in batch.iterrows():
            metadatas.append({
                "Document_ID": str(row["Document_ID"]),
                "Domain": str(row["Domain"]),
                "Document_Type": str(row["Document_Type"]),
                "Title": str(row["Title"]),
                "Authority": str(row["Authority"]),
                "Version": str(row["Version"]),
                "Freshness_Status": str(row["Freshness_Status"]),
                "Retrieval_Priority": int(row["Retrieval_Priority"]),
                "Chunk_Index": int(row["Chunk_Index"]),
            })

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"Indexed {min(start + len(batch), len(chunks))}/{len(chunks)}")

    print("\nRAG ingestion complete.")
    print("Collection:", COLLECTION_NAME)
    print("Expected chunks:", len(chunks))
    print("Collection count:", collection.count())
    print("Vector store:", VECTOR_DIR)


if __name__ == "__main__":
    main()
