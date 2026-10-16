"""
src/vector_store.py
===================
ChromaDB persistent vector storage and semantic retrieval layer.

Stage 4: ChromaDB Vector Storage
- Persistent storage using chromadb.PersistentClient.
- Generates deterministic, collision-resistant chunk IDs using SHA-256.
- Prevents duplicate chunks on repeated uploads by purging previous chunks for the same source.
- Preserves multi-document metadata (source filename, page number, chunk index).
- Configured with HuggingFace SentenceTransformer 'all-MiniLM-L6-v2' (cosine distance space).

Stage 5: Semantic Retrieval
- Retrieves top-k (default: 4) most relevant document chunks for a user query.
- Computes and attaches normalized similarity scores and distances to retrieved metadata.
- Supports optional similarity filtering.
- Safely abstains if the collection is empty.
"""

import os
import shutil
import hashlib
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from .embeddings import get_embeddings
from .pdf_processor import Document

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "docmind_collection"


def get_client() -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client pointing to the local storage path.
    """
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )


def get_collection():
    """
    Initializes or retrieves the persistent ChromaDB collection configured
    with the HuggingFace SentenceTransformer embedding function.
    Safely recovers if the stored collection has an incompatible dimension.
    """
    client = get_client()
    embedding_fn = get_embeddings()

    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_fn.model_name,
                "dimension": embedding_fn.dimension
            }
        )
        return collection
    except Exception as e:
        # If existing collection has incompatible dimension or embedding schema, reset safely
        client.delete_collection(name=COLLECTION_NAME)
        return client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_fn.model_name,
                "dimension": embedding_fn.dimension
            }
        )


def generate_stable_chunk_id(doc: Document, fallback_idx: int) -> str:
    """
    Generates a deterministic, collision-resistant chunk ID using SHA-256.
    Ensures stable IDs across runs and prevents randomized hash collisions.
    """
    source = doc.metadata.get("source", "doc")
    page = doc.metadata.get("page", 1)
    chunk_idx = doc.metadata.get("chunk_index", fallback_idx)

    # Hash the exact content
    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:12]
    # Clean source name for ID safety
    safe_source = "".join(c for c in source if c.isalnum() or c in ("-", "_", "."))
    return f"{safe_source}_p{page}_c{chunk_idx}_{content_hash}"


def add_documents_to_db(documents: List[Document]) -> int:
    """
    Adds document chunks and metadata to the ChromaDB vector store.
    If chunks from the same source file already exist, they are purged first
    to avoid duplicate records upon re-uploading.

    Returns:
        int: Number of chunks added.
    """
    if not documents:
        return 0

    collection = get_collection()

    # Identify all source files in this batch
    sources = list({doc.metadata.get("source") for doc in documents if doc.metadata.get("source")})

    # Purge existing chunks for these sources to guarantee idempotence
    for src in sources:
        try:
            collection.delete(where={"source": src})
        except Exception:
            pass

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for idx, doc in enumerate(documents):
        texts.append(doc.page_content)
        # Ensure metadata values are JSON/ChromaDB compatible (str, int, float, bool)
        clean_meta = {}
        for k, v in doc.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        metadatas.append(clean_meta)
        ids.append(generate_stable_chunk_id(doc, idx))

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    return len(documents)


def query_similar_documents(
    query: str,
    k: int = 4,
    max_distance: Optional[float] = None
) -> List[Document]:
    """
    Queries ChromaDB and returns top-k matching Document objects with metadata
    and similarity scores.

    Args:
        query: User search query string.
        k: Maximum number of chunks to return (default: 4).
        max_distance: Optional cosine distance cutoff (0.0 = identical, 1.0 = orthogonal).

    Returns:
        List[Document]: Retrieved document chunks sorted by relevance.
    """
    if not query or not query.strip():
        return []

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    n_results = min(k, count)
    results = collection.query(
        query_texts=[query.strip()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    docs: List[Document] = []
    if results and "documents" in results and results["documents"]:
        retrieved_texts = results["documents"][0]
        retrieved_metas = (
            results["metadatas"][0]
            if "metadatas" in results and results["metadatas"]
            else [{}] * len(retrieved_texts)
        )
        distances = (
            results["distances"][0]
            if "distances" in results and results["distances"]
            else [0.0] * len(retrieved_texts)
        )

        for text, meta, dist in zip(retrieved_texts, retrieved_metas, distances):
            # Optional distance threshold filtering
            if max_distance is not None and dist > max_distance:
                continue

            doc_meta = dict(meta or {})
            doc_meta["distance"] = round(float(dist), 4)
            # Cosine distance to similarity score: score = 1.0 - distance
            doc_meta["similarity_score"] = round(float(1.0 - dist), 4)

            docs.append(Document(page_content=text, metadata=doc_meta))

    return docs


def get_document_count() -> int:
    """
    Returns the total number of stored chunks in the vector collection.
    """
    try:
        collection = get_collection()
        return collection.count()
    except Exception:
        return 0


def get_indexed_sources() -> List[str]:
    """
    Returns a sorted list of unique source filenames indexed in ChromaDB.
    """
    try:
        collection = get_collection()
        if collection.count() == 0:
            return []
        data = collection.get(include=["metadatas"])
        if data and "metadatas" in data and data["metadatas"]:
            sources = {m.get("source") for m in data["metadatas"] if m and m.get("source")}
            return sorted(list(sources))
        return []
    except Exception:
        return []


def clear_db() -> Dict[str, Any]:
    """
    Safely clears the vector store by deleting the collection and recreating
    a fresh empty collection. Avoids Windows filesystem locking errors.
    """
    client = get_client()
    embedding_fn = get_embeddings()

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    # Recreate the fresh empty collection
    client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": embedding_fn.model_name,
            "dimension": embedding_fn.dimension
        }
    )

    return {"message": "Vector database cleared successfully."}
