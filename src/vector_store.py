import os
import shutil
from typing import List
import chromadb
from .embeddings import get_embeddings
from .pdf_processor import Document

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "docmind_collection"

def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection():
    """
    Initializes or loads the persistent ChromaDB collection.
    """
    client = get_client()
    embedding_fn = get_embeddings()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

def add_documents_to_db(documents: List[Document]):
    """
    Adds document chunks to the ChromaDB vector store.
    """
    if not documents:
        return

    collection = get_collection()
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [f"chunk_{i}_{abs(hash(doc.page_content))}" for i, doc in enumerate(documents)]

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

def query_similar_documents(query: str, k: int = 4) -> List[Document]:
    """
    Retrieves top-k most similar document chunks for the query.
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    n_results = min(k, count)
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    docs: List[Document] = []
    if results and "documents" in results and results["documents"]:
        retrieved_texts = results["documents"][0]
        retrieved_metas = (
            results["metadatas"][0]
            if "metadatas" in results and results["metadatas"]
            else [{}] * len(retrieved_texts)
        )

        for text, meta in zip(retrieved_texts, retrieved_metas):
            docs.append(Document(page_content=text, metadata=meta or {}))

    return docs

def clear_db():
    """
    Clears the current vector store.
    """
    try:
        client = get_client()
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
