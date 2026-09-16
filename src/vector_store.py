import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .embeddings import get_embeddings

CHROMA_PATH = "chroma_db"

def get_vector_store() -> Chroma:
    """
    Initializes or loads the ChromaDB vector store.
    """
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

def add_documents_to_db(documents: List[Document]):
    """
    Adds document chunks to the ChromaDB vector store.
    """
    vector_store = get_vector_store()
    vector_store.add_documents(documents)

def get_retriever(k: int = 4):
    """
    Returns a retriever configured to fetch the top-k most relevant chunks.
    """
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})

def clear_db():
    """
    Clears the current vector store by resetting it (or deleting the directory).
    """
    # Simple way to clear is just initializing and resetting if allowed, 
    # or deleting the directory.
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
