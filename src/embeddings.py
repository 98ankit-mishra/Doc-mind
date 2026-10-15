from chromadb.utils import embedding_functions

def get_embeddings():
    """
    Returns the native ChromaDB embedding function.
    Uses 'all-MiniLM-L6-v2' running locally without LangChain.
    """
    return embedding_functions.DefaultEmbeddingFunction()
