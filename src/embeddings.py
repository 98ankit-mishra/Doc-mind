from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    Returns the HuggingFace embeddings model.
    Uses 'all-MiniLM-L6-v2' as it is fast, free, and runs locally.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
