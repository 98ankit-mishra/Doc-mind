"""
src/embeddings.py
=================
Dense vector embedding generation using the HuggingFace Sentence Transformers ecosystem.
Model: 'all-MiniLM-L6-v2' (384-dimensional dense vectors).

Key Characteristics:
- Pure HuggingFace Sentence Transformers model without LangChain.
- Computes normalized embeddings (L2-norm = 1.0) so cosine similarity is directly
  proportional to dot product.
- Implements the ChromaDB EmbeddingFunction protocol for seamless integration.
- Provides thread-safe lazy-loading singleton to prevent redundant model instantiations.
- Keeps embedding inference local on CPU/GPU once the model weights are cached.
"""

from typing import List, Union, Any, Optional
import threading
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class HuggingFaceEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Explicit HuggingFace SentenceTransformer embedding function for DocMind.
    Uses 'all-MiniLM-L6-v2' to produce 384-dimensional dense vector embeddings.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def get_config(self) -> dict:
        """Returns serializable configuration for ChromaDB."""
        return {
            "model_name": self.model_name,
            "device": self.device
        }

    @classmethod
    def build_from_config(cls, config: dict) -> "HuggingFaceEmbeddingFunction":
        """Reconstructs embedding function from serialized configuration."""
        return cls(
            model_name=config.get("model_name", MODEL_NAME),
            device=config.get("device", "cpu")
        )

    @staticmethod
    def name() -> str:
        """Identifies the embedding function in ChromaDB collection configuration."""
        return "huggingface_sentence_transformer"

    @property
    def model(self):
        """
        Lazily loads the SentenceTransformer model on first invocation
        with thread-safety to avoid race conditions.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        """
        ChromaDB EmbeddingFunction interface.
        Takes a list of document strings and returns a list of float vectors.
        """
        if not input:
            return []

        # Generate normalized dense embeddings
        embeddings = self.model.encode(
            list(input),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, input: Any) -> Any:
        """
        Generates dense vector embeddings for query input.
        - If input is a single string: returns a single 384-dim vector List[float].
        - If input is a list of strings (as invoked by ChromaDB query): returns List[List[float]].
        """
        if isinstance(input, str):
            if not input.strip():
                raise ValueError("Query text cannot be empty.")
            raw = self.model.encode(
                input,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return raw.tolist()
        elif isinstance(input, (list, tuple)):
            if not input:
                return []
            raw = self.model.encode(
                list(input),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return raw.tolist()
        else:
            raw = self.model.encode(
                str(input),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return raw.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates 384-dimensional dense vectors for a batch of text chunks.
        """
        if not texts:
            return []
        return self(texts)

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM


# Global cached singleton instance
_embedding_instance: Optional[HuggingFaceEmbeddingFunction] = None
_instance_lock = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddingFunction:
    """
    Returns the singleton HuggingFaceEmbeddingFunction instance.
    """
    global _embedding_instance
    if _embedding_instance is None:
        with _instance_lock:
            if _embedding_instance is None:
                _embedding_instance = HuggingFaceEmbeddingFunction()
    return _embedding_instance
