"""
src/rag_chain.py
================
End-to-end RAG orchestrator integrating semantic retrieval with Groq LLM inference.

Stage 6: Context-Aware Answer Generation
- Integrates semantic retrieval with Groq API (Llama-3-8b-8192).
- Grounded prompt injection resistance and hallucination prevention.
- Source-page attribution tracking.
- Abstention handling when context is absent or insufficient.
- Supports dependency injection for seamless testing without live Groq API keys.
"""

import os
from typing import Dict, Any, List, Optional
from groq import Groq

from .vector_store import query_similar_documents, get_document_count
from .prompts import format_prompt, SYSTEM_INSTRUCTION, ABSTENTION_MESSAGE
from .pdf_processor import Document

DEFAULT_MODEL = "llama3-8b-8192"


def format_docs(docs: List[Document]) -> str:
    """
    Formats retrieved documents into delimited sections with metadata headers.
    """
    formatted_parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "Unknown")
        formatted_parts.append(f"[Document: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted_parts)


def get_groq_client(api_key: Optional[str] = None) -> Groq:
    """
    Initializes and returns the Groq client.
    Raises ValueError if GROQ_API_KEY is not configured.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key or not key.strip():
        raise ValueError("GROQ_API_KEY is not set. Please add your Groq API key to the .env file.")
    return Groq(api_key=key.strip())


def generate_answer(
    question: str,
    k: int = 4,
    groq_client: Optional[Any] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Executes the six-stage RAG pipeline:
    1. Validates query input and document index status.
    2. Retrieves top-k semantically similar chunks from ChromaDB.
    3. Builds grounded prompt with injection guardrails.
    4. Queries Groq LLM (Llama 3).
    5. Returns the grounded answer and page-level source references.

    Args:
        question: User query string.
        k: Number of relevant chunks to retrieve (default: 4).
        groq_client: Optional Groq client or mock for testing.
        model: Model identifier on Groq (default: 'llama3-8b-8192').

    Returns:
        Dict[str, Any]: {"answer": str, "sources": List[Dict[str, Any]]}
    """
    clean_question = question.strip() if question else ""
    if not clean_question:
        return {
            "answer": "Question cannot be empty.",
            "sources": []
        }

    # Verify if any documents have been indexed
    total_docs = get_document_count()
    if total_docs == 0:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF first.",
            "sources": []
        }

    # Stage 5: Semantic Retrieval
    retrieved_docs = query_similar_documents(clean_question, k=k)

    if not retrieved_docs:
        return {
            "answer": ABSTENTION_MESSAGE,
            "sources": []
        }

    # Format context string
    context_str = format_docs(retrieved_docs)

    # Extract source references
    raw_sources: List[Dict[str, Any]] = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "Unknown")
        source_file = doc.metadata.get("source", "Unknown")
        raw_sources.append({"page": page, "source": source_file})

    # Deduplicate sources while preserving ranking order
    unique_sources: List[Dict[str, Any]] = []
    seen = set()
    for s in raw_sources:
        identifier = f"{s['source']}_{s['page']}"
        if identifier not in seen:
            seen.add(identifier)
            unique_sources.append(s)

    # Stage 6: Answer Generation via Groq API
    client = groq_client if groq_client is not None else get_groq_client()

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": format_prompt(context=context_str, question=clean_question)}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0
    )

    answer = completion.choices[0].message.content or ""
    answer_clean = answer.strip()

    # Grounding check: If the model abstained, do not report supporting sources
    if ABSTENTION_MESSAGE.lower() in answer_clean.lower():
        return {
            "answer": ABSTENTION_MESSAGE,
            "sources": []
        }

    return {
        "answer": answer_clean,
        "sources": unique_sources
    }
