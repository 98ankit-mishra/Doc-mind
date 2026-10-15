import os
from typing import Dict, Any
from groq import Groq

from .vector_store import query_similar_documents
from .prompts import format_prompt

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question: str) -> Dict[str, Any]:
    """
    Executes the RAG pipeline without LangChain:
    1. Retrieve relevant chunks using native ChromaDB.
    2. Build context.
    3. Query Groq LLM using native Groq SDK.
    4. Return answer and source pages.
    """
    retrieved_docs = query_similar_documents(question, k=4)

    if not retrieved_docs:
        return {
            "answer": "I could not find this information in the uploaded document.",
            "sources": []
        }

    # Format context
    context = format_docs(retrieved_docs)

    # Extract sources (page numbers)
    sources = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "Unknown")
        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
        sources.append({"page": page, "source": source_file})

    # Remove duplicates from sources while preserving order
    unique_sources = []
    seen = set()
    for s in sources:
        identifier = f"{s['source']}_{s['page']}"
        if identifier not in seen:
            seen.add(identifier)
            unique_sources.append(s)

    # Initialize Groq client
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    prompt = format_prompt(context=context, question=question)

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0
    )

    answer = completion.choices[0].message.content or ""

    return {
        "answer": answer.strip(),
        "sources": unique_sources
    }
