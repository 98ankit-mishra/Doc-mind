import os
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .vector_store import get_retriever
from .prompts import get_prompt_template

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question: str) -> Dict[str, Any]:
    """
    Executes the RAG pipeline:
    1. Retrieve relevant chunks.
    2. Build context.
    3. Query Groq LLM.
    4. Return answer and source pages.
    """
    retriever = get_retriever()
    
    # Retrieve relevant documents directly to return sources
    retrieved_docs = retriever.invoke(question)
    
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

    # Initialize Groq LLM
    # Expects GROQ_API_KEY in environment variables
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
    
    prompt = get_prompt_template()
    
    # Create the LCEL chain
    chain = prompt | llm | StrOutputParser()
    
    # Execute chain
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    return {
        "answer": answer,
        "sources": unique_sources
    }
