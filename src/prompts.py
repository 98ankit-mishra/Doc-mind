RAG_PROMPT_TEMPLATE = """Answer the user's question only using the provided document context.

If the answer cannot be found in the provided context, say:
'I could not find this information in the uploaded document.'

Do not invent facts.
Do not use outside knowledge to answer document-specific questions.

Context:
{context}

Question: {question}

Answer:"""

def format_prompt(context: str, question: str) -> str:
    """
    Formats the RAG prompt string without LangChain PromptTemplate.
    """
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question).strip()
