from langchain_core.prompts import PromptTemplate

# A clear and strict prompt template as requested by the user.
RAG_PROMPT_TEMPLATE = """
Answer the user's question only using the provided document context.

If the answer cannot be found in the provided context, say:
'I could not find this information in the uploaded document.'

Do not invent facts.
Do not use outside knowledge to answer document-specific questions.

Context:
{context}

Question: {question}

Answer:
"""

def get_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
