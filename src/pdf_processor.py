import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Loads a PDF and splits it into chunks.
    
    Args:
        file_path: Path to the PDF file.
        chunk_size: Maximum size of chunks.
        chunk_overlap: Overlap between chunks.
        
    Returns:
        List of Document objects with text and metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    if not docs:
        raise ValueError("The PDF is empty or could not be parsed.")
        
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(docs)
    return chunks
