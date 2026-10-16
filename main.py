"""
main.py
=======
FastAPI application backend for DocMind - AI PDF Research Assistant.

Exposes REST API endpoints:
- GET /: Serves the web frontend UI.
- GET /health: Health check and index status.
- POST /upload: Uploads, validates, chunks, and indexes a PDF document into ChromaDB.
- POST /ask: Answers user questions using the six-stage RAG pipeline.
- POST /clear: Resets the vector database and clears uploaded files.
"""

import os
import re
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.pdf_processor import process_pdf
from src.vector_store import add_documents_to_db, clear_db, get_document_count, get_indexed_sources
from src.rag_chain import generate_answer

# Configure application logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("docmind")

app = FastAPI(
    title="DocMind - AI PDF Research Assistant",
    description="A production-quality Retrieval-Augmented Generation (RAG) assistant for PDF research.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB limit


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User search query or question.")


def sanitize_filename(filename: str) -> str:
    """
    Prevents path traversal and strips dangerous characters from uploaded filenames.
    """
    # Extract only the base name (prevents ../ and \..\ traversal)
    base = os.path.basename(filename).strip()
    # Replace any characters that are not alphanumeric, dots, underscores, or dashes
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    if not cleaned or cleaned.startswith("."):
        cleaned = f"upload_{cleaned}"
    return cleaned


@app.get("/health")
def health_check():
    """
    Health check endpoint returning system status and current index metadata.
    """
    return {
        "status": "healthy",
        "indexed_chunks": get_document_count(),
        "indexed_documents": get_indexed_sources()
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Validates, extracts, chunks, and stores a PDF document in ChromaDB.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload."
        )

    clean_name = sanitize_filename(file.filename)
    if not clean_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF (.pdf) documents are supported."
        )

    # Read file content with size check
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded file."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes)."
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 50 MB limit."
        )

    # Verify %PDF- magic bytes header
    if b"%PDF-" not in content[:1024]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format: File header is missing the standard '%PDF-' identifier."
        )

    file_path = os.path.join(UPLOAD_DIR, clean_name)

    try:
        # Save file to disk
        with open(file_path, "wb") as f:
            f.write(content)

        # Stage 1 & 2: PDF extraction and recursive chunking
        chunks = process_pdf(file_path)

        # Stage 3 & 4: Embeddings and persistent ChromaDB storage
        count = add_documents_to_db(chunks)

        logger.info(f"Successfully processed '{clean_name}': {count} chunks indexed.")
        return JSONResponse(
            content={
                "message": "PDF processed and indexed successfully",
                "filename": clean_name,
                "chunks_indexed": count
            },
            status_code=status.HTTP_200_OK
        )
    except ValueError as e:
        # User-correctable validation errors (empty PDF, scanned PDF needing OCR, etc.)
        logger.warning(f"Validation error processing PDF '{clean_name}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error processing PDF '{clean_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while extracting and indexing the document."
        )


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Answers a user question based on indexed document chunks using the RAG pipeline.
    """
    clean_q = request.question.strip()
    if not clean_q:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY is not configured in the environment. Please add it to your .env file."
        )

    try:
        response = generate_answer(clean_q)
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate an answer. Please check your API key and connection."
        )


@app.post("/clear")
def clear_session():
    """
    Clears the ChromaDB vector store and resets the index.
    """
    try:
        result = clear_db()
        return result
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear vector store."
        )


# Mount static assets (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount HTML templates for root UI serving
app.mount("/", StaticFiles(directory="templates", html=True), name="templates")
