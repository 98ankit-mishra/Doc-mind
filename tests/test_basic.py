"""
tests/test_basic.py
===================
Comprehensive test suite for DocMind RAG pipeline and FastAPI endpoints:
- PDF extraction and 1-based page numbering
- Recursive text chunking and overlap bounds
- HuggingFace SentenceTransformer embeddings (384 dimensions, normalized)
- ChromaDB vector store persistence, stable SHA-256 IDs, and multi-document handling
- Semantic retrieval (top-4 behavior and metadata tracking)
- Grounding prompt injection defense and abstention behavior
- Mock Groq LLM inference for offline/deterministic test execution
- FastAPI REST endpoints (/health, /upload, /ask, /clear)
"""

import os
import shutil
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from main import app
from src.pdf_processor import process_pdf, split_text, Document
from src.embeddings import get_embeddings, MODEL_NAME, EMBEDDING_DIM
from src.vector_store import (
    add_documents_to_db,
    query_similar_documents,
    clear_db,
    get_document_count,
    get_indexed_sources,
    generate_stable_chunk_id
)
from src.prompts import format_prompt, SYSTEM_INSTRUCTION, ABSTENTION_MESSAGE
from src.rag_chain import generate_answer, format_docs

client = TestClient(app)

DUMMY_PDF = "test_dummy.pdf"
DUMMY_PDF_2 = "test_dummy_2.pdf"
EMPTY_TEXT_PDF = "test_empty_text.pdf"
CORRUPT_PDF = "test_corrupt.pdf"


class MockGroqClient:
    """Mock Groq client for deterministic, offline testing."""
    def __init__(self, response_text: str = "Paris is the capital of France."):
        self.response_text = response_text
        self.last_messages = None

        class MockChoice:
            def __init__(self, content):
                self.message = type("Message", (), {"content": content})

        class MockCompletion:
            def __init__(self, content):
                self.choices = [MockChoice(content)]

        class MockCompletions:
            def __init__(self, outer):
                self.outer = outer

            def create(self, **kwargs):
                self.outer.last_messages = kwargs.get("messages")
                return MockCompletion(self.outer.response_text)

        self.chat = type("Chat", (), {"completions": MockCompletions(self)})


@pytest.fixture(scope="module", autouse=True)
def setup_teardown_suite():
    # 1. Create standard multi-page dummy PDF
    c = canvas.Canvas(DUMMY_PDF)
    c.drawString(100, 750, "DocMind Artificial Intelligence Research Assistant Overview.")
    c.drawString(100, 730, "The capital of France is Paris. Paris has a population of over two million.")
    c.showPage()
    c.drawString(100, 750, "Page two covers European economics and geography.")
    c.drawString(100, 730, "The currency of France is the Euro, adopted in 1999.")
    c.save()

    # 2. Create second dummy PDF for multi-document testing
    c2 = canvas.Canvas(DUMMY_PDF_2)
    c2.drawString(100, 750, "Second document regarding planetary science.")
    c2.drawString(100, 730, "Mars is the fourth planet from the Sun, known as the Red Planet.")
    c2.save()

    # 3. Create PDF with no extractable text (simulates scanned image/empty page)
    c3 = canvas.Canvas(EMPTY_TEXT_PDF)
    c3.showPage()  # Empty page
    c3.save()

    # 4. Create corrupt non-PDF file
    with open(CORRUPT_PDF, "wb") as f:
        f.write(b"This is not a PDF file at all.")

    yield

    # Clean up test files
    for path in [DUMMY_PDF, DUMMY_PDF_2, EMPTY_TEXT_PDF, CORRUPT_PDF]:
        if os.path.exists(path):
            os.remove(path)

    # Clean vector store
    clear_db()


# ============================================================================
# 1. Stage 1 & 2 Tests: PDF Extraction and Chunking
# ============================================================================

def test_pdf_extraction_page_numbering():
    """Verify page-by-page extraction and 1-based page numbering."""
    chunks = process_pdf(DUMMY_PDF)
    assert len(chunks) >= 2

    pages = {chunk.metadata["page"] for chunk in chunks}
    assert 1 in pages
    assert 2 in pages
    assert all(chunk.metadata["source"] == DUMMY_PDF for chunk in chunks)
    assert any("Paris" in chunk.page_content for chunk in chunks)
    assert any("Euro" in chunk.page_content for chunk in chunks)


def test_chunk_size_and_overlap_bounds():
    """Verify that chunks do not exceed chunk_size and overlap preserves context."""
    sample_text = (
        "Artificial intelligence is transforming scientific discovery. "
        "Retrieval Augmented Generation enables LLMs to access private documents. "
        "Dense vector embeddings represent the semantic meaning of text mathematically. "
        "ChromaDB stores these vectors efficiently for similarity search."
    )
    chunk_size = 80
    chunk_overlap = 25

    chunks = split_text(sample_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk) <= chunk_size, f"Chunk exceeded {chunk_size} chars: {len(chunk)}"

    # Verify overlap exists between consecutive chunks
    overlap_found = False
    for i in range(1, len(chunks)):
        words_prev = set(chunks[i - 1].split()[-3:])
        words_curr = set(chunks[i].split()[:3])
        if words_prev.intersection(words_curr):
            overlap_found = True
            break
    assert overlap_found, "Expected word overlap between consecutive chunks."


def test_pdf_extraction_scanned_ocr_detection():
    """Verify that text-free / scanned PDFs raise ValueError noting OCR requirement."""
    with pytest.raises(ValueError) as excinfo:
        process_pdf(EMPTY_TEXT_PDF)
    assert "OCR" in str(excinfo.value)


def test_pdf_extraction_corrupt_file():
    """Verify that invalid/corrupt files missing %PDF- header are rejected."""
    with pytest.raises(ValueError) as excinfo:
        process_pdf(CORRUPT_PDF)
    assert "header" in str(excinfo.value).lower() or "magic" in str(excinfo.value).lower()


# ============================================================================
# 2. Stage 3 Tests: HuggingFace Embeddings
# ============================================================================

def test_huggingface_embeddings_configuration():
    """Verify explicit HuggingFace SentenceTransformer model and 384 dimensions."""
    emb_fn = get_embeddings()
    assert emb_fn.model_name == MODEL_NAME
    assert emb_fn.dimension == EMBEDDING_DIM == 384

    # Test single query embedding
    q_vec = emb_fn.embed_query("What is the capital of France?")
    assert isinstance(q_vec, list)
    assert len(q_vec) == 384

    # Test batch document embedding
    docs = ["DocMind AI PDF Assistant", "ChromaDB vector search"]
    d_vecs = emb_fn.embed_documents(docs)
    assert len(d_vecs) == 2
    assert len(d_vecs[0]) == 384
    assert len(d_vecs[1]) == 384


# ============================================================================
# 3. Stage 4 & 5 Tests: ChromaDB Vector Storage and Retrieval
# ============================================================================

def test_vector_store_persistence_and_stable_ids():
    """Verify storage, stable SHA-256 IDs, and retrieval."""
    clear_db()
    assert get_document_count() == 0

    chunks = process_pdf(DUMMY_PDF)
    count = add_documents_to_db(chunks)
    assert count == len(chunks)
    assert get_document_count() == count

    # Verify ID determinism
    id1 = generate_stable_chunk_id(chunks[0], 0)
    id2 = generate_stable_chunk_id(chunks[0], 0)
    assert id1 == id2
    assert DUMMY_PDF in id1

    # Retrieve top-k
    results = query_similar_documents("capital of France", k=4)
    assert len(results) > 0
    assert len(results) <= 4

    # Verify top result is from page 1 mentioning Paris
    top_doc = results[0]
    assert "Paris" in top_doc.page_content
    assert top_doc.metadata["page"] == 1
    assert top_doc.metadata["source"] == DUMMY_PDF
    assert "similarity_score" in top_doc.metadata
    assert "distance" in top_doc.metadata


def test_vector_store_idempotent_reupload():
    """Re-uploading the same PDF should purge old chunks and avoid duplicate records."""
    clear_db()
    chunks = process_pdf(DUMMY_PDF)
    initial_count = add_documents_to_db(chunks)

    # Re-add identical document
    second_count = add_documents_to_db(chunks)
    assert second_count == initial_count
    # Database count must not double
    assert get_document_count() == initial_count


def test_multi_document_coexistence():
    """Uploading two different PDFs should preserve separate sources and page numbers."""
    clear_db()
    chunks1 = process_pdf(DUMMY_PDF)
    chunks2 = process_pdf(DUMMY_PDF_2)

    add_documents_to_db(chunks1)
    add_documents_to_db(chunks2)

    sources = get_indexed_sources()
    assert DUMMY_PDF in sources
    assert DUMMY_PDF_2 in sources

    # Query for doc 2 content
    res_mars = query_similar_documents("Which planet is the Red Planet?", k=2)
    assert len(res_mars) > 0
    assert "Mars" in res_mars[0].page_content
    assert res_mars[0].metadata["source"] == DUMMY_PDF_2


# ============================================================================
# 4. Stage 6 Tests: Grounding Prompts and Groq LLM Inference
# ============================================================================

def test_grounding_prompt_and_injection_resistance():
    """Verify prompt delimiters and untrusted data instructions."""
    context = "DocMind is built with Python and FastAPI."
    question = "What is DocMind built with?"
    prompt = format_prompt(context, question)

    assert "<document_context>" in prompt
    assert "</document_context>" in prompt
    assert context in prompt
    assert question in prompt

    # Verify system instruction contains injection resistance
    assert "UNTRUSTED DATA" in SYSTEM_INSTRUCTION
    assert "IGNORE" in SYSTEM_INSTRUCTION


def test_rag_chain_with_mock_groq():
    """Verify end-to-end answer generation with mock Groq client."""
    clear_db()
    chunks = process_pdf(DUMMY_PDF)
    add_documents_to_db(chunks)

    mock_groq = MockGroqClient(response_text="According to the document, the capital of France is Paris.")
    result = generate_answer("What is the capital of France?", k=4, groq_client=mock_groq)

    assert "Paris" in result["answer"]
    assert len(result["sources"]) > 0
    assert result["sources"][0]["page"] == 1
    assert result["sources"][0]["source"] == DUMMY_PDF


def test_rag_chain_abstention_no_sources():
    """When the model abstains, sources list must be empty to avoid false citations."""
    clear_db()
    chunks = process_pdf(DUMMY_PDF)
    add_documents_to_db(chunks)

    # Mock model returning standard abstention string
    mock_groq = MockGroqClient(response_text=ABSTENTION_MESSAGE)
    result = generate_answer("What is the speed of light?", k=4, groq_client=mock_groq)

    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["sources"] == [], "Abstaining response must not cite sources."


def test_rag_chain_empty_vector_store():
    """Querying before any document is uploaded should return a clear prompt."""
    clear_db()
    result = generate_answer("What is France?")
    assert "No documents have been uploaded yet" in result["answer"]
    assert result["sources"] == []


# ============================================================================
# 5. FastAPI REST API Endpoint Tests
# ============================================================================

def test_api_health_endpoint():
    """Test GET /health returns 200 and index metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_chunks" in data
    assert "indexed_documents" in data


def test_api_upload_valid_pdf():
    """Test POST /upload with valid PDF."""
    clear_db()
    with open(DUMMY_PDF, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": (DUMMY_PDF, f, "application/pdf")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "successfully" in data["message"]
    assert data["filename"] == DUMMY_PDF
    assert data["chunks_indexed"] >= 2


def test_api_upload_invalid_non_pdf():
    """Test POST /upload rejects non-PDF extension."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_api_upload_corrupt_file():
    """Test POST /upload rejects files missing %PDF- header."""
    with open(CORRUPT_PDF, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("corrupt.pdf", f, "application/pdf")}
        )
    assert response.status_code == 400
    assert "header" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_api_ask_empty_question():
    """Test POST /ask rejects empty questions with 400."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_api_clear_endpoint():
    """Test POST /clear resets the vector database."""
    response = client.post("/clear")
    assert response.status_code == 200
    assert "cleared" in response.json()["message"]
    assert get_document_count() == 0
