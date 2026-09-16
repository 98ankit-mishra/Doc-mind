import os
import pytest
from fastapi.testclient import TestClient
from main import app
from src.pdf_processor import process_pdf
from reportlab.pdfgen import canvas

client = TestClient(app)

DUMMY_PDF = "test_dummy.pdf"

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    # Create a dummy PDF for testing
    c = canvas.Canvas(DUMMY_PDF)
    c.drawString(100, 750, "This is a test document for DocMind RAG testing.")
    c.drawString(100, 730, "It contains some facts like the capital of France is Paris.")
    c.save()
    
    yield
    
    # Teardown
    if os.path.exists(DUMMY_PDF):
        os.remove(DUMMY_PDF)
    if os.path.exists("chroma_db"):
        import shutil
        shutil.rmtree("chroma_db", ignore_errors=True)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_pdf_processing():
    chunks = process_pdf(DUMMY_PDF)
    assert len(chunks) > 0
    assert "DocMind" in chunks[0].page_content

def test_upload_endpoint():
    with open(DUMMY_PDF, "rb") as f:
        response = client.post("/upload", files={"file": ("test.pdf", f, "application/pdf")})
    assert response.status_code == 200
    assert "successfully" in response.json()["message"]
