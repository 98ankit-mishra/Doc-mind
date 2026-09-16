import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables (GROQ_API_KEY)
load_dotenv()

from src.pdf_processor import process_pdf
from src.vector_store import add_documents_to_db, clear_db
from src.rag_chain import generate_answer

app = FastAPI(title="DocMind - AI PDF Research Assistant")

# CORS middleware for allowing frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF, extracts text, chunks it, and stores in ChromaDB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        # Process PDF (chunking)
        chunks = process_pdf(file_path)
        
        # Store in Vector DB
        add_documents_to_db(chunks)
        
        return JSONResponse(content={
            "message": "PDF processed successfully",
            "filename": file.filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Answers a question based on the uploaded PDF chunks.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set.")
        
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        response = generate_answer(request.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
def clear_session():
    """
    Clears the ChromaDB vector store.
    """
    try:
        clear_db()
        return {"message": "Vector database cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount templates as static for simple serving (or use Jinja2Templates, but static is easier for simple HTML)
app.mount("/", StaticFiles(directory="templates", html=True), name="templates")
