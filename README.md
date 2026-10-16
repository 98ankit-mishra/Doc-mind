# DocMind — AI PDF Research Assistant

DocMind is a high-performance, production-quality Retrieval-Augmented Generation (RAG) assistant for PDF document research. It extracts textual knowledge from uploaded PDF documents, creates dense vector embeddings locally using HuggingFace models, stores them persistently in ChromaDB, and performs semantic retrieval coupled with Groq-accelerated LLMs to deliver accurate, grounded answers with source page citations.

Built with **pure Python, FastAPI, HuggingFace Sentence Transformers, ChromaDB, and the official Groq SDK** — completely free of heavy framework bloat (100% LangChain-free architecture) for maximum execution speed, deterministic debugging, and transparent control.

---

## 1. Resume Claim Verification & Tech Stack

| Resume Claim | Implementation Status | Technical Details |
|---|---|---|
| **Six-stage RAG pipeline** | Fully Implemented | Extraction → Chunking → Embedding → Vector Storage → Retrieval → Generation |
| **Automated PDF extraction & chunking** | Fully Implemented | PyPDF page-by-page extraction, sliding window recursive chunker (1,000 chars, 200 overlap) |
| **Dense vector embeddings via HuggingFace** | Fully Implemented | `sentence-transformers/all-MiniLM-L6-v2` producing 384-dimensional normalized dense vectors |
| **Persistent vector storage in ChromaDB** | Fully Implemented | ChromaDB persistent client (`data/chroma_db`) with cosine similarity (`hnsw:space: cosine`) |
| **Semantic retrieval layer** | Fully Implemented | Top-$K$ ($K=4$) nearest neighbor semantic retrieval using matching query embeddings |
| **Groq LLM API integration** | Fully Implemented | Native `groq` SDK client invoking `llama3-8b-8192` with strict JSON-compatible responses |
| **Source-page tracking** | Fully Implemented | 1-based page metadata tracked per chunk and deduplicated across cited context |
| **Prompt-based grounding guardrails** | Fully Implemented | Delimited untrusted context `<document_context>`, injection defense, and strict abstention |

### Technology Stack
- **Backend Language:** Python 3.10+
- **API Framework:** FastAPI, Uvicorn, Pydantic v2
- **Vector Database:** ChromaDB (Persistent client, DuckDB/SQLite & HNSW index)
- **Embedding Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **LLM Inference:** Groq Cloud API (`llama3-8b-8192` via ultra-fast LPUs)
- **Document Processing:** PyPDF
- **Frontend:** Responsive Vanilla HTML5, Modern CSS (Glassmorphism), Vanilla JavaScript (Fetch API)
- **Testing:** Pytest, FastApi TestClient, ReportLab (synthetic test fixtures)

---

## 2. The Six-Stage RAG Pipeline Architecture

```
                    ┌────────────────────────────┐
                    │      Uploaded PDF File     │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: PDF Text Extraction (src/pdf_processor.py)                    │
│ - PyPDF page-by-page extraction                                        │
│ - 1-based page numbering preserved                                     │
│ - Magic bytes (%PDF-) validation & OCR/scanned document detection      │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Document Chunking (src/pdf_processor.py)                      │
│ - Recursive sliding window chunker: chunk_size=1000, chunk_overlap=200 │
│ - Natural paragraph (\n\n), line (\n), and sentence break boundaries   │
│ - Page-level metadata inherited by every chunk                         │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Embedding Generation (src/embeddings.py)                      │
│ - Model: sentence-transformers/all-MiniLM-L6-v2                        │
│ - Generates 384-dimensional dense vectors                              │
│ - L2-normalization applied for cosine distance indexing                │
│ - Local execution on CPU/GPU after one-time initial download           │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 4: ChromaDB Vector Storage (src/vector_store.py)                 │
│ - Persistent storage at data/chroma_db                                 │
│ - Deterministic SHA-256 chunk IDs prevent duplicate inflation          │
│ - Multi-document isolation and source-level idempotent re-indexing     │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 5: Semantic Retrieval (src/vector_store.py)                      │
│ - Query vectorization with identical HuggingFace embedding function    │
│ - Cosine similarity nearest neighbor search retrieves top 4 chunks     │
│ - Extracts text content, source filenames, and 1-based page numbers    │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 6: Grounded Generation (src/prompts.py & src/rag_chain.py)       │
│ - Retreived context wrapped in <document_context> security boundary    │
│ - Groq LPU inference (llama3-8b-8192) with temperature=0.1             │
│ - Strict prompt guardrails: abstains if information is missing         │
│ - Returns final answer and deduplicated unique source page references  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Dive into Implementation Details

### Stage 1: PDF Extraction
- `src/pdf_processor.py` validates the `%PDF-` file header before reading.
- Rejects corrupt or encrypted PDFs with clear, safe error messages.
- Detects image-only / scanned PDFs where extracted text length is negligible, returning an explicit message that OCR is required rather than failing silently.

### Stage 2: Recursive Document Chunking
- Configured with `chunk_size = 1000` characters and `chunk_overlap = 200` characters.
- Splits text hierarchically on paragraph breaks (`\n\n`), newlines (`\n`), sentence endings (`. `, `? `, `! `), and spaces (` `).
- Preserves word integrity: never cuts a token in half.
- Preserves context across chunk boundaries by carrying over an overlap window bounded strictly within `chunk_size`.
- Binds metadata `{ "source": filename, "page": page_number }` to every individual chunk.

### Stage 3: HuggingFace Dense Vector Embeddings
- Implemented in `src/embeddings.py` using `HuggingFaceEmbeddingFunction`.
- Conforms fully to the ChromaDB `EmbeddingFunction` protocol (with `name()`, `get_config()`, `build_from_config()`, `__call__()`, and `embed_query()`).
- Uses `all-MiniLM-L6-v2`:
  - **Vector Dimension:** 384
  - **Sequence Length:** 256 word pieces
  - **L2-Normalized:** Direct dot product equals cosine similarity.
- Singleton pattern ensures thread-safe, lazy model loading in memory once.
- Once downloaded to `~/.cache/huggingface`, embedding inference runs entirely locally on CPU/GPU without external API dependencies or network calls.

### Stage 4: Persistent ChromaDB Vector Storage
- Implemented in `src/vector_store.py`.
- Persists to `data/chroma_db` using SQLite and HNSW vector indexing.
- Generates collision-resistant, deterministic chunk IDs:
  $$\text{ID} = \text{SHA256}(\text{source} + \text{page} + \text{text[:64]} + \text{index})[:16]$$
- Supports multiple documents simultaneously with source-level filtering.
- Re-uploading the same PDF triggers an idempotent update: previous chunks for that document are cleared before new ones are inserted, preventing duplicate inflation.

### Stage 5: Semantic Retrieval
- Retrieves the top $K=4$ most relevant chunks using cosine similarity.
- Binds identical embedding model configuration to query vectors so query and document embeddings exist in the exact same metric space.
- Returns chunk text alongside document metadata (file name and 1-based page numbers).

### Stage 6: Grounded Generation & Guardrails
- Implemented in `src/prompts.py` and `src/rag_chain.py`.
- Passes retrieved text inside delimited XML blocks (`<document_context>...</document_context>`) to treat document content as untrusted data and mitigate prompt injection attacks.
- Explicit system instructions enforce factual grounding:
  - Answers MUST be based exclusively on the provided context.
  - If the context does not contain the answer, the LLM MUST return:
    > *"I could not find this information in the uploaded document."*
  - The model is forbidden from extrapolating or using outside training memory.
- If the model abstains or if no context is found, source citations are suppressed.

---

## 4. Project Directory Structure

```text
DocMind/
├── main.py                     # FastAPI application endpoints (/health, /upload, /ask, /clear)
├── requirements.txt            # Minimal, pinned Python dependencies (Zero LangChain)
├── .env.example                # Template for environment configuration
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py        # Stage 1 & 2: PyPDF extraction and recursive chunking
│   ├── embeddings.py           # Stage 3: HuggingFace SentenceTransformer (all-MiniLM-L6-v2)
│   ├── vector_store.py         # Stage 4 & 5: Persistent ChromaDB client and semantic retriever
│   ├── prompts.py              # Stage 6: Grounding guardrails and prompt templates
│   └── rag_chain.py            # Stage 6: Groq SDK inference and answer pipeline
├── static/
│   ├── style.css               # Modern glassmorphism UI styling
│   └── script.js               # Frontend async fetch logic and UI updates
├── templates/
│   └── index.html              # Clean single-page application UI
├── tests/
│   ├── __init__.py
│   └── test_basic.py           # 18-test automated unit & integration test suite
└── data/
    ├── uploads/                # Local uploaded PDF documents
    └── chroma_db/              # Persistent ChromaDB vector database files
```

---

## 5. Installation and Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- A free Groq Cloud API key from [console.groq.com](https://console.groq.com/)

### Step 1: Clone the Repository
```bash
git clone https://github.com/98ankit-mishra/Doc-mind.git
cd Doc-mind
```

### Step 2: Create and Activate a Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **On Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Open `.env` and paste your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

---

## 6. Running the Application

Start the FastAPI application using Uvicorn:
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

### Using the Web UI
1. **Upload a PDF:** Select any text-based PDF and click **Upload & Process**. The system validates, extracts, chunks, generates 384-dimensional embeddings, and indexes them into ChromaDB.
2. **Ask Questions:** Type your inquiry in the search bar and press **Ask**.
3. **Inspect Citations:** Verify the answer and see the exact page numbers cited.
4. **Clear Session:** Click **Clear Database** to wipe the current vector store and uploaded files.

---

## 7. Running Automated Tests

A comprehensive suite of 18 automated tests covers all pipeline stages, vector persistence, injection resistance, and API endpoints. Unit tests mock Groq inference to run deterministically and fully offline.

Execute pytest with verbose output:
```bash
pytest tests/test_basic.py -v
```

### Test Coverage Highlights
- `test_pdf_extraction_page_numbering`: Verifies 1-based page metadata extraction.
- `test_chunk_size_and_overlap_bounds`: Validates chunk size $\le 1000$ and overlap preservation.
- `test_pdf_extraction_scanned_ocr_detection`: Confirms proper handling of text-free scanned PDFs.
- `test_pdf_extraction_corrupt_file`: Rejects non-PDF files missing magic bytes.
- `test_huggingface_embeddings_configuration`: Validates `all-MiniLM-L6-v2` 384-dimensional vectors.
- `test_vector_store_persistence_and_stable_ids`: Checks SHA-256 ID stability and ChromaDB storage.
- `test_vector_store_idempotent_reupload`: Verifies re-uploading replaces chunks without duplication.
- `test_multi_document_coexistence`: Tests simultaneous multi-PDF isolation.
- `test_grounding_prompt_and_injection_resistance`: Validates prompt boundaries and injection defense.
- `test_rag_chain_with_mock_groq`: Verifies generation with mock LLM client.
- `test_rag_chain_abstention_no_sources`: Confirms source suppression upon abstention.
- `test_rag_chain_empty_vector_store`: Handles queries before documents are uploaded.
- `test_api_health_endpoint`: Checks `/health` endpoint metadata.
- `test_api_upload_valid_pdf`: Tests `/upload` with synthetic PDF fixture.
- `test_api_upload_invalid_non_pdf`: Rejects `.txt` files with 400 Bad Request.
- `test_api_upload_corrupt_file`: Rejects corrupt files with 400 Bad Request.
- `test_api_ask_empty_question`: Validates input rejection for empty questions.
- `test_api_clear_endpoint`: Tests `/clear` database reset.

---

## 8. REST API Reference

### `GET /health`
Returns system status and index metadata.
```json
{
  "status": "healthy",
  "indexed_chunks": 12,
  "indexed_documents": ["annual_report.pdf"]
}
```

### `POST /upload`
Uploads and indexes a PDF document (`multipart/form-data`).
- **Form Field:** `file: UploadFile`
- **Response:**
  ```json
  {
    "message": "PDF processed and indexed successfully",
    "filename": "annual_report.pdf",
    "chunks_indexed": 12
  }
  ```

### `POST /ask`
Queries indexed documents using the six-stage RAG pipeline.
- **Request:**
  ```json
  {
    "question": "What was the company's net profit in 2023?"
  }
  ```
- **Response:**
  ```json
  {
    "answer": "According to the report, the net profit in 2023 was $4.2 million.",
    "sources": [
      {
        "source": "annual_report.pdf",
        "page": 4
      }
    ]
  }
  ```

### `POST /clear`
Resets the ChromaDB vector database and deletes cached upload files.
- **Response:**
  ```json
  {
    "status": "success",
    "message": "Database cleared successfully"
  }
  ```

---

## 9. Technical Interview Guide & Known Limitations

### Common Interview Questions

#### Q1: Why did you eliminate LangChain from the project?
**Answer:** LangChain adds heavy abstraction layers, frequent breaking changes across minor releases, and significant runtime latency. By using native Python with ChromaDB and the official Groq SDK, our architecture has zero unnecessary dependencies, executes faster, enables clear stack traces during debugging, and provides complete control over prompts and chunking algorithms.

#### Q2: What embedding model did you choose and why?
**Answer:** We chose `sentence-transformers/all-MiniLM-L6-v2`. It produces 384-dimensional dense vectors and maps sentences to a dense vector space tuned for semantic similarity and clustering. It achieves high retrieval quality with a lightweight footprint (~80MB), allowing rapid local inference on standard CPUs without incurring third-party API costs or network latency.

#### Q3: How do you prevent duplicate chunks when a user re-uploads the same document?
**Answer:** In `src/vector_store.py`, chunk IDs are generated deterministically using a SHA-256 hash of `source + page + text[:64] + chunk_index`. When a document is re-uploaded, the system first executes an idempotent deletion of all existing records matching `{"source": filename}` before indexing the new chunks.

#### Q4: How does prompt grounding work and can it completely eliminate hallucinations?
**Answer:** The prompt encloses retrieved chunks inside `<document_context>` XML delimiters, instructing the LLM that this is data rather than instructions (defending against prompt injection). The system prompt mandates that answers be derived strictly from the context, requiring the model to output a standardized abstention message if information is absent. While this drastically reduces hallucinations, no prompt technique can mathematically guarantee 100% elimination because autoregressive LLMs sample tokens probabilistically.

---

### Known Limitations & Production Roadmap

1. **Scanned PDFs & OCR:**
   - *Current Behavior:* PyPDF extracts text streams. If a PDF is a scanned image without a text layer, the system detects zero extracted characters and raises a `ValueError` explaining that OCR is required.
   - *Roadmap:* Integrate Tesseract OCR or a cloud vision API for scanned document fallback.

2. **Complex Tables & Multi-Column Layouts:**
   - *Current Behavior:* Standard PDF text extraction reads linear text streams, which can interleave adjacent columns or scramble tabular structures.
   - *Roadmap:* Implement layout-aware chunking (e.g., PDFPlumber or Microsoft Table Transformer).

3. **Fixed Retrieval ($K=4$) vs. Re-ranking:**
   - *Current Behavior:* The retriever selects the top 4 chunks strictly based on cosine similarity.
   - *Roadmap:* Add a cross-encoder re-ranker (e.g., `bge-reranker-base`) to re-score the top 20 candidate chunks for higher semantic precision.

4. **Multi-Turn Chat History:**
   - *Current Behavior:* The current implementation operates on a single-turn question-answering basis.
   - *Roadmap:* Add a conversation buffer memory and query reformulation stage to support follow-up conversational queries.
