# DocMind – AI PDF Research Assistant

DocMind is a beginner-friendly web application that allows users to upload a PDF document and ask questions about its content. It is built using a **RAG (Retrieval-Augmented Generation)** architecture, demonstrating core AI/ML engineering concepts suitable for a Software Engineer interview.

## 1. Project Overview
The application takes a user's PDF, extracts its text, and stores it in a searchable local database. When the user asks a question, the application finds the most relevant sections of the document and sends them to an LLM (Large Language Model) to generate a precise, context-aware answer.

## 2. Features
- **PDF Upload:** Processes and chunks PDF documents.
- **AI Question Answering:** Answers questions based strictly on the uploaded document.
- **Source Tracking:** Shows exactly which page the answer came from.
- **Anti-Hallucination Guardrails:** Refuses to answer if the information is not in the document.
- **Local Vector Database:** Uses ChromaDB for fast, persistent, and local similarity search.
- **Free Local Embeddings:** Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) meaning no cost for vectorization.

## 3. Technology Stack
- **Language:** Python
- **Backend Framework:** FastAPI
- **Frontend:** HTML / CSS / JavaScript
- **LLM API:** Groq API
- **Embeddings:** HuggingFace Local
- **Vector Database:** ChromaDB

---

## What are the core concepts?

### 4. What is RAG?
**RAG (Retrieval-Augmented Generation)** is a technique that gives an LLM access to external knowledge (like your specific PDF) that it wasn't trained on. Instead of relying on the LLM's memory, RAG searches for relevant information in your document and provides it to the LLM as context before it answers the question.

### 5. What is ChromaDB?
ChromaDB is a Vector Database. It is designed to store and quickly search through mathematical representations of text (embeddings). 

### 6. What are Embeddings?
Embeddings are arrays of numbers (vectors) that represent the semantic meaning of text. Words or sentences with similar meanings will have vectors that are mathematically close to each other.

### 7. Why ChromaDB is used
Instead of scanning the entire PDF word-by-word (which is slow and expensive for an LLM), we convert the PDF into embeddings and store them in ChromaDB. When a question is asked, ChromaDB performs a "similarity search" to instantly find the chunks of text most related to the question. We used ChromaDB because it is open-source, runs locally, and is highly efficient for similarity search.

### 8. How the RAG Pipeline Works
The pipeline acts as the engine for our application. It handles:
- Loading the PDF and extracting the text
- Splitting the text into manageable chunks
- Managing embeddings and storing them in ChromaDB
- Creating the prompt template
- Connecting the retriever and the LLM together seamlessly

### 9. How Groq is used
Groq is the inference engine (the brain) that runs the LLM (like Llama 3). We use Groq because its specialized hardware (LPUs) generates text incredibly fast.

### 10. Complete RAG Architecture
```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
HuggingFace Embeddings (Convert text to numbers)
    ↓
ChromaDB (Store vectors locally)
    ↓
Question Asked
    ↓
Retriever (Search ChromaDB for similar chunks)
    ↓
Context Built (Combine chunks into a string)
    ↓
Groq LLM (Prompt + Context + Question)
    ↓
Final Answer + Source Pages
```

---

## 11. Project Structure
```text
docmind/
├── main.py                 # FastAPI backend and endpoints
├── requirements.txt        # Python dependencies
├── .env.example            # Example env variables
├── src/                    # Core RAG logic
│   ├── pdf_processor.py    # Loads and splits PDF
│   ├── embeddings.py       # Initializes HuggingFace embeddings
│   ├── vector_store.py     # ChromaDB logic
│   ├── rag_chain.py        # Connects retriever to LLM
│   └── prompts.py          # Strict LLM instructions
├── templates/              
│   └── index.html          # Frontend HTML
├── static/
│   ├── style.css           # Frontend CSS
│   └── script.js           # Frontend logic
└── data/uploads/           # Stores uploaded PDFs
```

---

## Getting Started

### 12. Installation & Virtual Environment Setup
1. Open your terminal.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 13. Environment Variable Setup
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Get a free API key from [Groq Cloud](https://console.groq.com/).
3. Add your key to the `.env` file:
   ```env
   GROQ_API_KEY=gsk_your_api_key_here
   ```

### 14. How to run FastAPI
Run the application using Uvicorn:
```bash
uvicorn main:app --reload
```
The application will be available at `http://localhost:8000`.

### 15. How to use the application
1. Open `http://localhost:8000` in your browser.
2. Click **Upload & Process** and select a PDF.
3. Wait for the success message.
4. Type a question in the chat box and click **Ask**.

### 16. API Endpoints
- `GET /` - Serves the frontend UI.
- `GET /health` - Health check status.
- `POST /upload` - Expects a multipart form data file. Processes and stores the PDF.
- `POST /ask` - Expects JSON `{"question": "..."}`. Returns the answer and sources.
- `POST /clear` - Clears the vector database.

### 17. Example questions
If you upload an employee handbook:
- "What is the work from home policy?"
- "How many vacation days do I get?"
- "What is the capital of France?" *(The AI should refuse to answer this!)*

### 18. Common Errors
- **`GROQ_API_KEY is not set`**: You forgot to create the `.env` file or missed the key.
- **SQLite3 Version Error**: ChromaDB requires a newer version of SQLite. If you see this on older systems, you may need to install `pysqlite3-binary`.
- **HuggingFace Download Error**: The first time you run the app, it downloads the embedding model. Ensure you have internet access.

### 19. Future Improvements
- Add persistent user chat history using a database like PostgreSQL.
- Support for other file types (docx, txt, csv).
- Add citations directly inline with the text.

---

## Interview Questions I Should Be Able To Answer

**1. What is RAG?**
Retrieval-Augmented Generation is a framework that improves LLM responses by fetching factual, relevant information from an external database (like our PDF) and providing it to the LLM before it answers.

**2. Why did you use RAG?**
LLMs are trained on general data up to a certain date. They do not know about private documents (like a user's PDF). RAG allows us to query private data securely and cheaply without retraining the model.

**3. What are embeddings?**
Embeddings are numerical representations (vectors) of text. They capture the semantic meaning of words. 

**4. Why do we need a vector database?**
Standard databases match exact keywords. Vector databases match *meaning*. We need a vector database to perform "similarity searches" to quickly find chunks of text that mean the same thing as the user's question, even if the exact words are different.

**5. What is ChromaDB and why did you choose it?**
ChromaDB is an open-source, AI-native vector database. I chose it because it is lightweight, runs locally (no complex cloud setup required), and is perfect for building beginner-friendly RAG pipelines.

**6. What is a retriever?**
A retriever is the component that takes the user's question, converts it to an embedding, searches the vector database, and returns the top `K` most relevant document chunks.

**7. Why do we split documents into chunks?**
LLMs have a "context window" limit (they can only read a certain amount of text at once). If we pass a 100-page PDF to an LLM, it will crash or forget information. Chunking ensures we only send the most relevant paragraphs.

**8. What is chunk overlap?**
When we split a document, a sentence might get cut in half. Overlap ensures that chunk 2 starts slightly before chunk 1 ends, preserving context across boundaries.

**9. Why did you use Groq?**
Groq provides exceptionally fast inference for open-source models like Llama 3 via their custom LPU hardware.

**10. What is an LLM?**
A Large Language Model is an AI trained on massive amounts of text to predict the next word in a sequence, allowing it to understand and generate human-like text.

**11. What is hallucination and how does your project reduce it?**
Hallucination is when an LLM confidently makes up false information. This project reduces it by providing strict ground-truth context (the PDF) and using a prompt that explicitly says: "Answer ONLY using the context. If you don't know, say so. Do not invent facts."

**12. RAG vs fine-tuning?**
Fine-tuning trains the model's internal weights on new data (expensive, slow, hard to update). RAG keeps the model static and provides the new data in the prompt (cheap, fast, easy to update/delete data).

**13. What happens when a user asks a question?**
1. Question is converted to a vector embedding.
2. ChromaDB finds similar vectors (chunks) from the PDF.
3. We combine the chunks into a "Context" string.
4. We send the Prompt + Context + Question to Groq.
5. Groq returns the answer.

**14. What is FastAPI and why did you use it?**
FastAPI is a modern, fast web framework for Python. I used it because it is easy to read, automatically handles JSON parsing, and is the industry standard for serving Python AI models. 

**15. How does the frontend communicate with FastAPI?**
The frontend HTML uses JavaScript (`fetch` API) to send HTTP requests (`POST /upload`, `POST /ask`) to the FastAPI backend, which processes the request and returns JSON responses.
