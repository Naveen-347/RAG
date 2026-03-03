<div align="center">

# 📄 DocuBot — RAG Document Q&A

**An AI-powered Retrieval-Augmented Generation application that lets you upload PDF or Word documents and ask questions about their content.**

Built with Python · FastAPI · LangChain · ChromaDB · HCL AI Cafe (GPT-4.1)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![Vercel](https://img.shields.io/badge/Deployed_on-Vercel-000000?logo=vercel&logoColor=white)](https://vercel.com)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [RAG Pipeline](#-rag-pipeline)
- [API Reference](#-api-reference)
- [Setup & Installation](#-setup--installation)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [Limitations & Future Improvements](#-limitations--future-improvements)
- [License](#-license)

---

## 🧠 Overview

**DocuBot** is a full-stack RAG (Retrieval-Augmented Generation) application that allows users to:

1. **Upload** PDF (`.pdf`) or Word (`.docx`) documents
2. **Process** them by extracting text, splitting into chunks, and indexing
3. **Ask questions** in natural language and receive AI-generated answers grounded in the document content
4. **View sources** — every answer includes the relevant document excerpts used to generate it

The application comes in **two versions**:

| Version | File | Use Case | Vector Store | LLM Integration |
|---------|------|----------|--------------|-----------------|
| **Local (Full)** | `app.py` | Local development | ChromaDB + HuggingFace Embeddings | LangChain + HCL AI Cafe |
| **Vercel (Lightweight)** | `api/index.py` | Cloud deployment | Pure Python BM25 Retriever | OpenAI SDK + HCL AI Cafe |

---

## 🏗 Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Web Browser (Client)           │
                    │  ┌────────────┐  ┌──────────────────────┐  │
                    │  │  File      │  │  Chat Interface      │  │
                    │  │  Upload    │  │  (Q&A Messages)      │  │
                    │  │  (Drag &   │  │                      │  │
                    │  │   Drop)    │  │  [User Question]     │  │
                    │  └─────┬──────┘  │  [Bot Answer]        │  │
                    │        │         │  [📎 Sources]        │  │
                    │        │         └──────────┬───────────┘  │
                    └────────┼───────────────────┼──────────────┘
                             │                    │
                    ─────────┼────────────────────┼────── HTTP ──
                             │ POST /upload       │ POST /ask
                             ▼                    ▼
                    ┌─────────────────────────────────────────────┐
                    │             FastAPI Backend                  │
                    │                                             │
                    │  ┌──────────────────────────────────────┐   │
                    │  │          Document Processor           │   │
                    │  │                                      │   │
                    │  │  PDF ──► PyPDF ──► Text Extraction   │   │
                    │  │  DOCX ─► python-docx ─► Text Extract │   │
                    │  │                  │                    │   │
                    │  │           Text Splitter               │   │
                    │  │    (1000 chars, 200 overlap)          │   │
                    │  │                  │                    │   │
                    │  │           Vector Store                │   │
                    │  │   Local: ChromaDB + HuggingFace      │   │
                    │  │   Vercel: BM25 (Pure Python)         │   │
                    │  └──────────────────┬───────────────────┘   │
                    │                     │                        │
                    │  ┌──────────────────▼───────────────────┐   │
                    │  │           RAG Chain                   │   │
                    │  │                                      │   │
                    │  │  1. Retrieve top-4 relevant chunks   │   │
                    │  │  2. Build context prompt             │   │
                    │  │  3. Send to LLM (GPT-4.1)           │   │
                    │  │  4. Return answer + sources          │   │
                    │  └──────────────────┬───────────────────┘   │
                    │                     │                        │
                    └─────────────────────┼────────────────────────┘
                                          │
                              ────────────┼──────── HTTPS ──
                                          ▼
                              ┌───────────────────────┐
                              │   HCL AI Cafe API     │
                              │   (GPT-4.1 Model)     │
                              │                       │
                              │   OpenAI-Compatible   │
                              │   Chat Completions    │
                              └───────────────────────┘
```

---

## 🛠 Tech Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core programming language |
| **FastAPI** | ≥ 0.115 | High-performance async web framework for REST API |
| **Uvicorn** | ≥ 0.30 | ASGI server for running FastAPI locally |
| **Pydantic** | v2 | Request/response data validation and serialization |
| **python-dotenv** | ≥ 1.0 | Environment variable loading from `.env` files |

### AI & NLP (Local Version — `app.py`)

| Technology | Version | Purpose |
|------------|---------|---------|
| **LangChain** | ≥ 0.3 | RAG orchestration framework (chains, prompts, splitters) |
| **LangChain-OpenAI** | ≥ 0.2 | OpenAI-compatible LLM integration for HCL AI Cafe |
| **LangChain-HuggingFace** | ≥ 0.1 | Local embedding model integration |
| **LangChain-Community** | ≥ 0.3 | Community integrations (PyPDF loader, Chroma vector store) |
| **ChromaDB** | ≥ 0.5 | Persistent vector database for document embeddings |
| **sentence-transformers** | ≥ 3.0 | HuggingFace models for local text embeddings |
| **HuggingFace Embeddings** | `all-MiniLM-L6-v2` | Embedding model (384-dim, runs locally on CPU, free) |

### AI & NLP (Vercel Version — `api/index.py`)

| Technology | Purpose |
|------------|---------|
| **OpenAI SDK** | Direct API calls to HCL AI Cafe (lightweight, no LangChain) |
| **BM25 Retriever** | Pure Python implementation of Okapi BM25 ranking algorithm |
| **Pure Python** | Zero ML dependencies — no PyTorch, no scikit-learn |

### Document Processing

| Technology | Version | Purpose |
|------------|---------|---------|
| **PyPDF** | ≥ 4.0 | PDF text extraction (page-by-page) |
| **python-docx** | ≥ 1.0 | Microsoft Word (.docx) text extraction |

### Frontend

| Technology | Purpose |
|------------|---------|
| **HTML5** | Semantic page structure |
| **CSS3** | Custom properties, glassmorphism, CSS Grid, animations |
| **Vanilla JavaScript** | Fetch API, DOM manipulation, drag & drop, event handling |
| **Google Fonts (Inter)** | Modern, clean typography |

### LLM Provider

| Provider | Model | API Format |
|----------|-------|------------|
| **HCL AI Cafe** | `gpt-4.1` | OpenAI-compatible Chat Completions |
| **API Version** | `2024-12-01-preview` | Azure-style versioning |
| **Auth** | `api-key` header | Subscription key based |

### Deployment & DevOps

| Tool | Purpose |
|------|---------|
| **Vercel** | Serverless cloud deployment (Python runtime) |
| **Git / GitHub** | Version control and CI/CD trigger |

---

## 📁 Project Structure

```
RAG/
│
├── app.py                    # 🖥  Local development server (full LangChain + ChromaDB)
├── requirements.txt          # 📦 Python dependencies
├── vercel.json               # ⚙️  Vercel deployment configuration
├── .env                      # 🔐 Environment variables (NOT committed to Git)
├── .env.example              # 📄 Template for environment variables
├── .gitignore                # 🚫 Git ignore rules
│
├── api/
│   └── index.py              # ☁️  Vercel serverless function (lightweight, no LangChain)
│
├── static/
│   └── index.html            # 🎨 Web UI (dark theme, chat interface, drag & drop)
│
├── uploads/                  # 📂 Uploaded files (local only, auto-created)
│
└── chroma_db/                # 🗄  ChromaDB persistent storage (local only, auto-created)
```

---

## 🔄 RAG Pipeline

The Retrieval-Augmented Generation pipeline works in two phases:

### Phase 1: Document Ingestion (on upload)

```
PDF / DOCX  ─────►  Text Extraction  ─────►  Text Chunking  ─────►  Indexing
                    (PyPDF / docx)           (1000 chars,           (ChromaDB or
                                              200 overlap)           BM25 index)
```

| Step | Description | Parameters |
|------|-------------|------------|
| **1. Text Extraction** | Extract raw text from PDF pages or Word paragraphs | — |
| **2. Text Splitting** | Split into overlapping chunks for better context retrieval | `chunk_size=1000`, `overlap=200` |
| **3. Embedding** | Convert chunks to vector representations (local version) | `all-MiniLM-L6-v2` (384 dimensions) |
| **4. Indexing** | Store in vector database for fast similarity search | ChromaDB (local) or BM25 (Vercel) |

### Phase 2: Question Answering (on ask)

```
User Question  ─►  Retrieval (top-4 chunks)  ─►  Context Assembly  ─►  LLM Prompt  ─►  Answer
                   (similarity search)              (concatenate)        (GPT-4.1)      + Sources
```

| Step | Description |
|------|-------------|
| **1. Query** | User submits a natural language question |
| **2. Retrieval** | Find the 4 most relevant document chunks (`k=4`) |
| **3. Context Building** | Concatenate retrieved chunks into a context string |
| **4. Prompt Engineering** | Inject context + question into a structured prompt |
| **5. LLM Generation** | GPT-4.1 generates an answer grounded in the context |
| **6. Source Attribution** | Return answer along with the source chunks used |

### Prompt Template

```
You are a helpful assistant that answers questions based on document context.
Use ONLY the context provided. If the answer is not in the context, say so.

Context:
{retrieved_chunks}

Question: {user_question}
```

---

## 📡 API Reference

### `GET /`

Serves the web UI (`static/index.html`).

**Response:** `200 OK` — HTML page

---

### `POST /upload`

Upload a document for processing.

**Content-Type:** `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | ✅ | PDF or DOCX file to upload |
| `session_id` | string | ❌ | Reuse an existing session (optional) |

**Success Response (`200`):**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "report.pdf",
  "chunks": 42,
  "message": "Processed 'report.pdf' into 42 chunks."
}
```

**Error Responses:**

| Code | Condition |
|------|-----------|
| `400` | Unsupported file type (not .pdf, .docx, .doc) |
| `500` | Text extraction or processing failure |

---

### `POST /ask`

Ask a question about the uploaded document.

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "What are the main findings of this report?"
}
```

**Success Response (`200`):**

```json
{
  "answer": "The main findings of the report include...",
  "sources": [
    {
      "content": "First 300 characters of the source chunk...",
      "metadata": { "source": "report.pdf", "page": 3 }
    }
  ]
}
```

**Error Responses:**

| Code | Condition |
|------|-----------|
| `400` | Empty question |
| `404` | Session not found (no document uploaded) |
| `500` | LLM API call failure |

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.10+** installed
- **HCL AI Cafe API Key** (subscription key)
- **Git** for version control

### 1. Clone the Repository

```bash
git clone https://github.com/Naveen-347/RAG.git
cd RAG
```

### 2. Create Virtual Environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For local development, you also need the full LangChain stack:
> ```bash
> pip install langchain langchain-community langchain-openai langchain-huggingface chromadb sentence-transformers
> ```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
AI_CAFE_API_KEY=your-actual-hcl-ai-cafe-subscription-key
```

### 5. Run the Application

```bash
python app.py
```

Open **http://localhost:8000** in your browser.

---

## ☁️ Deployment

### Vercel Deployment

The project includes a Vercel-optimized version (`api/index.py`) designed for serverless environments:

#### Why Two Versions?

| Feature | `app.py` (Local) | `api/index.py` (Vercel) |
|---------|-------------------|--------------------------|
| **Vector Store** | ChromaDB (persistent, ML-based) | BM25 (pure Python, in-memory) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | None needed (BM25 is keyword-based) |
| **LLM Client** | LangChain `ChatOpenAI` | OpenAI SDK directly |
| **Bundle Size** | ~2GB+ (PyTorch, etc.) | ~50MB (lightweight) |
| **Persistence** | ✅ Survives restarts | ❌ Stateless per invocation |

#### Deploy Steps

1. **Push to GitHub** (already done)
2. **Import in Vercel:**
   - Go to [vercel.com](https://vercel.com) → **Add New Project**
   - Import `Naveen-347/RAG` repository
3. **Set Environment Variables:**
   - Go to **Settings → Environment Variables**
   - Add `AI_CAFE_API_KEY` = `your-subscription-key`
4. **Deploy** — Vercel auto-builds and deploys on every push

#### Vercel Configuration (`vercel.json`)

```json
{
  "rewrites": [
    { "source": "/api/(.*)", "destination": "/api/index" },
    { "source": "/(.*)", "destination": "/api/index" }
  ]
}
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AI_CAFE_API_KEY` | ✅ | HCL AI Cafe subscription key for GPT-4.1 access |

### HCL AI Cafe API Configuration

| Setting | Value |
|---------|-------|
| **Endpoint** | `https://aicafe.hcl.com/AICafeService/api/v1/subscription/openai/deployments/gpt-4.1` |
| **Model** | `gpt-4.1` |
| **API Version** | `2024-12-01-preview` |
| **Auth Method** | `api-key` header |
| **Temperature** | `0.3` (focused, factual responses) |
| **Max Tokens** | `1024` |

### Text Splitting Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `chunk_size` | 1000 | Maximum characters per chunk |
| `chunk_overlap` | 200 | Overlapping characters between chunks |
| `separators` | `\n\n`, `\n`, `. `, ` ` | Priority order for split points |

### Retrieval Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k` | 4 | Number of top chunks to retrieve per query |
| BM25 `b` | 0.75 | Length normalization parameter |
| BM25 `k1` | 2.5 | Term frequency saturation parameter |

---

## 🧩 How It Works

### 1. Upload Flow

```
User drops file → Frontend sends POST /upload (multipart)
→ Backend saves to temp → Extracts text (PyPDF/docx)
→ Splits into overlapping chunks → Indexes in vector store
→ Returns session_id + chunk count
```

### 2. Question Flow

```
User types question → Frontend sends POST /ask (JSON)
→ Backend retrieves top-4 relevant chunks from index
→ Builds context prompt with chunks + question
→ Calls GPT-4.1 via HCL AI Cafe API
→ Returns generated answer + source excerpts
```

### 3. Session Management

- Each upload creates a unique `session_id` (UUID v4)
- Session maps to a vector store collection
- Multiple questions can be asked per session
- **Local:** Sessions persist across server restarts (ChromaDB)
- **Vercel:** Sessions are in-memory (lost between cold starts)

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

| Limitation | Details |
|------------|---------|
| **Vercel stateless** | On Vercel, uploaded documents are lost between cold starts |
| **Single document** | Each session supports one document at a time |
| **Text-only** | Does not process images, tables, or charts within documents |
| **No chat history** | Each question is independent (no conversation memory) |

### Potential Improvements

| Feature | Description |
|---------|-------------|
| **Cloud Vector DB** | Use Pinecone or Chroma Cloud for persistent storage |
| **Multi-document** | Support uploading and querying across multiple documents |
| **Chat Memory** | Add conversation history for follow-up questions |
| **Streaming** | Stream LLM responses for real-time typing effect |
| **Table Extraction** | Use `camelot` or `tabula` for structured table data |
| **Authentication** | Add user accounts and per-user document collections |

---

## 📄 License

This project is for educational and internal use. Ensure compliance with HCL AI Cafe's terms of service when using their API.

---

<div align="center">

**Built with ❤️ using Python, FastAPI & LangChain**

</div>
