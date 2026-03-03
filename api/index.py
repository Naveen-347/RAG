"""
RAG Application — Vercel Serverless Entry Point
Lightweight version using TF-IDF embeddings (no PyTorch/ChromaDB).
Fits within Vercel's 500MB Lambda limit.
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from docx import Document as DocxDocument

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
AI_CAFE_API_KEY = os.getenv("AI_CAFE_API_KEY", "")
AI_CAFE_API_VERSION = "2024-12-01-preview"

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
STATIC_DIR = BASE_DIR / "static"

# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
def get_llm():
    """Create OpenAI-compatible LLM pointing to HCL AI Cafe."""
    return ChatOpenAI(
        model="gpt-4.1",
        openai_api_key=AI_CAFE_API_KEY,
        openai_api_base=(
            "https://aicafe.hcl.com/AICafeService/api/v1/subscription"
            "/openai/deployments/gpt-4.1"
        ),
        default_headers={"api-key": AI_CAFE_API_KEY},
        default_query={"api-version": AI_CAFE_API_VERSION},
        temperature=0.3,
        max_tokens=1024,
    )


# ──────────────────────────────────────────────
# Lightweight Vector Store (TF-IDF + Cosine Sim)
# ──────────────────────────────────────────────
class TFIDFVectorStore:
    """Lightweight vector store using TF-IDF embeddings.
    No PyTorch, no GPU, no heavy dependencies — perfect for serverless."""

    def __init__(self):
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = None

    def add_documents(self, texts: list[str], metadatas: list[dict]):
        self.documents = texts
        self.metadatas = metadatas
        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not self.documents or self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_indices = similarities.argsort()[-k:][::-1]

        results = []
        for idx in top_k_indices:
            if similarities[idx] > 0.01:  # minimum relevance threshold
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(similarities[idx]),
                })
        return results


# ──────────────────────────────────────────────
# In-memory session store
# ──────────────────────────────────────────────
sessions: dict[str, dict] = {}
vectorstores: dict[str, TFIDFVectorStore] = {}

# ──────────────────────────────────────────────
# Document helpers
# ──────────────────────────────────────────────

def extract_text_from_docx(path: str) -> str:
    """Extract all text from a .docx file."""
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def process_document(file_path: str, session_id: str, filename: str):
    """Load document → split → embed with TF-IDF → store in memory."""
    ext = Path(file_path).suffix.lower()

    # 1. Load text
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        raw_texts = [{"text": p.page_content, "metadata": p.metadata} for p in pages]
    elif ext in (".docx", ".doc"):
        text = extract_text_from_docx(file_path)
        raw_texts = [{"text": text, "metadata": {"source": filename}}]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    all_metas = []
    for item in raw_texts:
        chunks = splitter.split_text(item["text"])
        for chunk in chunks:
            if chunk.strip():
                all_chunks.append(chunk)
                all_metas.append(item["metadata"])

    if not all_chunks:
        raise ValueError("No text could be extracted from the document.")

    # 3. Store in TF-IDF vector store
    store = TFIDFVectorStore()
    store.add_documents(all_chunks, all_metas)
    vectorstores[session_id] = store

    sessions[session_id] = {
        "filename": filename,
        "chunk_count": len(all_chunks),
    }

    return len(all_chunks)


# ──────────────────────────────────────────────
# RAG Chain
# ──────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based on the provided document context.
Use ONLY the information from the context below to answer. If the answer is not in the context, say "I couldn't find this information in the uploaded document."

Context:
{context}

Question: {question}

Answer in a clear, well-structured way. If relevant, quote specific parts of the document.""",
)


def ask_question(session_id: str, question: str) -> dict:
    """Retrieve relevant chunks via TF-IDF → generate answer with LLM."""
    store = vectorstores.get(session_id)
    if store is None:
        raise ValueError("No document uploaded for this session.")

    # Retrieve top-k relevant chunks
    results = store.search(question, k=4)

    if not results:
        return {
            "answer": "I couldn't find relevant information in the uploaded document for your question.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(r["content"] for r in results)

    # Generate answer
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=RAG_PROMPT)
    answer = chain.invoke({"context": context, "question": question})

    sources = [{"content": r["content"][:300], "metadata": r["metadata"]} for r in results]

    return {
        "answer": answer["text"],
        "sources": sources,
    }


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(title="RAG Document Q&A", version="1.0.0")


class AskRequest(BaseModel):
    session_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks: int
    message: str


@app.get("/")
async def serve_ui():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse("<h1>RAG App</h1><p>static/index.html not found</p>", status_code=500)


@app.get("/static/{filepath:path}")
async def serve_static(filepath: str):
    file_path = STATIC_DIR / filepath
    if file_path.exists():
        return FileResponse(str(file_path))
    raise HTTPException(404, "File not found")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), session_id: Optional[str] = None):
    if not file.filename:
        raise HTTPException(400, "No file provided.")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(400, f"Unsupported file type '{ext}'. Upload a PDF or Word document.")

    sid = session_id or str(uuid.uuid4())

    tmp_dir = Path(tempfile.gettempdir())
    save_path = tmp_dir / f"{sid}_{file.filename}"
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        chunk_count = process_document(str(save_path), sid, file.filename)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Error processing document: {str(e)}")
    finally:
        save_path.unlink(missing_ok=True)

    return UploadResponse(
        session_id=sid,
        filename=file.filename,
        chunks=chunk_count,
        message=f"Successfully processed '{file.filename}' into {chunk_count} chunks.",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found. Please upload a document first.")

    try:
        result = ask_question(req.session_id, req.question)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Error generating answer: {str(e)}")
