"""
RAG Application — Ultra-lightweight Vercel Serverless Function
No LangChain (too heavy for serverless). Uses openai SDK directly.
HTML is served inline to avoid file path issues on Vercel.
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
AI_CAFE_API_KEY = os.getenv("AI_CAFE_API_KEY", "")
AI_CAFE_BASE_URL = (
    "https://aicafe.hcl.com/AICafeService/api/v1/subscription"
    "/openai/deployments/gpt-4.1"
)
AI_CAFE_API_VERSION = "2024-12-01-preview"

# ──────────────────────────────────────────────
# OpenAI-compatible client for HCL AI Cafe
# ──────────────────────────────────────────────
def get_client():
    return OpenAI(
        api_key=AI_CAFE_API_KEY,
        base_url=AI_CAFE_BASE_URL,
        default_headers={"api-key": AI_CAFE_API_KEY},
        default_query={"api-version": AI_CAFE_API_VERSION},
    )


# ──────────────────────────────────────────────
# Simple text-based retrieval (no heavy ML libs)
# ──────────────────────────────────────────────
import re
from collections import Counter
import math


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


class SimpleRetriever:
    """BM25-like retriever using pure Python — zero heavy dependencies."""

    def __init__(self):
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.doc_tokens: list[list[str]] = []
        self.avg_dl: float = 0
        self.doc_freqs: Counter = Counter()
        self.n_docs: int = 0

    def add_documents(self, texts: list[str], metadatas: list[dict]):
        self.documents = texts
        self.metadatas = metadatas
        self.doc_tokens = [tokenize(t) for t in texts]
        self.n_docs = len(texts)
        if self.n_docs > 0:
            self.avg_dl = sum(len(t) for t in self.doc_tokens) / self.n_docs
        # Document frequency
        self.doc_freqs = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freqs[token] += 1

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not self.documents:
            return []

        query_tokens = tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self.doc_tokens):
            score = 0.0
            dl = len(doc_tokens)
            token_counts = Counter(doc_tokens)

            for qt in query_tokens:
                if qt not in self.doc_freqs:
                    continue
                df = self.doc_freqs[qt]
                tf = token_counts.get(qt, 0)
                # BM25 scoring
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * 2.5) / (tf + 2.5 * (1 - 0.75 + 0.75 * dl / max(self.avg_dl, 1)))
                score += idf * tf_norm

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:k]:
            if score > 0:
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": score,
                })
        return results


# ──────────────────────────────────────────────
# Document parsing (inline, no langchain loaders)
# ──────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> list[dict]:
    """Extract text from PDF using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "metadata": {"source": path, "page": i + 1}})
    return pages


def extract_text_from_docx(path: str) -> list[dict]:
    """Extract text from DOCX."""
    from docx import Document as DocxDocument
    doc = DocxDocument(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": text, "metadata": {"source": path}}] if text else []


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


# ──────────────────────────────────────────────
# In-memory session store
# ──────────────────────────────────────────────
sessions: dict[str, dict] = {}
retrievers: dict[str, SimpleRetriever] = {}


def process_document(file_path: str, session_id: str, filename: str) -> int:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        pages = extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        pages = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported: {ext}")

    if not pages:
        raise ValueError("No text could be extracted.")

    all_chunks = []
    all_metas = []
    for page in pages:
        chunks = split_text(page["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metas.append(page["metadata"])

    retriever = SimpleRetriever()
    retriever.add_documents(all_chunks, all_metas)
    retrievers[session_id] = retriever

    sessions[session_id] = {"filename": filename, "chunk_count": len(all_chunks)}
    return len(all_chunks)


def ask_question(session_id: str, question: str) -> dict:
    retriever = retrievers.get(session_id)
    if not retriever:
        raise ValueError("No document uploaded.")

    results = retriever.search(question, k=4)
    if not results:
        return {"answer": "I couldn't find relevant information in the document.", "sources": []}

    context = "\n\n---\n\n".join(r["content"] for r in results)

    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": (
                "You are a helpful assistant that answers questions based on document context. "
                "Use ONLY the context provided. If the answer is not in the context, say so."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    sources = [{"content": r["content"][:300], "metadata": r["metadata"]} for r in results]
    return {"answer": answer, "sources": sources}


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(title="RAG Document Q&A")


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.get("/")
async def serve_ui():
    html_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    try:
        html_content = html_path.read_text(encoding="utf-8")
        return HTMLResponse(html_content)
    except FileNotFoundError:
        return HTMLResponse(FALLBACK_HTML)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    if not file.filename:
        raise HTTPException(400, "No file provided.")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(400, f"Unsupported file type: {ext}")

    sid = session_id or str(uuid.uuid4())
    save_path = Path(tempfile.gettempdir()) / f"{sid}_{file.filename}"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    try:
        chunk_count = process_document(str(save_path), sid, file.filename)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        save_path.unlink(missing_ok=True)

    return {
        "session_id": sid,
        "filename": file.filename,
        "chunks": chunk_count,
        "message": f"Processed '{file.filename}' into {chunk_count} chunks.",
    }


@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found. Upload a document first.")
    try:
        result = ask_question(req.session_id, req.question)
        return result
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


# ──────────────────────────────────────────────
# Fallback HTML (if static/index.html not found)
# ──────────────────────────────────────────────
FALLBACK_HTML = """<!DOCTYPE html>
<html><head><title>DocuBot</title>
<style>body{font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;background:#0a0e1a;color:#f1f5f9;}
.c{text-align:center;} h1{font-size:2em;margin-bottom:10px;} p{color:#94a3b8;}</style>
</head><body><div class="c"><h1>DocuBot RAG</h1><p>API is running. Static files not found.</p></div></body></html>"""
