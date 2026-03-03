"""
RAG Application — PDF/Word Q&A with LangChain + ChromaDB + HCL AI Cafe
"""
import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from docx import Document as DocxDocument

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
load_dotenv()

AI_CAFE_API_KEY = os.getenv("AI_CAFE_API_KEY", "")
AI_CAFE_BASE_URL = (
    "https://aicafe.hcl.com/AICafeService/api/v1/subscription"
    "/openai/deployments/gpt-4.1/chat/completions"
)
AI_CAFE_API_VERSION = "2024-12-01-preview"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# LLM & Embeddings
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


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# ──────────────────────────────────────────────
# In-memory session store  {session_id: collection_name}
# ──────────────────────────────────────────────
sessions: dict[str, dict] = {}

# ──────────────────────────────────────────────
# Document helpers
# ──────────────────────────────────────────────

def extract_text_from_docx(path: str) -> str:
    """Extract all text from a .docx file."""
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def process_document(file_path: str, session_id: str, filename: str):
    """Load document → split → embed → store in ChromaDB."""
    ext = Path(file_path).suffix.lower()

    # 1. Load text
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        raw_texts = pages
    elif ext in (".docx", ".doc"):
        text = extract_text_from_docx(file_path)
        from langchain.schema import Document as LCDoc
        raw_texts = [LCDoc(page_content=text, metadata={"source": filename})]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_texts)

    if not chunks:
        raise ValueError("No text could be extracted from the document.")

    # 3. Store in ChromaDB (persistent)
    collection_name = f"session_{session_id.replace('-', '_')}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )

    sessions[session_id] = {
        "collection": collection_name,
        "filename": filename,
        "chunk_count": len(chunks),
    }

    return len(chunks)


def get_vectorstore(session_id: str):
    """Retrieve the ChromaDB vector store for a session."""
    info = sessions.get(session_id)
    if not info:
        return None
    return Chroma(
        collection_name=info["collection"],
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


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
    """Run RAG: retrieve relevant chunks → generate answer."""
    vectorstore = get_vectorstore(session_id)
    if vectorstore is None:
        raise ValueError("No document uploaded for this session.")

    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    result = qa_chain.invoke({"query": question})

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "content": doc.page_content[:300],
            "metadata": doc.metadata,
        })

    return {
        "answer": result["result"],
        "sources": sources,
    }


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(title="RAG Document Q&A", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


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
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), session_id: Optional[str] = None):
    # Validate file type
    if not file.filename:
        raise HTTPException(400, "No file provided.")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(400, f"Unsupported file type '{ext}'. Upload a PDF or Word document.")

    # Create or reuse session
    sid = session_id or str(uuid.uuid4())

    # Save file to disk
    save_path = UPLOAD_DIR / f"{sid}_{file.filename}"
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        chunk_count = process_document(str(save_path), sid, file.filename)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Error processing document: {str(e)}")

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


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🚀  RAG Server starting at http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
