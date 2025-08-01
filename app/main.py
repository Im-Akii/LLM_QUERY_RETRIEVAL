# main.py

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import fitz  # PyMuPDF
import uuid
from typing import List

# Import services and config
from app.config import BEARER_TOKEN
from app.services.document_loader import load_document
from app.services.embedding import embed_texts
from app.services.vector_db import initialize_pinecone, upsert_vectors, query_vectors
from app.services.llm import answer_question

# Initialize Pinecone on startup
initialize_pinecone()

# --- FastAPI App ---
app = FastAPI(
    title="HackRX Document QA API",
    version="1.0.0",
    description="An API to ask questions about a document."
)

# --- Pydantic Models for Request and Response ---
# This model matches the exact format from your screenshot
class RequestFormat(BaseModel):
    documents: str
    questions: List[str]

# This model ensures the response follows the required format
class ResponseFormat(BaseModel):
    answers: List[str]


# --- Authentication ---
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token from Authorization header."""
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing Bearer token")
    return credentials.credentials


# --- Helper Functions ---
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]


# --- Main API Endpoint ---
@app.post("/api/hackrx/run", response_model=ResponseFormat)
async def run_query(payload: RequestFormat, token: str = Depends(verify_token)):
    """
    This endpoint processes a document and answers questions about it in a single call.
    """
    try:
        # 1. Load and Extract Text from the Document
        pdf_bytes = load_document(payload.documents)
        text = extract_text_from_pdf(pdf_bytes)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in document.")

        # 2. Chunk Text and Index in Pinecone
        chunks = chunk_text(text)
        document_id = str(uuid.uuid4()) # Generate a unique ID for this one-time process
        embeddings = embed_texts(chunks)
        upsert_vectors(document_id, chunks, embeddings)

        # 3. Process Each Question
        final_answers = []
        for question in payload.questions:
            # Embed the question
            query_embedding = embed_texts(question)[0]
            
            # Query Pinecone for relevant context
            context_chunks = query_vectors(query_embedding, top_k=3)
            context = "\n---\n".join(context_chunks)
            
            # Get the final answer from the LLM
            answer = answer_question(question=question, context=context)
            final_answers.append(answer)

        return {"answers": final_answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))