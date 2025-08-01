# app/services/llm.py

from openai import OpenAI
from app.config import NEBIUS_API_KEY, NEBIUS_BASE_URL, LLM_MODEL

client = OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)

def answer_question(question: str, context: str) -> str:
    """Answers a single question based on the provided context."""
    prompt = f"Answer the question based only on this document context. Answer within 20 words:\n\n{context}\n\nQuestion: {question}"
    
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()