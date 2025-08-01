# app/services/embedding.py

from openai import OpenAI
from app.config import NEBIUS_API_KEY, NEBIUS_BASE_URL, EMBEDDING_MODEL

client = OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of text chunks using Nebius."""
    if isinstance(texts, str):
        texts = [texts]

    res = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in res.data]