# app/services/vector_db.py

from pinecone import Pinecone, ServerlessSpec
from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
)

# Define index as a global variable, it will be initialized later
index = None

def initialize_pinecone():
    """Initializes connection to Pinecone and creates the index if it doesn't exist."""
    global index
    
    # Initialize Pinecone connection
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if the index exists and create it if it doesn't
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Index created successfully.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    # Connect to the index and assign it to the global variable
    index = pc.Index(PINECONE_INDEX_NAME)


def upsert_vectors(document_id: str, chunks: list[str], embeddings: list[list[float]], batch_size: int = 100):
    """Upserts document chunks and their embeddings into Pinecone in batches."""
    if index is None:
        raise Exception("Pinecone index has not been initialized. Call initialize_pinecone() first.")
    
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]

        vectors_to_upsert = []
        for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
            vector_id = f"{document_id}-chunk-{i+j}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {"text": chunk}
            })
        
        index.upsert(vectors=vectors_to_upsert)
        print(f"Upserted batch covering chunks {i} to {batch_end-1}")

    print(f"Successfully upserted {len(chunks)} chunks for document {document_id}.")


def query_vectors(query_embedding: list[float], top_k: int = 3) -> list[str]:
    """Queries Pinecone to find the most relevant text chunks."""
    if index is None:
        raise Exception("Pinecone index has not been initialized. Call initialize_pinecone() first.")
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match['metadata']['text'] for match in results['matches']]