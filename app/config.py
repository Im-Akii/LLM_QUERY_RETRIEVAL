# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# Service Keys
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-llm-index")

# Authentication
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "1ebe8ffd9bdee2a4afda8a2efaa062ad75ac58c62a46422a5c3edb097520bdd6")

# Model & API Info
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"
EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"
# The BAAI/bge-multilingual-gemma2 model has a dimension of 3584
EMBEDDING_DIMENSION = 3584
LLM_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct"