"""Shared settings for Chroma persistence and local Ollama model endpoints."""

# Ingestion writes this collection; rag.py opens the same path and collection.
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "rag_documents"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
# This embedding model must stay consistent between ingestion and query-time search.
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"