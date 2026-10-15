"""Ollama HTTP adapter used to embed indexed documents and search questions."""

from __future__ import annotations

import logging
import os

import requests
from langchain_core.embeddings import Embeddings

from rag_config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class OllamaEmbeddings(Embeddings):
    """Implement LangChain's embedding interface using Ollama's HTTP API."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        """Configure the Ollama embedding model and API endpoint."""
        self.model = model or os.environ.get("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL)
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        ).rstrip("/")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings for text batches and validate Ollama's response."""
        if not texts:
            return []

        logger.debug("Requesting %d embedding(s) from Ollama model %s.", len(texts), self.model)
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=300,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            logger.exception("Ollama embedding request failed for model %s.", self.model)
            raise

        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != len(texts)
            or any(
                not isinstance(vector, list)
                or not vector
                or any(not isinstance(value, (int, float)) for value in vector)
                for vector in embeddings
            )
        ):
            logger.error("Ollama returned an invalid embedding response for model %s.", self.model)
            raise RuntimeError(
                f"Ollama returned invalid embeddings for model {self.model!r}."
            )

        logger.debug("Received %d embedding(s) from Ollama model %s.", len(embeddings), self.model)
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks when Chroma builds or updates the index."""
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a user query for vector-similarity search in Chroma."""
        return self._encode([text])[0]


def get_embeddings() -> Embeddings:
    """Create the shared embedding adapter configured for this application."""
    return OllamaEmbeddings()