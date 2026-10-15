"""Interactive RAG entry point: initialize services, select mode, and answer questions."""

import logging
import os

import requests
import ssl_bootstrap
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from generative_rag import GenerativeRAG
from local_embeddings import get_embeddings
from rag_config import COLLECTION_NAME, OLLAMA_EMBEDDING_MODEL, PERSIST_DIRECTORY
from qa_engine import QAEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def check_llm_requirements(
    chat_model: str,
    embedding_model: str,
    base_url: str,
    embeddings: Embeddings,
) -> tuple[bool, str]:
    """Verify Ollama, the chat model, and the embedding endpoint before LLM mode."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        return False, f"Ollama is unavailable: {error}"
    except ValueError as error:
        return False, f"Ollama returned an invalid model list: {error}"

    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return False, "Ollama returned an invalid model list."

    installed_models = {
        name.casefold()
        for model_info in payload["models"]
        if isinstance(model_info, dict)
        for name in (model_info.get("name"), model_info.get("model"))
        if isinstance(name, str)
    }

    def is_installed(model: str) -> bool:
        """Match a requested model against Ollama's installed names and tags."""
        model_name = model.casefold()
        candidates = {model_name}
        if ":" not in model_name:
            candidates.add(f"{model_name}:latest")
        return bool(candidates & installed_models)

    missing_models = [
        model
        for model in (chat_model, embedding_model)
        if not is_installed(model)
    ]
    if missing_models:
        return False, f"Required Ollama model(s) not found: {', '.join(missing_models)}"

    try:
        embeddings.embed_query("RAG startup embedding availability check")
    except (requests.RequestException, RuntimeError) as error:
        return False, f"Ollama embedding model {embedding_model!r} is unavailable: {error}"

    return True, "Ollama chat and embedding models are available."

# Initialize the shared embedding function and open the collection made by ingest.py.
embeddings = get_embeddings()

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

retriever = QAEngine(db)
engine = GenerativeRAG(retriever)
embedding_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL)

# Startup checks select LLM mode only when Ollama and both configured models are ready.
USE_LLM, llm_status = check_llm_requirements(
    chat_model=engine.model,
    embedding_model=embedding_model,
    base_url=engine.base_url,
    embeddings=embeddings,
)
if USE_LLM:
    logger.info("%s", llm_status)
else:
    logger.warning("%s", llm_status)
print(f"{llm_status} Using {'LLM with extractive fallback' if USE_LLM else 'extractive-only mode'}.")

while True:
    question = input("\nQuestion: ")
    logger.info("Received question (question_chars=%d).", len(question))

    if USE_LLM:
        # GenerativeRAG retrieves context, calls Ollama, and handles runtime fallback.
        answer = engine.answer(question)
        backend = engine.last_backend
    else:
        # Startup checks failed, so use the local section/TF-IDF extractive path.
        answer = retriever.answer(question)
        backend = "Extractive"

    logger.info("Answer completed using backend %s (answer_chars=%d).", backend, len(answer))
    print(f"\nAnswer [{backend}]:")
    print(answer)
