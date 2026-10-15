"""Interactive diagnostic utility that prints Chroma's top three matching chunks."""

import ssl_bootstrap

from langchain_community.vectorstores import Chroma
from local_embeddings import get_embeddings
from rag_config import COLLECTION_NAME, PERSIST_DIRECTORY

# Reuse the embedding model and collection populated by ingest.py.
embeddings = get_embeddings()

# Connect this diagnostic utility to the same persisted collection as the RAG app.
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

while True:
    # Retrieve and print raw matching chunks to inspect Chroma's results.
    question = input("\nAsk a question: ")

    # This utility inspects raw vector-search results; it does not call the chat LLM.
    docs = db.similarity_search(
        question,
        k=3
    )

    print("\nRelevant Content:\n")

    for doc in docs:
        print(doc.page_content)
        print("-" * 50)
