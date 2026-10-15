"""One-shot diagnostic that shows the top five Chroma chunks and their metadata."""

import ssl_bootstrap

from langchain_community.vectorstores import Chroma
from local_embeddings import get_embeddings
from rag_config import COLLECTION_NAME, PERSIST_DIRECTORY

# Connect to the saved collection without running ingestion or generation.
embeddings = get_embeddings()

# Open the existing index so a query can be inspected without running generation.
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

question = input("Question: ")

# Inspect selected passages before generation to diagnose retrieval quality.
docs = db.similarity_search(question, k=5)

# Print each result with its source metadata so ranking issues are traceable.
for i, doc in enumerate(docs, 1):
    print("\n====================")
    print(f"Chunk {i}")
    print("Metadata:", doc.metadata)
    print("\nContent:")
    print(doc.page_content[:1000])
