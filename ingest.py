"""Load local sources, split them into chunks, embed them, and rebuild Chroma."""

import os
import logging
from html.parser import HTMLParser

import ssl_bootstrap
import requests
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from local_embeddings import get_embeddings
from rag_config import COLLECTION_NAME, PERSIST_DIRECTORY


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class HTMLTextExtractor(HTMLParser):
    """Extract headings and readable text blocks from article/main HTML content."""

    def __init__(self) -> None:
        """Initialize parser state for titles and article/main text blocks."""
        super().__init__()
        self.blocks: list[tuple[str, str]] = []
        self.title = ""
        self._tag_stack: list[str] = []
        self._capture_title = False
        self._container_stack: list[str] = []
        self._block_tags = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}
        self._current_block_tag: str | None = None
        self._current_block_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track HTML nesting and begin collecting relevant content elements."""
        self._tag_stack.append(tag)
        if tag == "title":
            self._capture_title = True
        if tag in {"article", "main"}:
            self._container_stack.append(tag)
        if self._container_stack and tag in self._block_tags and self._current_block_tag is None:
            self._current_block_tag = tag
            self._current_block_parts = []

    def handle_endtag(self, tag: str) -> None:
        """Finish captured blocks and update the parser's nesting state."""
        if tag == "title":
            self._capture_title = False
        if self._current_block_tag == tag:
            text = " ".join(part for part in self._current_block_parts if part.strip()).strip()
            if text:
                self.blocks.append((tag, text))
            self._current_block_tag = None
            self._current_block_parts = []
        if tag in {"article", "main"} and self._container_stack:
            self._container_stack.pop()
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        """Capture title and block text while ignoring scripts and page chrome."""
        text = data.strip()
        if not text:
            return

        if self._tag_stack and self._tag_stack[-1] in {"script", "style", "noscript"}:
            return

        if not self._container_stack and not self._capture_title:
            return

        if self._capture_title and not self.title:
            self.title = text

        if self._current_block_tag is not None:
            self._current_block_parts.append(text)


def extract_html_content(html: str) -> tuple[str, list[tuple[str, str]]]:
    """Parse an HTML page into its title and normalized article text blocks."""
    parser = HTMLTextExtractor()
    parser.feed(html)
    parser.close()

    normalized_blocks: list[tuple[str, str]] = []
    for tag, chunk in parser.blocks:
        cleaned = " ".join(chunk.split())
        if cleaned:
            normalized_blocks.append((tag, cleaned))

    return parser.title, normalized_blocks


def split_url_document(url: str, title: str, blocks: list[tuple[str, str]], source_order: int) -> tuple[list[Document], int]:
    """Group URL text blocks under headings and attach source-order metadata."""
    documents: list[Document] = []
    current_heading = title or url
    current_lines: list[str] = []
    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    skipped_headings = {
        "share",
        "table of contents",
        "related resources",
        "related articles",
        "additional links",
        "cookie preferences",
    }

    def flush_section() -> None:
        """Persist the current non-empty heading section and advance its order."""
        nonlocal source_order, current_lines, current_heading, documents
        body = "\n".join(line for line in current_lines if line.strip()).strip()
        if not body:
            current_lines = []
            return

        content = f"{current_heading}\n{body}".strip()
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": url,
                    "source_type": "url",
                    "title": title or url,
                    "source_file": os.path.basename(url_list_path),
                    "source_order": source_order,
                },
            )
        )
        source_order += 1
        current_lines = []

    for tag, text in blocks:
        lowered = text.lower().strip()

        if tag in heading_tags:
            if lowered in skipped_headings:
                continue

            if current_lines:
                flush_section()
            current_heading = text
            continue

        if lowered in skipped_headings:
            continue

        current_lines.append(text)

    if current_lines:
        flush_section()

    return documents, source_order


def load_url_documents(url_file_path: str, source_order: int) -> tuple[list[Document], int]:
    """Fetch configured URLs, extract their readable sections, and skip bad pages."""
    if not os.path.exists(url_file_path):
        logger.warning("URL list does not exist; skipping URL ingestion: %s", url_file_path)
        return [], source_order

    documents: list[Document] = []
    try:
        with open(url_file_path, "r", encoding="utf-8") as handle:
            urls = [
                line.strip()
                for line in handle
                if line.strip() and not line.strip().startswith("#")
            ]
    except OSError:
        logger.exception("Could not read URL list: %s", url_file_path)
        raise
    logger.info("URL ingestion found %d URL(s).", len(urls))

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    }

    for url in urls:
        logger.info("Loading URL source: %s", url)
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.RequestException as error:
            logger.warning("Skipping URL after fetch failure: %s (%s)", url, error)
            continue

        title, blocks = extract_html_content(response.text)
        if not blocks:
            logger.warning("Skipping URL with no extractable page content: %s", url)
            continue

        page_documents, source_order = split_url_document(url, title, blocks, source_order)
        if not page_documents:
            logger.warning("Skipping URL with no structured content: %s", url)
            continue

        documents.extend(page_documents)
        logger.info("Extracted %d section document(s) from URL: %s", len(page_documents), url)

    return documents, source_order


def split_text_document(document: Document) -> list[Document]:
    """Split plain text at short colon headings while retaining section metadata."""
    segments: list[Document] = []
    heading = ""
    section_id = -1
    first_line_after_heading = False
    has_labeled_items = False

    for raw_line in document.page_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        is_section_heading = line.endswith(":") and len(line.rstrip(":").split()) <= 5

        if is_section_heading:
            heading = line
            section_id += 1
            first_line_after_heading = True
            has_labeled_items = False
            continue

        if heading and not first_line_after_heading and has_labeled_items and ":" not in line:
            heading = ""

        attach_heading = False
        if heading and first_line_after_heading:
            attach_heading = True
        elif heading and ":" in line:
            attach_heading = True
            has_labeled_items = True

        content = f"{heading} {line}".strip() if attach_heading else line
        metadata = document.metadata.copy()
        if heading:
            metadata.update(
                {
                    "section_id": section_id,
                    "section_heading": heading.rstrip(":"),
                }
            )
        segments.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )
        first_line_after_heading = False

        if heading and not attach_heading and not has_labeled_items:
            heading = ""

    return segments or [document]

logger.info("Starting ingestion from the data directory.")
documents = []
text_segment_order = 0
url_list_path = os.path.join("data", "urls.txt")

# Extract each supported local file before loading configured URLs.
try:
    source_files = os.listdir("data")
except OSError:
    logger.exception("Could not read the source data directory.")
    raise

for file in source_files:

    path = os.path.join("data", file)

    if file == os.path.basename(url_list_path):
        continue

    if file.endswith(".pdf"):
        logger.info("Loading PDF source: %s", file)
        loader = PyPDFLoader(path)
        try:
            loaded_documents = loader.load()
        except Exception:
            logger.exception("Failed to extract PDF source: %s", file)
            raise
        documents.extend(loaded_documents)
        logger.info("Extracted %d page document(s) from PDF: %s", len(loaded_documents), file)

    elif file.endswith(".txt"):
        logger.info("Loading text source: %s", file)
        loader = TextLoader(path)
        try:
            loaded_documents = loader.load()
        except Exception:
            logger.exception("Failed to extract text source: %s", file)
            raise
        for document in loaded_documents:
            for segment in split_text_document(document):
                segment.metadata = {
                    **segment.metadata,
                    "source_file": file,
                    "source_order": text_segment_order,
                }
                text_segment_order += 1
                documents.append(segment)

# Read data from URL
url_documents, text_segment_order = load_url_documents(url_list_path, text_segment_order)
documents.extend(url_documents)

logger.info("Extraction complete: %d source document(s) loaded.", len(documents))

# Chunk text with overlap so content at boundaries is less likely to be lost.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

logger.info(
    "Chunking complete: %d chunk(s) created (chunk_size=500 characters, overlap=100).",
    len(chunks),
)

# Confirm the embedding endpoint works before replacing the existing index.
embeddings = get_embeddings()
logger.info("Checking Ollama embedding model before indexing.")
embeddings.embed_query("Ollama embedding model availability check")
logger.info("Embedding model check succeeded.")

if os.path.exists(PERSIST_DIRECTORY):
    logger.info("Removing the existing Chroma collection before rebuilding it.")
    existing_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    try:
        existing_db.delete_collection()
    except Exception:
        logger.exception("Could not delete the existing Chroma collection; stopping to avoid stale index data.")
        raise

# Chroma embeds and persists all chunks as part of this operation.
logger.info("Indexing %d chunk(s) into Chroma collection %s.", len(chunks), COLLECTION_NAME)
try:
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )
except Exception:
    logger.exception("Chroma indexing failed.")
    raise

logger.info("Index created successfully: %d chunk(s) in collection %s.", len(chunks), COLLECTION_NAME)
