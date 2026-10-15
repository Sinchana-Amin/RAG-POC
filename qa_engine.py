"""Section matching and TF-IDF based extractive answer support for RAG."""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

QUESTION_STOPWORDS = {
    "a",
    "about",
    "an",
    "are",
    "did",
    "do",
    "does",
    "explain",
    "how",
    "if",
    "in",
    "is",
    "me",
    "of",
    "list",
    "tell",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
}
logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens for lexical matching."""
    return [token.lower() for token in re.findall(r"\w+", text)]


def query_terms(text: str) -> set[str]:
    """Remove common question words and return meaningful matching terms."""
    return {token for token in tokenize(text) if token not in QUESTION_STOPWORDS and len(token) > 1}


def query_entities(text: str) -> list[str]:
    """Extract acronym-like and title-cased terms for extra retrieval scoring."""
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text)
    entities: list[str] = []

    for index, token in enumerate(tokens):
        lowered = token.lower()
        if lowered in QUESTION_STOPWORDS:
            continue
        if token.isupper() and len(token) >= 2:
            entities.append(token)
        elif index > 0 and token[0].isupper() and token[1:].islower():
            entities.append(token)

    return entities


def split_sentences(text: str) -> list[str]:
    """Split text at sentence-ending punctuation for passage ranking."""
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def build_passages(text: str) -> list[str]:
    """Build single-sentence and adjacent-sentence candidates from a chunk."""
    sentences = split_sentences(text)
    passages: list[str] = []

    for index, sentence in enumerate(sentences):
        passages.append(sentence)
        if index + 1 < len(sentences):
            passages.append(f"{sentence} {sentences[index + 1]}")

    return passages or [text.strip()]


class QAEngine:
    """Provide section matching and local TF-IDF extractive retrieval."""

    def __init__(self, db: Chroma) -> None:
        """Load indexed text and prepare a local TF-IDF matrix for fallback search."""
        # Load the indexed chunks once; both section matching and TF-IDF use them.
        self.db = db
        stored = db.get(include=["documents", "metadatas"])
        documents = stored.get("documents", [])
        metadatas = stored.get("metadatas", [])

        self.documents = [
            Document(
                page_content=content,
                metadata={
                    **(metadatas[index] if index < len(metadatas) and metadatas[index] else {}),
                    "_document_index": index,
                },
            )
            for index, content in enumerate(documents)
            if content and content.strip()
        ]

        if self.documents:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.document_matrix = self.vectorizer.fit_transform(
                [document.page_content for document in self.documents]
            )
        else:
            self.vectorizer = None
            self.document_matrix = None
        logger.info("QAEngine loaded %d indexed chunks for local retrieval.", len(self.documents))

    def retrieve(self, question: str, limit: int = 3) -> list[Document]:
        """Rank indexed chunks with TF-IDF, query coverage, and keyword boosts."""
        if not self.documents or self.vectorizer is None or self.document_matrix is None:
            logger.warning("TF-IDF retrieval is unavailable because the index is empty.")
            return []

        # TF-IDF ranks lexical overlap; it is independent of the Chroma embeddings.
        query_vector = self.vectorizer.transform([question])
        similarities = linear_kernel(query_vector, self.document_matrix).ravel()

        ranked: list[tuple[float, int, Document]] = []
        for index, (score, document) in enumerate(zip(similarities, self.documents)):
            adjusted_score = (
                float(score)
                + self._query_coverage_score(question, document.page_content)
                + self._keyword_boost(question, document.page_content)
            )
            if adjusted_score > 0:
                ranked.append((adjusted_score, index, document))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        results = [document for _, _, document in ranked[:limit]]
        logger.info(
            "TF-IDF retrieval returned %d of up to %d candidate chunk(s).",
            len(results),
            limit,
        )
        for rank, (score, _, document) in enumerate(ranked[:limit], start=1):
            logger.debug(
                "TF-IDF result rank=%d score=%.3f source=%s page=%s",
                rank,
                score,
                document.metadata.get("source", "unknown"),
                document.metadata.get("page", "unknown"),
            )
        return results

    def answer(self, question: str) -> str:
        """Return a matched section or compose an extractive answer from ranked text."""
        section_docs = self._find_matching_section(question)
        if section_docs:
            logger.info("Extractive answer uses a matched section (%d document(s)).", len(section_docs))
            for rank, document in enumerate(section_docs, start=1):
                logger.debug(
                    "Extractive section chunk rank=%d source=%s page=%s",
                    rank,
                    document.metadata.get("source", "unknown"),
                    document.metadata.get("page", "unknown"),
                )
            section_content = "\n".join(
                document.page_content.strip()
                for document in section_docs
                if document.page_content.strip()
            )
            return section_content

        multiline = self._needs_multiline_answer(question) or self._is_list_question(question)
        docs = self.retrieve(question, limit=8 if multiline else 3)
        if not docs:
            logger.warning("Extractive answer could not find relevant text.")
            return "I could not find that information in the document."

        docs = self._expand_related_section_docs(question, docs)

        acronym_answer = self._extract_acronym_answer(question, docs)
        if acronym_answer:
            return acronym_answer

        following_list_answer = self._extract_following_list_answer(question, docs)
        if following_list_answer:
            return following_list_answer

        if self._is_structured_list_question(question, docs):
            structured_list_answer = self._compose_structured_list_answer(docs)
            if structured_list_answer:
                return structured_list_answer

        if self._is_list_question(question):
            list_answer = self._compose_list_answer(docs)
            if list_answer:
                return list_answer

        if self._is_section_question(question, docs):
            section_answer = self._compose_explanatory_answer(question, docs)
            if section_answer:
                return section_answer

        passages = self._rank_passages(question, docs)
        if not passages or passages[0][0] <= 0:
            return "I could not find that information in the document."

        if multiline:
            if self._is_explanatory_question(question):
                explanatory_answer = self._compose_explanatory_answer(question, docs)
                if explanatory_answer:
                    return explanatory_answer

            selected = [passage for score, passage in passages[:3] if score > 0]
            if not selected:
                return "I could not find that information in the document."
            return "\n".join(selected)

        concise_answer = self._extract_concise_answer(question, passages)
        if concise_answer:
            return concise_answer

        return passages[0][1]

    def _find_matching_section(self, question: str) -> list[Document]:
        """Try numbered heading matching first, then match a named heading."""
        numbered_section = self._find_matching_numbered_section(question)
        if numbered_section:
            logger.info("Section retrieval matched a numbered heading.")
            return numbered_section
        named_section = self._find_matching_named_section(question)
        if named_section:
            logger.info("Section retrieval matched a named heading.")
            return named_section
        logger.debug("Section retrieval found no heading match.")
        return []

    def _find_matching_named_section(self, question: str) -> list[Document]:
        """Find the best title-cased heading match and collect its section text."""
        question_terms = query_terms(question)
        if not question_terms:
            return []

        candidates: list[tuple[float, int, int, Document, str]] = []
        for document in self.documents:
            lines = document.page_content.splitlines()
            for line_index, line in enumerate(lines):
                heading = line.strip()
                if not self._is_named_section_heading(lines, line_index):
                    continue

                heading_terms = query_terms(heading)
                overlap = question_terms & heading_terms
                if not overlap:
                    continue

                coverage = len(overlap) / len(question_terms)
                precision = len(overlap) / len(heading_terms) if heading_terms else 0.0
                if coverage >= 0.5 and precision >= 0.5:
                    candidates.append(
                        (
                            coverage + precision,
                            len(heading_terms),
                            line_index,
                            document,
                            heading,
                        )
                    )

        if not candidates:
            return []

        _, _, _, anchor_document, heading = max(
            candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[1],
                -int(candidate[3].metadata.get("page", 0)),
            ),
        )
        related_documents = self._related_documents_for_anchor(anchor_document)
        combined_lines = [
            line
            for document in related_documents
            for line in document.page_content.splitlines()
        ]

        start = next(
            (
                index
                for index, line in enumerate(combined_lines)
                if line.strip().casefold() == heading.casefold()
            ),
            -1,
        )
        if start < 0:
            return []

        end = len(combined_lines)
        for line_index in range(start + 1, len(combined_lines)):
            line = combined_lines[line_index]
            if re.match(r"^\s*\d{1,2}\.\s+", line):
                end = line_index
                break
            if self._is_named_section_heading(combined_lines, line_index):
                end = line_index
                break

        content = "\n".join(combined_lines[start:end]).strip()
        if not content:
            return []
        return [Document(page_content=content, metadata=anchor_document.metadata)]

    def _is_named_section_heading(self, lines: list[str], line_index: int) -> bool:
        """Decide whether a line looks like a standalone, non-numbered section heading."""
        line = lines[line_index].strip()
        if not line or line.endswith(":") or re.match(r"^(?:[•*-]|\d+[.)]\s+)", line):
            return False

        words = line.split()
        if not 1 <= len(words) <= 6:
            return False
        heading_word = r"[A-Z][A-Za-z]*(?:-[A-Za-z]+)*"
        if not all(re.fullmatch(heading_word, word) for word in words):
            return False

        next_line = next(
            (candidate.strip() for candidate in lines[line_index + 1:] if candidate.strip()),
            "",
        )
        if not next_line:
            return False
        if len(words) == 1 and not re.match(
            rf"^{re.escape(line)}\b",
            next_line,
            flags=re.IGNORECASE,
        ):
            return False
        if next_line.endswith(":") and len(next_line.split()) <= 3:
            return False
        if re.fullmatch(
            rf"{heading_word}(?:\s+{heading_word}){{1,5}}",
            next_line,
        ):
            return False
        return True

    def _find_matching_numbered_section(self, question: str) -> list[Document]:
        """Match a numbered heading and gather its following chunks until the next section."""
        question_terms = query_terms(question)
        if not question_terms:
            return []

        heading_pattern = re.compile(r"^\s*(\d{1,2})\.\s*(.+?)\s*$")
        candidates: list[tuple[float, int, int, Document, str]] = []
        for document in self.documents:
            lines = document.page_content.splitlines()
            if any(line.strip().lower() == "index" for line in lines[:5]):
                continue

            for line_index, line in enumerate(lines):
                match = heading_pattern.match(line)
                if not match:
                    continue

                heading_terms = query_terms(match.group(2))
                overlap = question_terms & heading_terms
                if not overlap:
                    continue

                coverage = len(overlap) / len(question_terms)
                precision = len(overlap) / len(heading_terms) if heading_terms else 0.0
                score = coverage + precision
                if score >= 1.0:
                    candidates.append(
                        (
                            score,
                            len(heading_terms),
                            line_index,
                            document,
                            match.group(1),
                        )
                    )

        if not candidates:
            return []

        _, _, heading_line_index, heading_document, section_number = max(
            candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[1],
                -int(candidate[3].metadata.get("page", 0)),
            ),
        )
        source = heading_document.metadata.get("source")
        if not isinstance(source, str):
            return []

        ordered_documents = sorted(
            (
                document
                for document in self.documents
                if document.metadata.get("source") == source
                and isinstance(document.metadata.get("page"), int)
                and isinstance(heading_document.metadata.get("page"), int)
                and document.metadata["page"] >= heading_document.metadata["page"]
            ),
            key=self._document_sort_key,
        )

        section_documents: list[Document] = []
        started = False
        previous_content = ""
        for document in ordered_documents:
            lines = document.page_content.splitlines()
            start_line = 0
            if not started:
                if document.metadata.get("_document_index") != heading_document.metadata.get("_document_index"):
                    continue
                start_line = heading_line_index
                started = True
            else:
                first_content_line = next(
                    (
                        line_index
                        for line_index, line in enumerate(lines)
                        if line.strip()
                    ),
                    -1,
                )
                if first_content_line >= 0:
                    first_line = lines[first_content_line]
                    if heading_pattern.match(first_line) or self._is_named_section_heading(
                        lines,
                        first_content_line,
                    ):
                        break

            section_end = len(lines)
            for line_index in range(start_line, len(lines)):
                match = heading_pattern.match(lines[line_index])
                if match and int(match.group(1)) > int(section_number):
                    section_end = line_index
                    break

            content = "\n".join(lines[start_line:section_end]).strip()
            if content:
                original_content = content
                max_overlap = min(len(previous_content), len(content))
                for overlap in range(max_overlap, 0, -1):
                    if previous_content.endswith(content[:overlap]):
                        content = content[overlap:].lstrip()
                        break
                previous_content = original_content
            if content:
                section_documents.append(
                    Document(page_content=content, metadata=document.metadata)
                )

            if section_end < len(lines):
                break

        return section_documents

    def _keyword_boost(self, question: str, content: str) -> float:
        """Increase lexical ranking for named entities and expected answer patterns."""
        question_lower = question.lower()
        content_lower = content.lower()
        proper_nouns = query_entities(question)
        boost = 0.0

        for noun in proper_nouns:
            if noun.lower() in content_lower:
                boost += 0.75

        if question_lower.startswith("who is") or question_lower.startswith("who was"):
            for noun in proper_nouns:
                if f"{noun.lower()} is" in content_lower or f"{noun.lower()} was" in content_lower:
                    boost += 1.5

        if question_lower.startswith("how many") and re.search(r"\b\d+\b", content):
            boost += 1.0

        if question_lower.startswith("when") and re.search(r"\b(19|20)\d{2}\b", content):
            boost += 1.0

        return boost

    def _query_coverage_score(self, question: str, content: str) -> float:
        """Score the fraction of meaningful question terms present in a chunk."""
        terms = query_terms(question)
        if not terms:
            return 0.0

        content_terms = set(tokenize(content))
        coverage = len(terms & content_terms) / len(terms)
        return coverage * 1.5

    def _rank_passages(self, question: str, docs: list[Document]) -> list[tuple[float, str]]:
        """Rank sentence-level passages using term overlap and quality adjustments."""
        ranked: list[tuple[float, str]] = []
        question_tokens = query_terms(question)
        proper_nouns = query_entities(question)
        question_lower = question.lower()

        for document in docs:
            for passage in build_passages(document.page_content):
                passage_lower = passage.lower()
                passage_tokens = set(tokenize(passage))
                overlap = len(question_tokens & passage_tokens)
                score = float(overlap)

                for noun in proper_nouns:
                    if noun.lower() in passage_lower:
                        score += 2.0

                if question_lower.startswith("who is") or question_lower.startswith("who was"):
                    if proper_nouns and any(f"{noun.lower()} is" in passage_lower or f"{noun.lower()} was" in passage_lower for noun in proper_nouns):
                        score += 4.0

                if "created" in question_lower and "created" in passage_lower:
                    score += 3.0

                if question_lower.startswith("who") and re.search(r"\b[A-Z][a-z]+(?:\s+[a-z]+)?(?:\s+[A-Z][a-z]+)+\b", passage):
                    score += 1.0

                if question_lower.startswith("how many") and re.search(r"\b\d+\b", passage):
                    score += 2.0

                if question_lower.startswith("when") and re.search(r"\b(19|20)\d{2}\b", passage):
                    score += 2.0

                score += self._passage_quality_adjustment(passage)

                if passage.strip():
                    ranked.append((score, passage.strip()))

        ranked.sort(key=lambda item: item[0], reverse=True)

        unique_ranked: list[tuple[float, str]] = []
        seen: set[str] = set()
        for score, sentence in ranked:
            if sentence not in seen:
                seen.add(sentence)
                unique_ranked.append((score, sentence))
        return unique_ranked

    def _needs_multiline_answer(self, question: str) -> bool:
        """Recognize question openings that usually need more than a short fact."""
        question_lower = question.lower().strip()
        return (
            question_lower.startswith("explain")
            or question_lower.startswith("describe")
            or (question_lower.startswith("how") and not question_lower.startswith("how many"))
            or question_lower.startswith("tell me about")
            or question_lower.startswith("why")
            or question_lower.startswith("list")
            or question_lower.startswith("show")
            or question_lower.startswith("give me all")
        )

    def _is_list_question(self, question: str) -> bool:
        """Identify requests whose wording indicates a list-style response."""
        question_lower = question.lower().strip()
        return question_lower.startswith("list") or " all " in f" {question_lower} "

    def _compose_list_answer(self, docs: list[Document]) -> str:
        """Order and combine unique retrieved chunks into a plain-text list answer."""
        section_heading = self._best_section_heading(docs)
        if section_heading:
            docs = [document for document in docs if document.page_content.strip().startswith(f"{section_heading} ")]

        unique_docs: list[Document] = []
        seen: set[str] = set()

        for document in sorted(docs, key=self._document_sort_key):
            content = document.page_content.strip()
            if not content or content in seen:
                continue
            seen.add(content)
            unique_docs.append(document)

        if not unique_docs:
            return ""

        heading = self._shared_heading([document.page_content for document in unique_docs])
        lines: list[str] = []

        for document in unique_docs:
            content = document.page_content.strip()
            if heading and content.startswith(f"{heading} "):
                content = content[len(heading):].strip()
            lines.append(content)

        if heading:
            return f"{heading}\n\n" + "\n\n".join(lines)

        return "\n".join(lines)

    def _best_section_heading(self, docs: list[Document]) -> str:
        """Return a heading repeated across the retrieved documents, if any."""
        counts: dict[str, int] = {}

        for document in docs:
            heading = self._extract_heading(document.page_content)
            if heading:
                counts[heading] = counts.get(heading, 0) + 1

        if not counts:
            return ""

        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        best_heading, occurrences = ranked[0]
        if occurrences < 2:
            return ""
        return best_heading

    def _shared_heading(self, contents: list[str]) -> str:
        """Return a heading only when every content block has the same heading."""
        headings = []
        for content in contents:
            heading = self._extract_heading(content)
            if heading:
                headings.append(heading)

        if not headings:
            return ""

        first_heading = headings[0]
        if all(heading == first_heading for heading in headings):
            return first_heading

        return ""

    def _extract_heading(self, content: str) -> str:
        """Extract a leading colon-terminated label from chunk text."""
        match = re.match(r"^([^:]+:)\s+", content)
        if not match:
            return ""
        return match.group(1).strip()

    def _document_sort_key(self, document: Document) -> tuple[int, str]:
        """Sort chunks by source order and then by their indexed document position."""
        order = document.metadata.get("source_order")
        document_index = document.metadata.get("_document_index")
        if isinstance(order, int) and isinstance(document_index, int):
            return (order, document_index)
        if isinstance(order, int):
            return (order, 10**9)
        if isinstance(document_index, int):
            return (10**9, document_index)
        return (10**9, 10**9)

    def _is_explanatory_question(self, question: str) -> bool:
        """Recognize question openings that request an explanation or description."""
        question_lower = question.lower().strip()
        return (
            question_lower.startswith("explain")
            or question_lower.startswith("describe")
            or question_lower.startswith("tell me about")
            or (question_lower.startswith("how") and not question_lower.startswith("how many"))
            or question_lower.startswith("why")
        )

    def _compose_explanatory_answer(self, question: str, docs: list[Document]) -> str:
        """Join related source chunks and clean the result for an explanation."""
        if not docs:
            return ""

        combined_section = self._combine_section_documents(question, docs)
        cleaned = self._clean_document_content(question, combined_section)
        return cleaned.strip()

    def _combine_section_documents(self, question: str, docs: list[Document]) -> str:
        """Choose the strongest anchor chunk and combine its related source chunks."""
        anchor_doc = max(docs, key=lambda document: self._best_passage_score(question, document))
        ordered_docs = self._related_documents_for_anchor(anchor_doc)
        combined = "\n".join(document.page_content.strip() for document in ordered_docs if document.page_content.strip())
        return self._truncate_at_callout(question, combined)

    def _related_documents_for_anchor(self, anchor_doc: Document) -> list[Document]:
        """Find same-section, same-page, or same-source-order chunks for an anchor."""
        source = anchor_doc.metadata.get("source")
        section_id = anchor_doc.metadata.get("section_id")
        source_order = anchor_doc.metadata.get("source_order")

        if source is None:
            return [anchor_doc]

        if isinstance(section_id, int):
            related_docs = [
                document
                for document in self.documents
                if document.metadata.get("source") == source
                and document.metadata.get("section_id") == section_id
            ]
            return sorted(related_docs, key=self._document_sort_key) or [anchor_doc]

        page = anchor_doc.metadata.get("page")
        if isinstance(page, int):
            related_docs = [
                document
                for document in self.documents
                if document.metadata.get("source") == source
                and document.metadata.get("page") == page
            ]
            ordered_docs = sorted(related_docs, key=self._document_sort_key)
            if not ordered_docs:
                return [anchor_doc]

            deduplicated_docs: list[Document] = []
            previous_content = ""
            for document in ordered_docs:
                content = document.page_content
                max_overlap = min(len(previous_content), len(content))
                for overlap in range(max_overlap, 0, -1):
                    if previous_content.endswith(content[:overlap]):
                        content = content[overlap:]
                        break

                if content:
                    deduplicated_docs.append(
                        Document(page_content=content, metadata=document.metadata)
                    )
                previous_content = document.page_content

            return deduplicated_docs or [anchor_doc]

        if not isinstance(source_order, int):
            return [anchor_doc]

        related_docs = [
            document
            for document in self.documents
            if document.metadata.get("source") == source
            and document.metadata.get("source_order") == source_order
        ]

        if not related_docs:
            return [anchor_doc]

        return sorted(related_docs, key=self._document_sort_key)

    def _is_section_question(self, question: str, docs: list[Document]) -> bool:
        """Check whether the question and retrieved chunks identify one text section."""
        question_lower = question.lower().strip()
        if not question_lower.startswith(("what are", "what is", "describe", "explain")):
            return False

        section_ids = {
            (document.metadata.get("source"), document.metadata.get("section_id"))
            for document in docs
            if isinstance(document.metadata.get("section_id"), int)
        }
        return len(section_ids) == 1 and len(docs) > 1

    def _clean_document_content(self, question: str, content: str) -> str:
        """Trim irrelevant section text and normalize spacing in extracted content."""
        content = self._trim_to_requested_section(question, content)
        trimmed = self._trim_to_relevant_lines(question, content)
        if trimmed:
            content = trimmed

        cleaned = re.sub(r"^\s*\d+\.\s*", "", content.strip())
        cleaned = re.sub(r"\n\s*(\d+)\.\s*", r"\n\1. ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _trim_to_requested_section(self, question: str, content: str) -> str:
        """Keep content from the numbered heading that best matches the question."""
        lines = content.splitlines()
        heading_pattern = re.compile(r"^\s*\d{1,2}\.\s+([^.!?]+?)\s*$")
        headings: list[tuple[int, set[str]]] = []
        for index, line in enumerate(lines):
            match = heading_pattern.match(line)
            if match:
                headings.append((index, query_terms(match.group(1))))

        question_tokens = query_terms(question)
        if not headings or not question_tokens:
            return content

        best_heading_index = -1
        best_score = 0.0
        for position, (line_index, heading_tokens) in enumerate(headings):
            coverage = len(question_tokens & heading_tokens) / len(question_tokens)
            if coverage > best_score:
                best_score = coverage
                best_heading_index = position

        if best_heading_index < 0 or best_score < 0.5:
            return content

        start = headings[best_heading_index][0]
        end = (
            headings[best_heading_index + 1][0]
            if best_heading_index + 1 < len(headings)
            else len(lines)
        )
        return "\n".join(lines[start:end]).strip()

    def _best_passage_score(self, question: str, document: Document) -> float:
        """Return a document's top passage score for selecting an anchor chunk."""
        ranked = self._rank_passages(question, [document])
        if not ranked:
            return float("-inf")
        return ranked[0][0]

    def _truncate_at_callout(self, question: str, content: str) -> str:
        """Stop at Tip, Warning, or Info unless the question asks about one."""
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in {"tip", "warning", "info"}):
            return content

        lines = content.splitlines()
        kept_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped in {"Tip", "Warning", "Info"}:
                break
            kept_lines.append(line)

        cleaned = "\n".join(kept_lines).strip()
        return cleaned or content

    def _trim_to_relevant_lines(self, question: str, content: str) -> str:
        """Start extracted content at the line with the strongest query overlap."""
        lines = [line.rstrip() for line in content.splitlines()]
        if not lines:
            return content

        question_tokens = query_terms(question)
        best_index = 0
        best_score = float("-inf")

        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            line_tokens = set(tokenize(stripped))
            score = float(len(question_tokens & line_tokens))

            if re.search(rf"\b{re.escape(question.strip().rstrip('?').lower())}\b", stripped.lower()):
                score += 4.0

            if stripped.endswith(":") or re.match(r"^\d+\.\s+", stripped):
                score += 1.0

            if score > best_score:
                best_score = score
                best_index = index

        trimmed_lines = lines[best_index:]
        while trimmed_lines and not trimmed_lines[0].strip():
            trimmed_lines.pop(0)

        return "\n".join(trimmed_lines).strip() or content

    def _extract_concise_answer(self, question: str, passages: list[tuple[float, str]]) -> str:
        """Apply simple patterns for year, count, and creator questions."""
        question_lower = question.lower().strip()

        for _, passage in passages:
            if question_lower.startswith("when"):
                released_match = re.search(
                    r"(?:released|release(?:d)?)\s+(?:in|on)?\s*((?:19|20)\d{2})",
                    passage,
                    flags=re.IGNORECASE,
                )
                if released_match:
                    return f"It was released in {released_match.group(1)}."

                year_match = re.search(r"\b((?:19|20)\d{2})\b", passage)
                if year_match:
                    return year_match.group(1)

            if question_lower.startswith("how many"):
                count_match = re.search(r"\b(\d+)\b", passage)
                if count_match:
                    return count_match.group(1)

            if question_lower.startswith("who created"):
                creator_match = re.search(
                    r"created\s+by\s+([^\n.]+?)(?:\s+and\b|\.|$)",
                    passage,
                    flags=re.IGNORECASE,
                )
                if creator_match:
                    return creator_match.group(1).strip()

        return ""

    def _extract_acronym_answer(self, question: str, docs: list[Document]) -> str:
        """Build an acronym expansion from source lines shaped as component labels."""
        question_lower = question.lower()
        if not any(term in question_lower for term in {"full form", "fulform", "stands for", "stand for", "acronym", "expand"}):
            return ""

        content = self._combine_section_documents(question, docs)
        component_matches = re.findall(r"^([A-Z][a-z]+)\s+--", content, flags=re.MULTILINE)
        if len(component_matches) < 2:
            return ""

        parsed_acronym = "".join(component[0].upper() for component in component_matches)
        target_acronym = self._extract_target_acronym(question)
        acronym = parsed_acronym if parsed_acronym else target_acronym

        if not acronym:
            return ""

        if target_acronym and parsed_acronym and target_acronym != parsed_acronym:
            acronym = parsed_acronym

        if len(component_matches) == 2:
            expansion = f"{component_matches[0]} and {component_matches[1]}"
        else:
            expansion = ", ".join(component_matches[:-1]) + f", and {component_matches[-1]}"

        return f"{acronym} stands for {expansion}."

    def _extract_following_list_answer(self, question: str, docs: list[Document]) -> str:
        """Return a bullet list following a line that closely matches the query."""
        terms = query_terms(question)
        if not terms:
            return ""

        best_coverage = 0.0
        best_answer = ""

        for document in docs:
            lines = [line.strip() for line in document.page_content.splitlines() if line.strip()]
            for index, line in enumerate(lines[:-1]):
                next_line = lines[index + 1]
                if not re.match(r"^[•\-*]\s+", next_line):
                    continue

                line_terms = set(tokenize(line))
                coverage = len(terms & line_terms) / len(terms)
                if coverage < 0.75 or coverage <= best_coverage:
                    continue

                answer_lines = lines[index + 1:]
                best_coverage = coverage
                best_answer = "\n".join(answer_lines)

        return best_answer

    def _extract_target_acronym(self, question: str) -> str:
        """Infer the acronym token requested by an acronym-expansion question."""
        explicit_match = re.search(r"\b([A-Za-z]{2,})\s+framework\b", question, flags=re.IGNORECASE)
        if explicit_match:
            return explicit_match.group(1).upper()

        uppercase_tokens = re.findall(r"\b[A-Z]{2,}\b", question)
        if uppercase_tokens:
            return uppercase_tokens[0].upper()

        title_tokens = re.findall(r"\b[A-Za-z]{2,}\b", question)
        for token in title_tokens:
            if token.lower() not in QUESTION_STOPWORDS and len(token) >= 3:
                return token.upper()

        return ""

    def _passage_quality_adjustment(self, passage: str) -> float:
        """Adjust passage scores to favor readable prose over index-like text."""
        score = 0.0
        stripped = passage.strip()
        lowered = stripped.lower()
        numbered_headings = re.findall(r"(?:^|\n)\s*\d+\.\s+[A-Z][^\n]+", stripped)
        line_count = len([line for line in stripped.splitlines() if line.strip()])

        if "index" in lowered:
            score -= 4.0

        if len(numbered_headings) >= 2:
            score -= 3.0

        if re.search(r"\b\d+\.\s+[A-Z][^\n]+\n\s*\d+\.\s+[A-Z]", stripped):
            score -= 3.0

        if line_count >= 3 and not re.search(r"[.!?]", stripped):
            score -= 2.0

        if re.search(r"\b(the model|when a request|process|receives|identifies|predicts|generating)\b", lowered):
            score += 1.5

        if re.search(r"[.!?]", stripped):
            score += 0.5

        return score

    def _expand_related_section_docs(self, question: str, docs: list[Document]) -> list[Document]:
        """Expand retrieval around the best chunk when its related source chunks exist."""
        if not docs:
            return docs

        anchor_doc = max(docs, key=lambda document: self._best_passage_score(question, document))
        related_docs = self._related_documents_for_anchor(anchor_doc)

        if len(related_docs) <= 1:
            return docs

        return related_docs

    def _is_structured_list_question(self, question: str, docs: list[Document]) -> bool:
        """Check for list-like wording and at least two extractable list items."""
        question_lower = question.lower().strip()
        if not (
            question_lower.startswith("list")
            or question_lower.startswith("show")
            or question_lower.startswith("give me")
            or question_lower.startswith("what are")
            or question_lower.startswith("which are")
        ):
            return False

        return len(self._extract_structured_list_items(docs)) >= 2

    def _compose_structured_list_answer(self, docs: list[Document]) -> str:
        """Return parsed numbered, bullet, or named-mistake items on separate lines."""
        items = self._extract_structured_list_items(docs)
        if not items:
            return ""
        return "\n".join(items)

    def _extract_structured_list_items(self, docs: list[Document]) -> list[str]:
        """Parse common numbered, bullet, or Mistake-labelled lists from chunks."""
        combined = "\n".join(document.page_content for document in docs)
        combined = re.sub(r"\bWarning\b", "", combined)
        combined = re.sub(r"\n{2,}", "\n", combined)

        mistake_matches = re.findall(
            r"(Mistake\s+#\d+:.*?)(?=Mistake\s+#\d+:|\bTip\b|$)",
            combined,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if mistake_matches:
            return [self._normalize_list_item(match) for match in mistake_matches]

        numbered_matches = re.findall(
            r"(^\d+\.\s+.*?)(?=^\d+\.\s+|$)",
            combined,
            flags=re.MULTILINE | re.DOTALL,
        )
        if numbered_matches:
            return [self._normalize_list_item(match) for match in numbered_matches]

        bullet_matches = re.findall(
            r"(^[•\-*]\s+.*?)(?=^[•\-*]\s+|$)",
            combined,
            flags=re.MULTILINE | re.DOTALL,
        )
        return [self._normalize_list_item(match) for match in bullet_matches]

    def _normalize_list_item(self, item: str) -> str:
        """Collapse whitespace inside one extracted structured-list item."""
        cleaned = re.sub(r"\s+", " ", item).strip()
        return cleaned