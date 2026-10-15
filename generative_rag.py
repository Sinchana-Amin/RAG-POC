"""Retrieve source passages, ask Ollama to answer, and fall back on failures."""

from __future__ import annotations

import logging
import os
import re

import requests
from langchain_core.documents import Document

from qa_engine import QAEngine

MIN_CONTEXT_RELEVANCE = 0.3
logger = logging.getLogger(__name__)


class GenerativeRAG:
    """Coordinate context retrieval, Ollama generation, and extractive fallback."""

    def __init__(
        self,
        retriever: QAEngine,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Set up retrieval, Ollama configuration, and per-session follow-up history."""
        self.retriever = retriever
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.last_backend = "not used"
        self.chat_history: list[dict[str, str]] = []
        self._last_topic_question = ""

    def answer(self, question: str) -> str:
        """Retrieve context, generate an answer, and retain recent conversation turns."""
        is_follow_up = self._is_follow_up(question) and bool(self._last_topic_question)
        retrieval_query = self._last_topic_question if is_follow_up else question
        logger.info(
            "Answer request started (question_chars=%d, follow_up=%s)",
            len(question),
            is_follow_up,
        )
        answer = self._answer_question(question, retrieval_query, include_history=is_follow_up)

        self.chat_history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        self.chat_history = self.chat_history[-8:]
        if not is_follow_up:
            self._last_topic_question = question
        return answer

    def _answer_question(
        self,
        question: str,
        retrieval_query: str,
        include_history: bool,
    ) -> str:
        """Run retrieval and generation, using extractive output if Ollama fails."""
        try:
            documents = self._retrieve_context(retrieval_query)
        except requests.RequestException:
            logger.exception("Context retrieval request failed; using extractive fallback.")
            self.last_backend = "extractive fallback"
            return self.retriever.answer(retrieval_query)

        if not documents:
            self.last_backend = "retrieval"
            logger.warning("Retrieval returned no context; generation was skipped.")
            return "I could not find that information in the indexed sources."

        context = self._format_context(documents)
        logger.info(
            "Sending %d retrieved context characters to Ollama model %s.",
            len(context),
            self.model,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-augmented question-answering assistant. "
                    "Answer only from the supplied context. Do not use outside knowledge or invent facts. "
                    "Use conversation history to resolve follow-up references such as it, that, or its example. "
                    "For a simple fact, give one concise sentence. For a list, return every relevant item. "
                    "For an explanation, include all relevant details while avoiding unrelated text. "
                    "When a section has multiple named subsections, explain every subsection in source order. "
                    "Preserve source lists as separate bullet points; do not replace a list with a summary. "
                    "When the source has a numbered sequence, include every step in order. "
                    "Do not stop after the first item of a list or sequence. "
                    "For a named topic or section, include its definition, every associated list item, "
                    "and any relevant example or concluding sentence. "
                    "For examples shown as templates or diagrams, treat labels without values as "
                    "placeholders and do not invent values for them. "
                    "If the context does not contain the answer, reply exactly: "
                    "I could not find that information in the indexed sources."
                ),
            },
            *(self.chat_history[-6:] if include_history else []),
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        logger.info("Calling Ollama chat endpoint for model %s.", self.model)
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": messages,
                    "options": {
                        "temperature": 0,
                        "num_ctx": 4096,
                        "num_predict": 512,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("Ollama chat request failed; using extractive fallback.")
            self.last_backend = "extractive fallback"
            return self.retriever.answer(question)

        try:
            payload = response.json()
        except ValueError:
            logger.exception("Ollama returned invalid JSON; using extractive fallback.")
            self.last_backend = "extractive fallback"
            return self.retriever.answer(question)

        message = payload.get("message") if isinstance(payload, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        answer = content.strip() if isinstance(content, str) else ""
        if not answer:
            logger.warning("Ollama returned an empty answer; using extractive fallback.")
            self.last_backend = "extractive fallback"
            return self.retriever.answer(retrieval_query)

        source_answer = self.retriever.answer(retrieval_query)
        if self._omits_requested_content(question, answer, source_answer):
            logger.info("Answer completeness check requested one LLM retry.")
            answer = self._retry_structured_answer(
                question,
                answer,
                source_answer,
            )

        self.last_backend = f"Ollama/{self.model}"
        logger.info("Answer generated successfully (answer_chars=%d).", len(answer))
        return answer

    def _is_follow_up(self, question: str) -> bool:
        """Detect common follow-up wording so prior topic context can be reused."""
        lowered = question.casefold().strip()
        follow_up_patterns = (
            r"\b(it|its|this|that|these|those|them|they)\b",
            r"\b(elaborate|expand|more details|explain more)\b",
            r"\b(also|further|continue|what about)\b",
        )
        return any(re.search(pattern, lowered) for pattern in follow_up_patterns)

    def _retry_structured_answer(
        self,
        question: str,
        first_answer: str,
        source_answer: str,
    ) -> str:
        """Ask the LLM once more to cover recognizable required source items."""
        required_items = self._required_answer_items(question, source_answer)
        if not required_items:
            logger.debug("No structured completeness requirements were detected.")
            return first_answer

        numbered_sequence = self._longest_numbered_sequence(source_answer)
        if len(numbered_sequence) > 1:
            formatted_items = "\n".join(
                f"{index}. {item}" for index, item in enumerate(numbered_sequence, start=1)
            )
        else:
            formatted_items = "\n".join(f"- {item}" for item in required_items)

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Write the answer using only the supplied source material. "
                                "Include every required item exactly once and in the given order. "
                                "For named components, include each component and its source description. "
                                "For templates or diagrams, treat labels without values as placeholders; "
                                "do not invent values for them. Summarize related labels together when appropriate. "
                                "Do not omit a required item or stop early."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n\n"
                                f"Source passage:\n{source_answer}\n\n"
                                f"Required answer items:\n{formatted_items}\n\n"
                                "Answer the question using the source passage. Include every required item "
                                "and its explanation when the source provides one. Do not omit, merge, "
                                "or stop early."
                            ),
                        },
                    ],
                    "options": {
                        "temperature": 0,
                        "num_ctx": 4096,
                        "num_predict": 1024,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            logger.exception("Structured-answer retry failed; keeping the first LLM answer.")
            return first_answer

        message = payload.get("message") if isinstance(payload, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        retry_answer = content.strip() if isinstance(content, str) else ""
        if not retry_answer:
            return first_answer

        if self._answer_completeness_score(question, retry_answer, source_answer) >= self._answer_completeness_score(
            question,
            first_answer,
            source_answer,
        ) and self._answer_completeness_score(question, retry_answer, source_answer) >= len(required_items):
            logger.info("Structured-answer retry met the completeness check.")
            return retry_answer
        logger.warning("Structured-answer retry did not meet completeness checks; keeping first answer.")
        return first_answer

    def _omits_requested_content(self, question: str, answer: str, source_answer: str) -> bool:
        """Check whether the response omitted expected structured or concluding content."""
        if self._omits_structured_items(answer, source_answer):
            return True

        if re.search(r"\bexamples?\b", question, flags=re.IGNORECASE):
            example_items = self._example_source_items(source_answer)
            if example_items and not self._contains_example_items(answer, example_items):
                return True

        conclusion = self._section_conclusion_item(source_answer)
        if self._is_explanatory_question(question) and conclusion:
            return conclusion.casefold() not in re.sub(r"\s+", " ", answer).casefold()

        return False

    def _omits_structured_items(self, answer: str, source_answer: str) -> bool:
        """Compare recognized source list items with those present in the answer."""
        source_items = self._structured_source_items(source_answer)
        if len(source_items) < 2:
            return False
        return self._structured_answer_score(answer, source_answer) < len(source_items)

    def _required_answer_items(self, question: str, source_answer: str) -> list[str]:
        """Extract checklist items used by the structured-answer retry and scoring."""
        if re.search(r"\bexamples?\b", question, flags=re.IGNORECASE):
            example_items = self._example_source_items(source_answer)
            if example_items:
                return example_items

        if self._is_explanatory_question(question):
            named_items = self._named_section_items(source_answer)
            if len(named_items) > 1:
                required_items = named_items + self._structured_source_items(source_answer)
                conclusion = self._section_conclusion_item(source_answer)
                if conclusion:
                    required_items.append(conclusion)
                return list(dict.fromkeys(required_items))

        items = self._structured_source_items(source_answer)
        conclusion = self._section_conclusion_item(source_answer)
        if self._is_explanatory_question(question) and conclusion and conclusion not in items:
            items.append(conclusion)
        return items

    def _section_conclusion_item(self, source_answer: str) -> str:
        """Find the final non-bullet sentence after any source list."""
        lines = source_answer.splitlines()
        last_bullet_index = max(
            (
                index
                for index, line in enumerate(lines)
                if re.match(r"^\s*[•*-]\s+", line)
            ),
            default=-1,
        )
        trailing_lines = [
            line.strip()
            for line in lines[last_bullet_index + 1:]
            if line.strip() and not re.match(r"^\s*[•*-]\s+", line)
        ]
        if not trailing_lines:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", " ".join(trailing_lines))
        return sentences[-1].strip()

    def _is_explanatory_question(self, question: str) -> bool:
        """Identify question forms that request a narrative explanation."""
        return bool(
            re.match(
                r"^\s*(?:explain|describe|why|tell me about)\b",
                question,
                flags=re.IGNORECASE,
            )
        )

    def _example_source_items(self, source_answer: str) -> list[str]:
        """Read the lines following an Example label as checklist items."""
        example_match = re.search(r"^\s*Example\s*:\s*$", source_answer, flags=re.IGNORECASE | re.MULTILINE)
        if not example_match:
            return []

        items: list[str] = []
        for line in source_answer[example_match.end():].splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^[A-Za-z][A-Za-z ]*\s*=\s*.+$", stripped):
                items.append(stripped)
            elif items:
                break
        if items:
            return items

        return [
            line.strip()
            for line in source_answer[example_match.end():].splitlines()
            if line.strip()
        ]

    def _contains_example_items(self, answer: str, example_items: list[str]) -> bool:
        """Check whether the answer includes each extracted example label or value."""
        normalized_answer = re.sub(r"\s+", " ", answer).casefold()
        for item in example_items:
            if "=" not in item:
                normalized_item = re.sub(r"\s+", " ", item).strip().casefold()
                if normalized_item not in normalized_answer:
                    return False
                continue
            _, value = item.split("=", maxsplit=1)
            quantities = re.findall(r"\b\d+(?:\.\d+)?\s*[KMB]?\b", value, flags=re.IGNORECASE)
            if quantities:
                if not all(
                    re.search(rf"(?<!\w){re.escape(quantity.strip())}(?!\w)", answer, flags=re.IGNORECASE)
                    for quantity in quantities
                ):
                    return False
            elif re.sub(r"\s+", " ", value).strip().casefold() not in normalized_answer:
                return False
        return True

    def _structured_source_items(self, text: str) -> list[str]:
        """Prefer numbered or bulleted source items, then fall back to headings."""
        numbered_sequence = self._longest_numbered_sequence(text)
        if len(numbered_sequence) > 1:
            return numbered_sequence

        bullet_item = re.compile(r"^\s*[•*-]\s+(.+)$", flags=re.MULTILINE)
        bullet_items = [item.strip() for item in bullet_item.findall(text)]
        if len(bullet_items) > 1:
            return bullet_items

        return self._named_section_items(text)

    def _named_section_items(self, text: str) -> list[str]:
        """Collect unique title-cased lines treated as subsection labels."""
        heading_pattern = re.compile(r"^[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,5}$")
        excluded_headings = {"Example", "Tip", "Warning", "Info"}
        items: list[str] = []
        lines = text.splitlines()

        for line in lines:
            heading = line.strip()
            if (
                heading
                and heading not in excluded_headings
                and heading_pattern.fullmatch(heading)
                and heading not in items
            ):
                items.append(heading)
        return items

    def _longest_numbered_sequence(self, text: str) -> list[str]:
        """Return the longest uninterrupted increasing numbered list in text."""
        item_pattern = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")
        sequences: list[list[str]] = []
        current_sequence: list[str] = []
        previous_number: int | None = None

        for line in text.splitlines():
            match = item_pattern.match(line)
            if not match:
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = []
                previous_number = None
                continue

            number = int(match.group(1))
            if previous_number is not None and number != previous_number + 1:
                sequences.append(current_sequence)
                current_sequence = []
            current_sequence.append(match.group(2).strip())
            previous_number = number

        if current_sequence:
            sequences.append(current_sequence)
        return max(sequences, key=len, default=[])

    def _structured_answer_score(self, answer: str, source_answer: str) -> int:
        """Count recognizable source list or heading items reproduced in the answer."""
        source_items = self._structured_source_items(source_answer)
        named_items = self._named_section_items(source_answer)
        if len(named_items) > 1:
            answer_lower = answer.casefold()
            return sum(item.casefold() in answer_lower for item in named_items)

        answer_sequence = self._longest_numbered_sequence(answer)
        bullet_item = re.compile(r"^\s*[•*-]\s+.+$", flags=re.MULTILINE)
        answer_item_count = max(len(answer_sequence), len(bullet_item.findall(answer)))
        return min(answer_item_count, len(source_items))

    def _answer_completeness_score(self, question: str, answer: str, source_answer: str) -> int:
        """Score answer coverage against the checklist appropriate for the question."""
        example_items = self._example_source_items(source_answer)
        if re.search(r"\bexamples?\b", question, flags=re.IGNORECASE) and example_items:
            normalized_answer = re.sub(r"\s+", " ", answer).casefold()
            score = 0
            for item in example_items:
                if "=" not in item:
                    normalized_item = re.sub(r"\s+", " ", item).strip().casefold()
                    score += normalized_item in normalized_answer
                    continue
                _, value = item.split("=", maxsplit=1)
                quantities = re.findall(r"\b\d+(?:\.\d+)?\s*[KMB]?\b", value, flags=re.IGNORECASE)
                if quantities and all(
                    re.search(rf"(?<!\w){re.escape(quantity.strip())}(?!\w)", answer, flags=re.IGNORECASE)
                    for quantity in quantities
                ):
                    score += 1
                elif not quantities and re.sub(r"\s+", " ", value).strip().casefold() in normalized_answer:
                    score += 1
            return score

        if self._is_explanatory_question(question):
            required_items = self._required_answer_items(question, source_answer)
            normalized_answer = re.sub(r"\s+", " ", answer).casefold()
            return sum(
                re.sub(r"\s+", " ", item).strip().casefold() in normalized_answer
                for item in required_items
            )

        score = self._structured_answer_score(answer, source_answer)
        conclusion = self._section_conclusion_item(source_answer)
        if self._is_explanatory_question(question) and conclusion:
            normalized_answer = re.sub(r"\s+", " ", answer).casefold()
            score += conclusion.casefold() in normalized_answer
        return score

    def _retrieve_context(self, question: str) -> list[Document]:
        """Prefer a matching source section, otherwise use Chroma vector similarity."""
        section_documents = self.retriever._find_matching_section(question)
        if section_documents:
            logger.info(
                "Retrieval selected section-matching path (%d document(s)).",
                len(section_documents),
            )
            for rank, document in enumerate(section_documents, start=1):
                logger.info(
                    "Section chunk rank=%d source=%s page=%s",
                    rank,
                    document.metadata.get("source", "unknown"),
                    document.metadata.get("page", "unknown"),
                )
            return section_documents

        logger.info(
            "Retrieval selected Chroma vector search (k=8, relevance_threshold=%.2f).",
            MIN_CONTEXT_RELEVANCE,
        )
        try:
            results = self.retriever.db.similarity_search_with_relevance_scores(
                question,
                k=8,
                score_threshold=MIN_CONTEXT_RELEVANCE,
            )
        except Exception:
            logger.exception("Chroma vector search failed.")
            raise
        documents = [document for document, _ in results]
        if not documents:
            logger.warning("Chroma returned no chunks above the relevance threshold.")
            return []

        for rank, (document, score) in enumerate(results, start=1):
            logger.info(
                "Retrieved chunk rank=%d score=%.3f source=%s page=%s",
                rank,
                score,
                document.metadata.get("source", "unknown"),
                document.metadata.get("page", "unknown"),
            )

        initial_count = len(documents)
        documents = self.retriever._expand_related_section_docs(question, documents)
        documents = sorted(documents, key=self.retriever._document_sort_key)
        if len(documents) != initial_count:
            logger.info(
                "Related-section expansion changed context from %d to %d chunks.",
                initial_count,
                len(documents),
            )
        return documents

    def _format_context(self, documents: list[Document]) -> str:
        """Format unique source chunks for the prompt within a character budget."""
        sections: list[str] = []
        seen: set[str] = set()
        used_characters = 0
        character_budget = 10_000

        for document in documents:
            content = document.page_content.strip()
            if not content or content in seen:
                continue

            seen.add(content)
            source = document.metadata.get("source", "unknown source")
            section = f"[Source: {source}]\n{content}"
            if sections and used_characters + len(section) > character_budget:
                break
            sections.append(section)
            used_characters += len(section)

        context = "\n\n---\n\n".join(sections)
        logger.info(
            "Formatted context: %d source section(s), %d characters (character budget=%d).",
            len(sections),
            len(context),
            character_budget,
        )
        return context