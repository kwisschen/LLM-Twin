import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Generic, TypeVar

from loguru import logger

from llm_engineering.domain.cleaned_documents import (
    CleanedArticleDocument,
    CleanedConversationDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)
from llm_engineering.domain.documents import (
    ArticleDocument,
    ConversationDocument,
    Document,
    PostDocument,
    RepositoryDocument,
)

from .operations import clean_text

DocumentT = TypeVar("DocumentT", bound=Document)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


def _is_garbled(text: str, is_code: bool = False) -> bool:
    """
    Quality gate: returns True if text is likely binary/garbled content.

    Four layered checks, with category-aware thresholds for code vs prose:
      1. Non-ASCII ratio     — 0.15 prose / 0.15 code (same — binary headers are binary regardless)
      2. Whitespace density  — prose only (skipped for code — concatenated source files are whitespace-sparse by nature)
      3. Repetition check    — same 4-char n-gram dominates the text
      4. Shannon entropy     — prose: (1.5, 5.8) / code: (1.5, 6.2) — code has higher legitimate entropy

    Natural language entropy: ~3.5–5.5 bits/char.
    Source code entropy:      ~4.5–6.0 bits/char.
    Base64/random bytes:      ~6.0–7.5 bits/char.
    Pure repetition:          ~0–1.5 bits/char.
    """
    if not text or len(text.strip()) == 0:
        return True

    # ── Check 1: Non-ASCII ratio ──────────────────────────────────────────────
    # Binary headers and encoding artifacts are invalid for both prose and code.
    # CJK content in otherwise-ASCII repos also caught here (patent-translator-core).
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if non_ascii / len(text) > 0.15:
        return True

    # ── Check 2: Whitespace density (prose only) ──────────────────────────────
    # Skipped for code: ETL concatenates source files without spaces, producing
    # avg_word_len of 40–95 for legitimate repos. This is an ETL artifact, not
    # a signal of garbled content.
    if not is_code and len(text) > 200:
        words = text.split()
        avg_word_len = len(text.replace(" ", "")) / max(len(words), 1)
        if avg_word_len > 40:
            return True

    # ── Check 3: Repetition check ─────────────────────────────────────────────
    # Catches ===...=== / ---...--- floods in both prose and code.
    if len(text) > 100:
        ngrams = [text[i : i + 4] for i in range(len(text) - 3)]
        if ngrams:
            most_common_count = Counter(ngrams).most_common(1)[0][1]
            if most_common_count / len(ngrams) > 0.25:
                return True

    # ── Check 4: Shannon entropy ──────────────────────────────────────────────
    # Code has higher legitimate entropy than prose due to identifiers,
    # operators, and punctuation diversity. Ceiling raised 5.8 → 6.2 for code.
    if len(text) > 100:
        freq = Counter(text)
        total = len(text)
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
        entropy_ceiling = 6.2 if is_code else 5.8
        if entropy > entropy_ceiling or entropy < 1.5:
            return True

    return False


class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass


class PostCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: PostDocument) -> CleanedPostDocument:
        raw = " ".join(data_model.content.values())
        if _is_garbled(raw):
            logger.warning(f"Garbled content detected in PostDocument id={data_model.id}, skipping.")
            raise ValueError(f"Garbled content in PostDocument id={data_model.id}")

        return CleanedPostDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(data_model.content.values())),
            platform=data_model.platform,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            image=data_model.image if data_model.image else None,
        )


class ArticleCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: ArticleDocument) -> CleanedArticleDocument:
        valid_content = [content for content in data_model.content.values() if content]
        raw = " ".join(valid_content)
        if _is_garbled(raw):
            logger.warning(f"Garbled content detected in ArticleDocument id={data_model.id}, skipping.")
            raise ValueError(f"Garbled content in ArticleDocument id={data_model.id}")

        return CleanedArticleDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(valid_content)),
            platform=data_model.platform,
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: RepositoryDocument) -> CleanedRepositoryDocument:
        raw = " ".join(data_model.content.values())
        if _is_garbled(raw, is_code=True):
            logger.warning(f"Garbled content detected in RepositoryDocument id={data_model.id}, skipping.")
            raise ValueError(f"Garbled content in RepositoryDocument id={data_model.id}")

        return CleanedRepositoryDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(data_model.content.values())),
            platform=data_model.platform,
            name=data_model.name,
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )


class ConversationCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: ConversationDocument) -> CleanedConversationDocument:
        prompt = data_model.content.get("prompt", "")
        response = data_model.content.get("response", "")
        raw = " ".join(filter(None, [prompt, response]))
        if _is_garbled(raw):
            logger.warning(f"Garbled content detected in ConversationDocument id={data_model.id}, skipping.")
            raise ValueError(f"Garbled content in ConversationDocument id={data_model.id}")

        joined = " #### ".join(filter(None, [prompt, response]))

        return CleanedConversationDocument(
            id=data_model.id,
            content=clean_text(joined),
            platform=data_model.platform,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )