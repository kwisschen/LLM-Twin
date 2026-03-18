import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from llm_engineering.domain.chunks import ArticleChunk, Chunk, ConversationChunk, PostChunk, RepositoryChunk
from llm_engineering.domain.cleaned_documents import (
    CleanedArticleDocument,
    CleanedConversationDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)

from .operations import chunk_article, chunk_text

CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
ChunkT = TypeVar("ChunkT", bound=Chunk)


class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]:
        pass


class PostChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 250,
            "chunk_overlap": 25,
        }

    def chunk(self, data_model: CleanedPostDocument) -> list[PostChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(
            cleaned_content, chunk_size=self.metadata["chunk_size"], chunk_overlap=self.metadata["chunk_overlap"]
        )

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = PostChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                image=data_model.image if data_model.image else None,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list


class ArticleChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "min_length": 1000,
            "max_length": 2000,
        }

    def chunk(self, data_model: CleanedArticleDocument) -> list[ArticleChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_article(
            cleaned_content, min_length=self.metadata["min_length"], max_length=self.metadata["max_length"]
        )

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = ArticleChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                link=data_model.link,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list


class RepositoryChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "chunk_strategy": "per_file",
        }

    def chunk(self, data_model: CleanedRepositoryDocument) -> list[RepositoryChunk]:
        """
        Chunk repository content per file rather than as a joined blob.

        Joining all files into one string and splitting produces chunks that
        span multiple files, lose file context, and exceed the embedding model's
        8192 token limit for large repos. Per-file chunking preserves semantic
        boundaries — a file is the natural atomic unit in a repository, just as
        a Q&A pair is the atomic unit in a conversation.
        """
        data_models_list = []

        # data_model.content is the full joined string: "file1_content #### file2_content #### ..."
        # Split it back into per-file segments using the #### separator inserted during cleaning
        file_segments = [seg.strip() for seg in data_model.content.split("####") if seg.strip()]

        for segment in file_segments:
            # Apply token-based chunking within each file segment
            # 500 tokens ≈ 2000 chars — safely under the 8192 token limit
            chunks = chunk_text(
                segment,
                chunk_size=self.metadata["chunk_size"],
                chunk_overlap=self.metadata["chunk_overlap"],
            )

            for chunk in chunks:
                if not chunk.strip():
                    continue
                chunk_id = hashlib.md5(chunk.encode()).hexdigest()
                model = RepositoryChunk(
                    id=UUID(chunk_id, version=4),
                    content=chunk,
                    platform=data_model.platform,
                    name=data_model.name,
                    link=data_model.link,
                    document_id=data_model.id,
                    author_id=data_model.author_id,
                    author_full_name=data_model.author_full_name,
                    metadata=self.metadata,
                )
                data_models_list.append(model)

        return data_models_list


class ConversationChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {"chunk_strategy": "atomic_pair"}

    def chunk(self, data_model: CleanedConversationDocument) -> list[ConversationChunk]:
        content = data_model.content
        if not content.strip():
            return []

        chunk_id = hashlib.md5(content.encode()).hexdigest()

        # Extract the prompt portion (before " #### " separator) for embedding
        parts = content.split(" #### ", maxsplit=1)
        prompt_text = parts[0].strip()

        model = ConversationChunk(
            id=UUID(chunk_id, version=4),
            content=content,
            platform=data_model.platform,
            document_id=data_model.id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={**self.metadata, "prompt": prompt_text},
        )

        return [model]
