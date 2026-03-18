import re

from langchain.text_splitter import RecursiveCharacterTextSplitter

from llm_engineering.application.networks import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()


def _split_by_tokens(text: str, tokens_per_chunk: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks using the embedding model's own tokenizer.

    Replaces SentenceTransformersTokenTextSplitter which redundantly reloads the
    SentenceTransformer model without trust_remote_code, breaking Nomic and other
    custom-code models.
    """
    tokenizer = embedding_model.tokenizer
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= tokens_per_chunk:
        return [text]

    step = max(tokens_per_chunk - chunk_overlap, 1)
    chunks = []
    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start : start + tokens_per_chunk]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        if start + tokens_per_chunk >= len(token_ids):
            break

    return chunks


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0)
    text_split_by_characters = character_splitter.split_text(text)

    chunks_by_tokens = []
    for section in text_split_by_characters:
        chunks_by_tokens.extend(
            _split_by_tokens(section, tokens_per_chunk=chunk_size, chunk_overlap=chunk_overlap)
        )

    return chunks_by_tokens


def chunk_document(text: str, min_length: int, max_length: int) -> list[str]:
    """Alias for chunk_article()."""

    return chunk_article(text, min_length, max_length)


def chunk_article(text: str, min_length: int, max_length: int) -> list[str]:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

    extracts = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())
            current_chunk = sentence + " "

    if len(current_chunk) >= min_length:
        extracts.append(current_chunk.strip())

    return extracts
