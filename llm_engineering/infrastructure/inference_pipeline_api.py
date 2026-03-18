from fastapi import FastAPI, HTTPException
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk

app = FastAPI()


# Alpaca prompt template — must match SFT training template exactly
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


def build_prompt(query: str, context: str | None) -> str:
    """Build prompt matching the Alpaca template used during SFT training."""
    system_content = (
        "You are Christopher Chen, a technical content creator. "
        "Respond directly to the following request. Use the provided "
        "context as your primary source of information. Write in a "
        "detailed, engaging style."
    )

    if context:
        instruction = f"{system_content}\n\nUser query: {query}\nContext: {context}"
    else:
        instruction = f"{system_content}\n\nUser query: {query}"

    return ALPACA_TEMPLATE.format(instruction=instruction)


def call_llm_service(query: str, context: str | None) -> str:
    client = OpenAI(
        base_url=f"http://{settings.VLLM_HOST}:{settings.VLLM_PORT}/v1",
        api_key="not-needed",
    )

    prompt = build_prompt(query, context)

    response = client.completions.create(
        model=settings.VLLM_MODEL_ID,
        prompt=prompt,
        max_tokens=settings.MAX_NEW_TOKENS_INFERENCE,
        temperature=settings.TEMPERATURE_INFERENCE,
        extra_body={"repetition_penalty": 1.1},
    )

    return response.choices[0].text.strip()


def rag(query: str) -> str:
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

    logger.info(
        f"RAG retrieval complete: {len(documents)} documents, "
        f"query_tokens={misc.compute_num_tokens(query)}, "
        f"context_tokens={misc.compute_num_tokens(context)}"
    )

    answer = call_llm_service(query, context)

    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
