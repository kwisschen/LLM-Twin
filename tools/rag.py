from langchain.globals import set_verbose
from loguru import logger

from llm_engineering.application.rag.retriever import ContextRetriever

try:
    from llm_engineering.infrastructure.opik_utils import configure_opik

    configure_opik()
except Exception as e:
    logger.warning(f"Opik configuration failed (non-fatal, tracing disabled): {e}")

if __name__ == "__main__":
    set_verbose(True)

    query = """
        My name is Paul Iusztin.
        
        Could you draft a LinkedIn post discussing RAG systems?
        I'm particularly interested in:
            - how RAG works
            - how it is integrated with vector DBs and large language models (LLMs).
        """

    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=9)

    logger.info("Retrieved documents:")
    for rank, document in enumerate(documents):
        logger.info(f"{rank + 1}: {document}")
