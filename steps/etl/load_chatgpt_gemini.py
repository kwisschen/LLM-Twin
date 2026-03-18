from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.crawlers.chatgpt_parser import ChatGPTParser
from llm_engineering.application.crawlers.gemini_parser import GeminiParser
from llm_engineering.domain.documents import UserDocument
from llm_engineering.infrastructure.db.mongo import connection
from llm_engineering.settings import settings


@step
def load_chatgpt_gemini(
    user: UserDocument,
    chatgpt_path: str,
    gemini_path: str,
) -> Annotated[dict, "load_stats"]:
    logger.info(f"Loading ChatGPT + Gemini data for {user.full_name}")

    chatgpt_parser = ChatGPTParser(export_path=chatgpt_path)
    chatgpt_loaded, chatgpt_skipped = chatgpt_parser.extract(user=user)

    gemini_parser = GeminiParser(export_path=gemini_path)
    gemini_loaded, gemini_skipped = gemini_parser.extract(user=user)

    stats = _collect_stats(user, chatgpt_loaded, chatgpt_skipped, gemini_loaded, gemini_skipped)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="load_stats", metadata=stats)

    logger.info(f"ChatGPT + Gemini load complete: {stats}")
    return stats


def _collect_stats(
    user: UserDocument,
    chatgpt_loaded: int,
    chatgpt_skipped: int,
    gemini_loaded: int,
    gemini_skipped: int,
) -> dict:
    db = connection[settings.DATABASE_NAME]
    total_conversations = db["conversations"].count_documents({"author_id": str(user.id)})

    return {
        "user": user.full_name,
        "chatgpt_loaded": chatgpt_loaded,
        "chatgpt_skipped": chatgpt_skipped,
        "gemini_loaded": gemini_loaded,
        "gemini_skipped": gemini_skipped,
        "total_conversations": total_conversations,
    }
