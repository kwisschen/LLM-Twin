from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.crawlers.linkedin_csv_loader import LinkedInCSVLoader
from llm_engineering.domain.documents import UserDocument


@step
def load_linkedin_csv(user: UserDocument, export_dir: str) -> Annotated[dict, "load_stats"]:
    logger.info(f"Loading LinkedIn CSV data for {user.full_name} from {export_dir}")

    loader = LinkedInCSVLoader(export_dir=export_dir)
    loader.extract(link=f"https://www.linkedin.com/in/{user.first_name.lower()}/", user=user)

    stats = _collect_stats(loader, user)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="load_stats", metadata=stats)

    logger.info(f"LinkedIn CSV load complete: {stats}")

    return stats


def _collect_stats(loader: LinkedInCSVLoader, user: UserDocument) -> dict:
    """Query MongoDB to count what was loaded."""

    from llm_engineering.infrastructure.db.mongo import connection
    from llm_engineering.settings import settings

    db = connection[settings.DATABASE_NAME]

    posts_count = db["posts"].count_documents({"author_id": str(user.id), "platform": "linkedin"})
    user_doc = db["users"].find_one({"_id": str(user.id)})

    return {
        "user": user.full_name,
        "linkedin_posts": posts_count,
        "positions": len(user_doc.get("positions", [])) if user_doc else 0,
        "education": len(user_doc.get("education", [])) if user_doc else 0,
        "skills": len(user_doc.get("skills", [])) if user_doc else 0,
        "certifications": len(user_doc.get("certifications", [])) if user_doc else 0,
        "has_headline": bool(user_doc.get("headline")) if user_doc else False,
        "has_summary": bool(user_doc.get("summary")) if user_doc else False,
    }
