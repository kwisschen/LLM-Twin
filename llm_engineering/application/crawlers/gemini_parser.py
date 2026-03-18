import json
from datetime import datetime
from pathlib import Path

import html2text
from loguru import logger

from llm_engineering.domain.documents import ConversationDocument, UserDocument


class GeminiParser:
    def __init__(self, export_path: str) -> None:
        self._export_path = Path(export_path).expanduser()
        self._html_converter = html2text.HTML2Text()
        self._html_converter.ignore_links = True
        self._html_converter.ignore_images = True

    def extract(self, user: UserDocument) -> tuple[int, int]:
        """Parse Gemini MyActivity.json and save as ConversationDocuments.

        Returns:
            (loaded, skipped) counts.
        """
        if not self._export_path.exists():
            logger.warning(f"Gemini export not found: {self._export_path}")
            return 0, 0

        with self._export_path.open("r", encoding="utf-8") as f:
            activities = json.load(f)

        loaded, skipped = 0, 0
        for item in activities:
            title = item.get("title", "")
            prompt = title.removeprefix("Prompted ").strip()

            if not prompt:
                skipped += 1
                continue

            # Extract response from safeHtmlItem
            response = ""
            safe_items = item.get("safeHtmlItem", [])
            if safe_items and isinstance(safe_items, list):
                html_content = safe_items[0].get("html", "")
                response = self._html_converter.handle(html_content).strip()

            # Parse ISO 8601 timestamp
            time_str = item.get("time", "")
            created_at = None
            if time_str:
                try:
                    created_at = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

            doc = ConversationDocument(
                content={"prompt": prompt, "response": response},
                platform="gemini",
                author_id=user.id,
                author_full_name=user.full_name,
                conversation_id="",
                created_at=created_at,
            )
            doc.save()
            loaded += 1

        logger.info(f"Gemini: loaded={loaded}, skipped={skipped}")
        return loaded, skipped
