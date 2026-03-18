import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from llm_engineering.domain.documents import ConversationDocument, UserDocument


class ChatGPTParser:
    def __init__(self, export_path: str) -> None:
        self._export_path = Path(export_path).expanduser()

    def extract(self, user: UserDocument) -> tuple[int, int]:
        """Parse ChatGPT conversations.json and save as ConversationDocuments.

        Returns:
            (loaded, skipped) counts.
        """
        if not self._export_path.exists():
            logger.warning(f"ChatGPT export not found: {self._export_path}")
            return 0, 0

        with self._export_path.open("r", encoding="utf-8") as f:
            conversations = json.load(f)

        loaded, skipped = 0, 0
        for conv in conversations:
            conv_id = conv.get("conversation_id", conv.get("id", ""))
            mapping = conv.get("mapping", {})

            pairs = self._extract_pairs(mapping)
            for prompt, response, created_at in pairs:
                if not prompt.strip():
                    skipped += 1
                    continue

                doc = ConversationDocument(
                    content={"prompt": prompt, "response": response},
                    platform="chatgpt",
                    author_id=user.id,
                    author_full_name=user.full_name,
                    conversation_id=conv_id,
                    created_at=created_at,
                )
                doc.save()
                loaded += 1

        logger.info(f"ChatGPT: loaded={loaded}, skipped={skipped}")
        return loaded, skipped

    @staticmethod
    def _extract_pairs(mapping: dict) -> list[tuple[str, str, datetime | None]]:
        """Walk the conversation tree and extract (prompt, response, timestamp) pairs."""
        pairs = []

        # Build node lookup
        nodes = {}
        for node_id, node in mapping.items():
            msg = node.get("message")
            if msg is None:
                continue
            role = msg.get("author", {}).get("role")
            content = msg.get("content", {})
            parts = content.get("parts", [])
            text = " ".join(str(p) for p in parts if isinstance(p, str)).strip()
            create_time = msg.get("create_time")
            timestamp = datetime.fromtimestamp(create_time, tz=timezone.utc) if create_time else None
            nodes[node_id] = {"role": role, "text": text, "timestamp": timestamp, "children": node.get("children", [])}

        # Find user→assistant pairs by walking children
        for _node_id, node in nodes.items():
            if node["role"] != "user" or not node["text"]:
                continue

            # Look for assistant response among children
            response_text = ""
            response_ts = node["timestamp"]
            for child_id in node["children"]:
                child = nodes.get(child_id)
                if child and child["role"] == "assistant" and child["text"]:
                    response_text = child["text"]
                    response_ts = child["timestamp"] or response_ts
                    break

            pairs.append((node["text"], response_text, response_ts))

        return pairs
