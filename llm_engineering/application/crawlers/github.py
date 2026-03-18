import os
import shutil
import subprocess
import tempfile

from loguru import logger

from llm_engineering.domain.documents import RepositoryDocument

from .base import BaseCrawler

# Binary and non-text extensions that should never be read as source content.
# Ingesting these as text produces null bytes, garbage entropy, and wastes
# embedding compute — root cause of dataset contamination (Decision 9).
_BINARY_EXTENSIONS = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".bmp", ".webp", ".tiff",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Audio / video
    ".mp3", ".mp4", ".wav", ".ogg", ".avi", ".mov", ".webm",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    # Compiled / binary
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".o", ".a", ".lib",
    # Data / DB
    ".db", ".sqlite", ".sqlite3", ".pkl", ".parquet", ".npy", ".npz",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    # Config formats that add noise without signal
    ".lock", ".toml",
    # Map / sourcemap files
    ".map",
}


def _is_binary_file(filepath: str) -> bool:
    """
    Sniff the first 8KB of a file for null bytes.
    Null bytes are the definitive signal of binary content — text files never
    contain them. Faster and more reliable than extension matching alone.
    """
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except (OSError, PermissionError):
        return True  # Unreadable file — treat as binary and skip


class GithubCrawler(BaseCrawler):
    model = RepositoryDocument

    def __init__(self, ignore=(".git",)) -> None:
        super().__init__()
        self._ignore = ignore

    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Repository already exists in the database: {link}")
            return

        logger.info(f"Starting scrapping GitHub repository: {link}")

        repo_name = link.rstrip("/").split("/")[-1]
        local_temp = tempfile.mkdtemp()

        try:
            os.chdir(local_temp)
            subprocess.run(["git", "clone", link])

            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])  # noqa: PTH118

            tree = {}
            skipped = 0
            for root, _, files in os.walk(repo_path):
                dir = root.replace(repo_path, "").lstrip("/")
                if dir.startswith(self._ignore):
                    continue

                for file in files:
                    _, ext = os.path.splitext(file)

                    # Extension blocklist — skip known binary formats
                    if ext.lower() in _BINARY_EXTENSIONS:
                        skipped += 1
                        continue

                    full_path = os.path.join(root, file)  # noqa: PTH118
                    file_path = os.path.join(dir, file)   # noqa: PTH118

                    # Null-byte sniff — catches binary files with non-standard
                    # or missing extensions (compiled assets, data blobs, etc.)
                    if _is_binary_file(full_path):
                        skipped += 1
                        continue

                    try:
                        with open(full_path, "r", encoding="utf-8", errors="strict") as f:
                            content = f.read()
                        tree[file_path] = content  # Preserve original whitespace
                    except (UnicodeDecodeError, OSError):
                        # File claims to be text but contains non-UTF-8 bytes — skip
                        skipped += 1
                        continue

            logger.info(f"Repository {repo_name}: {len(tree)} files ingested, {skipped} binary/non-UTF-8 files skipped.")

            user = kwargs["user"]
            instance = self.model(
                content=tree,
                name=repo_name,
                link=link,
                platform="github",
                author_id=user.id,
                author_full_name=user.full_name,
            )
            instance.save()

        except Exception:
            raise
        finally:
            shutil.rmtree(local_temp)

        logger.info(f"Finished scrapping GitHub repository: {link}")