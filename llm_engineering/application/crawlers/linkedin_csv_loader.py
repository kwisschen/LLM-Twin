import csv
import os
from pathlib import Path

from loguru import logger

from llm_engineering.domain.documents import PostDocument, UserDocument

from .base import BaseCrawler


class LinkedInCSVLoader(BaseCrawler):
    """Loads LinkedIn data from the official CSV export (Settings > Data Privacy > Get a copy of your data).

    Bypasses deprecated Selenium-based scraping entirely.
    """

    model = PostDocument

    def __init__(self, export_dir: str = "~/Documents/Exports/linkedin_export") -> None:
        self._export_dir = Path(export_dir).expanduser()

    def extract(self, link: str, **kwargs) -> None:
        user = kwargs["user"]

        shares_path = self._export_dir / "Shares.csv"
        comments_path = self._export_dir / "Comments.csv"
        profile_path = self._export_dir / "Profile.csv"

        shares_loaded, shares_skipped, shares_dupes = self._load_shares(shares_path, user)
        comments_loaded, comments_skipped, comments_dupes = self._load_comments(comments_path, user)
        self._update_profile(profile_path, user)

        logger.info(
            f"LinkedIn CSV import complete. "
            f"Shares: {shares_loaded} loaded, {shares_skipped} empty, {shares_dupes} duplicates. "
            f"Comments: {comments_loaded} loaded, {comments_skipped} empty, {comments_dupes} duplicates."
        )

    def _load_shares(self, path: Path, user: UserDocument) -> tuple[int, int, int]:
        """Parse Shares.csv → PostDocument entries in twin.posts."""

        if not path.exists():
            logger.warning(f"Shares file not found: {path}")
            return 0, 0, 0

        loaded, skipped, dupes = 0, 0, 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                commentary = (row.get("ShareCommentary") or "").strip()
                share_link = (row.get("ShareLink") or "").strip()

                if not commentary:
                    skipped += 1
                    continue

                if share_link and self.model.find(link=share_link) is not None:
                    dupes += 1
                    continue

                doc = PostDocument(
                    content={"text": commentary},
                    platform="linkedin",
                    author_id=user.id,
                    author_full_name=user.full_name,
                    link=share_link or None,
                )
                doc.save()
                loaded += 1

        logger.info(f"Shares: {loaded} loaded, {skipped} empty, {dupes} duplicates")
        return loaded, skipped, dupes

    def _load_comments(self, path: Path, user: UserDocument) -> tuple[int, int, int]:
        """Parse Comments.csv → PostDocument entries in twin.posts."""

        if not path.exists():
            logger.warning(f"Comments file not found: {path}")
            return 0, 0, 0

        loaded, skipped, dupes = 0, 0, 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                message = (row.get("Message") or "").strip()
                comment_link = (row.get("Link") or "").strip()

                if not message:
                    skipped += 1
                    continue

                if comment_link and self.model.find(link=comment_link) is not None:
                    dupes += 1
                    continue

                doc = PostDocument(
                    content={"text": message},
                    platform="linkedin",
                    author_id=user.id,
                    author_full_name=user.full_name,
                    link=comment_link or None,
                )
                doc.save()
                loaded += 1

        logger.info(f"Comments: {loaded} loaded, {skipped} empty, {dupes} duplicates")
        return loaded, skipped, dupes

    def _update_profile(self, path: Path, user: UserDocument) -> None:
        """Parse Profile.csv, Positions.csv, Education.csv, Skills.csv, and Certifications.csv.

        Updates the existing UserDocument in MongoDB with all enrichment fields.
        """

        update_fields = {}

        # Profile.csv → headline, summary
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                row = next(csv.DictReader(f), None)
            if row:
                update_fields["headline"] = (row.get("Headline") or "").strip()
                update_fields["summary"] = (row.get("Summary") or "").strip()
        else:
            logger.warning(f"Profile file not found: {path}")

        # Positions.csv → positions (list of dicts)
        positions_path = self._export_dir / "Positions.csv"
        if positions_path.exists():
            positions = []
            with open(positions_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    positions.append({
                        "company": (row.get("Company Name") or "").strip(),
                        "title": (row.get("Title") or "").strip(),
                        "description": (row.get("Description") or "").strip(),
                        "location": (row.get("Location") or "").strip(),
                        "started_on": (row.get("Started On") or "").strip(),
                        "finished_on": (row.get("Finished On") or "").strip(),
                    })
            update_fields["positions"] = positions
            logger.info(f"Positions: {len(positions)} entries parsed")
        else:
            logger.warning(f"Positions file not found: {positions_path}")

        # Education.csv → education (list of dicts)
        education_path = self._export_dir / "Education.csv"
        if education_path.exists():
            education = []
            with open(education_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    education.append({
                        "school": (row.get("School Name") or "").strip(),
                        "start_date": (row.get("Start Date") or "").strip(),
                        "end_date": (row.get("End Date") or "").strip(),
                        "notes": (row.get("Notes") or "").strip(),
                        "degree": (row.get("Degree Name") or "").strip(),
                        "activities": (row.get("Activities") or "").strip(),
                    })
            update_fields["education"] = education
            logger.info(f"Education: {len(education)} entries parsed")
        else:
            logger.warning(f"Education file not found: {education_path}")

        # Skills.csv → skills (list of strings)
        skills_path = self._export_dir / "Skills.csv"
        if skills_path.exists():
            skills = []
            with open(skills_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    name = (row.get("Name") or "").strip()
                    if name:
                        skills.append(name)
            update_fields["skills"] = skills
            logger.info(f"Skills: {len(skills)} entries parsed")
        else:
            logger.warning(f"Skills file not found: {skills_path}")

        # Certifications.csv → certifications (list of dicts)
        certs_path = self._export_dir / "Certifications.csv"
        if certs_path.exists():
            certs = []
            with open(certs_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    certs.append({
                        "name": (row.get("Name") or "").strip(),
                        "url": (row.get("Url") or "").strip(),
                        "authority": (row.get("Authority") or "").strip(),
                        "started_on": (row.get("Started On") or "").strip(),
                        "finished_on": (row.get("Finished On") or "").strip(),
                        "license_number": (row.get("License Number") or "").strip(),
                    })
            update_fields["certifications"] = certs
            logger.info(f"Certifications: {len(certs)} entries parsed")
        else:
            logger.warning(f"Certifications file not found: {certs_path}")

        if not update_fields:
            logger.warning("No profile data to update")
            return

        from llm_engineering.infrastructure.db.mongo import connection
        from llm_engineering.settings import settings as app_settings

        db = connection[app_settings.DATABASE_NAME]
        collection = db["users"]
        result = collection.update_one(
            {"_id": str(user.id)},
            {"$set": update_fields},
        )

        if result.modified_count:
            logger.info(f"Updated profile for {user.full_name}: {', '.join(update_fields.keys())}")
        else:
            logger.warning(f"No profile update for {user.full_name} (already current or not found)")
