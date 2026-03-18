from llm_engineering import settings
from pymongo import MongoClient
from qdrant_client import QdrantClient
from loguru import logger


def verify_mongodb() -> bool:
    """Verify MongoDB connectivity and report collection counts."""
    print("\n--- MongoDB ---")
    try:
        client = MongoClient(settings.DATABASE_HOST, serverSelectionTimeoutMS=5000)
        client.server_info()  # forces connection attempt
        db = client[settings.DATABASE_NAME]

        collections = {
            "repositories": db.repositories.count_documents({}),
            "articles": db.articles.count_documents({}),
            "posts": db.posts.count_documents({}),
            "conversations": db.conversations.count_documents({}),
        }

        print(f"Host:     {settings.DATABASE_HOST}")
        print(f"Database: {settings.DATABASE_NAME}")
        for name, count in collections.items():
            print(f"  {name}: {count} documents")

        return True

    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        return False


def verify_qdrant() -> bool:
    """Verify Qdrant connectivity and report collection counts."""
    print("\n--- Qdrant ---")
    try:
        client = QdrantClient(
            host=settings.QDRANT_DATABASE_HOST,
            port=settings.QDRANT_DATABASE_PORT,
            timeout=5,
        )
        collections = client.get_collections().collections
        names = [c.name for c in collections]

        print(f"Host: {settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}")
        print(f"Collections ({len(names)}):")
        for name in sorted(names):
            info = client.get_collection(name)
            print(f"  {name}: {info.points_count} points")

        return True

    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Infrastructure Verification ===")
    print(f"--- Configuration ---")
    print(f"DATABASE_HOST: {settings.DATABASE_HOST}")
    print(f"DATABASE_NAME: {settings.DATABASE_NAME}")

    mongo_ok = verify_mongodb()
    qdrant_ok = verify_qdrant()

    print("\n=== Summary ===")
    print(f"MongoDB:  {'✅ PASS' if mongo_ok else '❌ FAIL'}")
    print(f"Qdrant:   {'✅ PASS' if qdrant_ok else '❌ FAIL'}")

    if not (mongo_ok and qdrant_ok):
        raise SystemExit(1)