from typing import Dict

# from backend.processing.categorizer import CategorizerEngine
from backend.loaders.chroma_endpoints import ChromaUpserter


def update_db() -> Dict[str, str]:
    """
    Update the database contents.

    Runs the ChromaUpserter to add any missing conversations
    to the chroma database.
    """

    # Upsert the categorized data into the database
    chroma_upserter = ChromaUpserter()
    chroma_upserter.upsert_all_conversations()

    return {"message": "Database updated successfully."}
