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
    chroma_upserter = ChromaUpserter(api_endpoint="http://localhost:8000/update_db_update")
    chroma_upserter.upsert_all_conversations()

    return {"message": "Database updated successfully."}


def categorize() -> Dict[str, str]:
    """
    Categorize messages in the database.

    Runs the CategorizerEngine to categorize messages
    and update their status in the database.
    """
    raise NotImplementedError(
        "This endpoint is not implemented yet. Use /categorizer_update to update job info."
    )

    return {"message": "Categorized successfully."}
