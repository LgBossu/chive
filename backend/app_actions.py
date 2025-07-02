from pathlib import Path

# from backend.processing.categorizer import CategorizerEngine
from backend.loaders.chroma_endpoints import ChromaUpserter
from backend.utils.log_setup import LoggerSetup


def update_db(log_path: Path):
    """
    Update the database contents.

    Runs the ChromaUpserter to add any missing conversations
    to the chroma database.
    """
    # Set up logging
    logger_setup = LoggerSetup()
    logger_setup.configure_logger(force_log_file=log_path)

    # Upsert the categorized data into the database
    chroma_upserter = ChromaUpserter(
        api_endpoint="http://127.0.0.1:8000/update_db/update"
    )  # TODO : do not hardcode endpoints
    chroma_upserter.upsert_all_conversations()

    return None


def categorize(log_path: Path):
    """
    Categorize messages in the database.

    Runs the CategorizerEngine to categorize messages
    and update their status in the database.
    """
    raise NotImplementedError(
        "This endpoint is not implemented yet. Use /categorizer_update to update job info."
    )

    return None
