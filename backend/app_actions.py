from pathlib import Path

import requests
from loguru import logger

# from backend.processing.categorizer import CategorizerEngine
from backend.loaders.chroma_endpoints import ChromaUpserter
from backend.models.app_models import JobStatus, UpdaterJobInfo
from backend.utils.log_setup import LoggerSetup

UPDATE_DB_ENDPOINT = "http://127.0.0.1:8000/update_db/update"  # TODO : do not hardcode endpoints


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
    chroma_upserter = ChromaUpserter(api_endpoint=UPDATE_DB_ENDPOINT)
    try:
        chroma_upserter.upsert_all_conversations()
    except Exception as e:
        # Listen for errors during execution and notify the user
        logger.error(f"Failed to update database: {e}")
        failed_update = UpdaterJobInfo(
            status=JobStatus.FAILED,
            updated_conversations=[],
        )
        requests.post(
            UPDATE_DB_ENDPOINT,
            json=failed_update.model_dump(),
        )

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
