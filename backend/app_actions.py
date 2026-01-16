from pathlib import Path
from typing import Dict, List, Tuple

import requests
from loguru import logger

from backend.loaders.chroma_endpoints import ChromaQuerier, ChromaUpserter
from backend.models.app_models import (
    CategorizerJobInfo,
    JobStatus,
    QueryDatabaseModel,
    UpdaterJobInfo,
)
from backend.models.message_node import MessageNode
from backend.processing.categorizer import categorize
from backend.utils.log_setup import LoggerSetup
from backend.utils.config_utils import get_config

CONFIG = get_config()

API_ENDPOINT = f"{CONFIG.API.HOST}:{CONFIG.API.PORT}"
UPDATE_DB_ENDPOINT = f"{API_ENDPOINT}/{CONFIG.API.ENDPOINTS.UPDATE_DB}"


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


def app_categorize(log_path: Path):
    """
    Categorize messages in the database.

    Runs the CategorizerEngine to categorize messages
    and update their status in the database.
    """

    # Set up logging
    logger_setup = LoggerSetup()
    logger_setup.configure_logger(force_log_file=log_path, console_level="DEBUG")

    try:
        categorize(log_path=log_path, api_endpoint=API_ENDPOINT, console_log_level="DEBUG")
    except Exception as e:
        # Listen for errors during execution and notify the user
        logger.error(f"Failed to categorize messages: {e}")
        failed_categorization = CategorizerJobInfo(
            status=JobStatus.FAILED,
        )
        requests.post(
            f"{API_ENDPOINT}/categorizer/update",
            json=failed_categorization.model_dump(),
        )

    return None


def search_database(
    query: QueryDatabaseModel,
    log_path: Path,
) -> Tuple[ChromaQuerier, Dict[str, List[MessageNode]]]:
    """
    Search the database for messages matching the given query.

    Args:
        query (QueryDatabaseModel): The query parameters for searching the database.
        log_path (Path): The path to the log file.

    Returns:
        Tuple[ChromaQuerier, Dict[str, List[MessageNode]]]:
            A tuple containing the ChromaQuerier instance and a dictionary mapping categories to lists of MessageNode objects.
    """  # noqa: E501
    # Set up logging
    logger_setup = LoggerSetup()
    logger_setup.configure_logger(force_log_file=log_path)

    # Initialize the ChromaQuerier
    chroma_querier = ChromaQuerier()

    try:
        sorted_nodes: Dict[str, List[MessageNode]] = chroma_querier.query_sorted_nodes(query)
    except Exception as e:
        # Listen for errors during execution and notify the user
        logger.error(f"Failed to search database: {e}")
        sorted_nodes: Dict[str, List[MessageNode]] = dict()

    return chroma_querier, sorted_nodes
