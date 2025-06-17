import sqlite3
from abc import ABC, abstractmethod
from csv import reader
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import numpy as np
from chroma_endpoints import ChromaQuerier
from loguru import logger

BLACKLIST_TAG = "######"
EMPTY_TAG = "##EMPTY##"
NO_TAGS_TAG = "##NO_TAGS##"


def clean_tags(raw_tags_lists: List[str]) -> List[str]:
    """
    Clean and deduplicate tags from a list of aggregated tags.

    :param raw_tags_lists: List of raw tags (semicolon separated).
    :return: A list of cleaned and deduplicated tags.
    """
    all_tags = ";".join(raw_tags_lists)  # Join all tags into a single string
    all_tags = all_tags.split(";")  # Split by semicolon to get individual tags
    all_tags = [tag.strip() for tag in all_tags if tag.strip()]  # Clean up tags
    all_tags = list(set(all_tags))  # Remove duplicates

    return all_tags if all_tags else [NO_TAGS_TAG]


class Linker(ABC):
    def __init__(self, past_db_path: str):
        """Initialize the Linker with a path to the past database."""
        self.past_db_path = Path(past_db_path)
        assert self.past_db_path.exists(), f"Past database path {self.past_db_path} does not exist."

    @abstractmethod
    def get_taglist(self) -> np.ndarray:
        """
        Return the taglist as a numpy array.

        On the first coordinate, the message ID.
        On the second coordinate, the tag list.
        """
        pass

    def get_past_messages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return messages from the past database.

        :return: A tuple of two numpy arrays:
            - The first array contains past messages (raw text).
            - The second array contains empty messages (if any).
        :rtype: Tuple[np.ndarray, np.ndarray]
        :raises ValueError: If no past messages are found in the database.
        """
        logger.debug("Connecting to the past database")
        client = chromadb.PersistentClient(path=str(self.past_db_path))

        logger.debug("Fetching past messages from the database")
        past_querier = ChromaQuerier(client)
        past_messages = past_querier.get_all_nonempty_messages()
        past_empty_messages = past_querier.get_all_empty_messages()

        logger.debug(f"Fetched {len(past_messages)} past messages")
        if len(past_messages) == 0:
            logger.error("No past messages found in the database.")
            raise ValueError("No past messages found in the database.")

        return np.array(past_messages, dtype=str), np.array(past_empty_messages, dtype=str)

    @abstractmethod
    def link_past_tags(self) -> Dict[str, List[str]]:
        """
        Link past tags to the past messages' document (raw text) content.

        Returns a dictionary where:
        - The key is the message content (raw text).
        - The value is a list of tags associated with that message.
        """
        pass
