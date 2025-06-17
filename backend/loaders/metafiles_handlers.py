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


class LegacyLinker(Linker):
    def __init__(self, legacy_db_path: str, legacy_metafiles_path: Tuple[str, str]):
        # Legacy metafile is a pair of CSV files, not a proper database.
        # first string is the path to the CSV file with tags, second is the blacklist
        super().__init__(legacy_db_path)

        self.legacy_metafiles_path = (
            Path(legacy_metafiles_path[0]),
            Path(legacy_metafiles_path[1]),
        )

        logger.debug(
            f"Initializing LegacyLinker with db_path: {self.past_db_path} and metafiles: {self.legacy_metafiles_path}"  # noqa: E501
        )

        assert (
            self.legacy_metafiles_path[0].exists() and self.legacy_metafiles_path[1].exists()
        ), f"Legacy metafiles paths {self.legacy_metafiles_path} do not exist."

    def get_taglist(self) -> np.ndarray:
        """
        Return the legacy taglist from metafiles as a numpy array.

        The first column contains message IDs, and the second column contains tags.
        If a message is blacklisted, it will have the tag "######" in the second column.

        :return: A 2D numpy array with message IDs and tags.
        :rtype: np.ndarray
        """
        logger.debug("Loading legacy taglist from metafiles")
        legacy_taglist = []
        with open(self.legacy_metafiles_path[0], "r") as tagfile:
            csv_reader = reader(tagfile, delimiter=",")
            for row in csv_reader:
                legacy_taglist.append(row)
        with open(self.legacy_metafiles_path[1], "r") as blacklistfile:
            for line in blacklistfile:
                legacy_taglist.append([line[:-1], BLACKLIST_TAG])
                # Remove newline character and add a specific tag for blacklisted items

        return np.array(legacy_taglist, dtype=str)

    def link_past_tags(self) -> Dict[str, List[str]]:
        """Link past tags to the past messages' document (raw text) content."""
        logger.debug("Linking past tags to past messages")
        taglist = self.get_taglist()
        past_messages, _ = self.get_past_messages()

        # Create a mapping from message content to tags
        message_to_tags: Dict[str, List[str]] = dict()

        for message, id_mess in past_messages:
            message = str(message)
            id_mess = str(id_mess)

            if message not in message_to_tags:
                message_to_tags[message] = []

            # Find tags for the current message
            tags_strings: List[str] = list(taglist[taglist[:, 0] == id_mess, 1].flatten())
            all_tags = clean_tags(tags_strings)

            # Add tags to the message
            message_to_tags[message].extend(all_tags)

        logger.debug(f"Linked tags to {len(message_to_tags)} messages")
        return message_to_tags
