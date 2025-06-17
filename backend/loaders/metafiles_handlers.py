import sqlite3
from abc import ABC, abstractmethod
from csv import reader
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import numpy as np
from loguru import logger

from backend.loaders.chroma_endpoints import ChromaQuerier

BLACKLIST_TAG = "######"
EMPTY_TAG = "##EMPTY##"
NO_TAGS_TAG = "##NO_TAGS##"

NOT_TAGGED = True


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
    def __init__(self, past_db_path: str | Path):
        """Initialize the Linker with a path to the past database."""
        if isinstance(past_db_path, str):
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
    def link_past_to_tags(self) -> Dict[str, List[str]]:
        """
        Link past tags to the past messages' document (raw text) content.

        Returns a dictionary where:
        - The key is the message content (raw text).
        - The value is a list of tags associated with that message.
        """
        pass

    def get_current_messages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current messages from the database.

        This method retrieves all messages from the current database, which is expected to be
        a ChromaDB collection.

        :return: A 2D numpy array with message IDs and content.
        :rtype: np.ndarray
        """
        logger.debug("Connecting to the current database")
        current_querier = ChromaQuerier()

        logger.debug("Fetching current messages from the database")
        current_nonempty_messages = current_querier.get_all_nonempty_messages()
        current_empty_messages = current_querier.get_all_empty_messages()
        logger.debug(f"Fetched {len(current_nonempty_messages)} current non-empty messages")
        logger.debug(f"Fetched {len(current_empty_messages)} current empty messages")
        if len(current_nonempty_messages) == 0:
            logger.error("No current messages found in the database.")
            raise ValueError("No current messages found in the database.")

        return (
            np.array(current_nonempty_messages, dtype=str),
            np.array(current_empty_messages, dtype=str),
        )

    def link_current_to_tags(self, message_to_tags: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Link current messages to tags.

        This method takes a dictionary mapping message content to lists of tags
        and returns a dictionary where the keys are message IDs and the values are lists of tags
        associated with those messages.

        :param message_to_tags: A dictionary mapping message content to lists of tags.
        :return: A dictionary mapping message IDs to lists of tags.
        :rtype: Dict[str, List[str]]
        """
        current_to_tags: Dict[str, List[str]] = dict()

        current_messages, current_empty_messages = self.get_current_messages()

        for empty_id in current_empty_messages:
            current_to_tags[str(empty_id)] = [EMPTY_TAG]
        for current_message, current_id in current_messages:
            current_message = str(current_message)
            current_id = str(current_id)

            tags = message_to_tags.get(current_message, None)
            if tags is None:
                continue  # Skip if no tags found for the message, it will be tagged on future runs
            current_to_tags[current_id] = []
            current_to_tags[current_id].extend(tags.copy())

        return current_to_tags

    def pipeline(self) -> Dict[str, List[str]]:
        """
        Main pipeline method to link past messages to tags and return the mapping.

        This method orchestrates the linking of past messages to tags and returns a dictionary
        mapping message content to lists of tags.

        :return: A dictionary mapping message content to lists of tags.
        :rtype: Dict[str, List[str]]
        """
        logger.debug("Starting the linking pipeline")
        message_to_tags = self.link_past_to_tags()
        logger.debug("Linking current messages to tags")
        current_to_tags = self.link_current_to_tags(message_to_tags)
        logger.debug("Linking pipeline completed")
        return current_to_tags


class LegacyLinker(Linker):
    def __init__(
        self,
        legacy_db_path: str | Path,
        legacy_metafiles_path: Tuple[str, str] | Tuple[Path, Path],
    ) -> None:
        # Legacy metafile is a pair of CSV files, not a proper database.
        # first string is the path to the CSV file with tags, second is the blacklist
        super().__init__(legacy_db_path)

        if isinstance(legacy_metafiles_path[0], str):
            legacy_metafiles_path = (Path(legacy_metafiles_path[0]), Path(legacy_metafiles_path[1]))
        self.legacy_metafiles_path = legacy_metafiles_path
        assert (
            isinstance(self.legacy_metafiles_path[0], Path)
            and isinstance(self.legacy_metafiles_path[1], Path)  # noqa: E501
        ), (
            "Legacy metafiles paths must be Path objects."
        )  # Keeping the linter happy, for clarity and maintainability  # noqa: E501

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

    def link_past_to_tags(self) -> Dict[str, List[str]]:
        """
        Link past tags to the past messages' document (raw text) content.

        This method reads the legacy metafiles, processes the tags, and returns a dictionary
        where the keys are message contents and the values are lists of tags associated with those
        messages.

        :return: A dictionary mapping message content to lists of tags.
        :rtype: Dict[str, List[str]]
        :raises ValueError: If no past messages are found in the database.
        """
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

        logger.debug(f"Linked tags to {len(message_to_tags)} unique text messages")
        return message_to_tags


class MetafileWriter:
    """
    Class to handle generating and writing the new metafile contents.

    This class is responsible for creating the metafile content based on the linked tags
    and writing it to the proper database.
    """

    pass


if __name__ == "__main__":
    from pprint import pprint

    from backend.utils.log_setup import LoggerSetup
    from backend.utils.path_utils import get_paths

    LoggerSetup.configure_logger()

    logger.info("Starting Metafiles process...")
    logger.warning("""You are running the metafiles handling script directly.
                   This is intended for debugging purposes only.

                   This script is not intended for production use.
                   Users stay advised.""")

    # Get paths from the environment variables
    paths = get_paths()

    legacy_chroma = paths.legacy_chroma_dir / "V1 postprocess_chroma"
    legacy_metafiles = (paths.messages_categories, paths.blacklist_categories)

    # Debug run
    # TODO : at some point, ask for user input to proceed OR remove the debug run script.

    # Example usage
    legacy_linker = LegacyLinker(
        legacy_db_path=legacy_chroma,
        legacy_metafiles_path=legacy_metafiles,
    )

    logger.debug("Running the legacy linker pipeline")
    current_to_tags = legacy_linker.pipeline()
    logger.debug("Legacy linker pipeline completed")
    logger.debug("Current to tags mapping:")
    pprint(current_to_tags)
    logger.debug("Finished running the legacy linker")
