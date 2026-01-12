import sqlite3
from abc import ABC, abstractmethod
from csv import reader
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import numpy as np
from loguru import logger

from backend.loaders.chroma_endpoints import ChromaQuerier
from backend.utils.path_utils import get_paths

# TODO // # WARNING :
# dynamic_tags is an FTS5 table used as a presence index (message_id → tags).
# This is convenient but not ideal: FTS5 is not optimized for key-based joins.
# Current TEMP-table JOIN logic works but may be suboptimal.
# Revisit only after streaming/batching and OOM issues are fully resolved.


# TODO : deeply analyze this module for potential memory leaks on linking operations.
# Up to estimated 8GB of RAM usage is measured on system monitor :
# Verify if this is due to module level manipulation, design issues, or other factors.

# Constant tags literals
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

    return all_tags if all_tags else [NO_TAGS_TAG]


class Linker(ABC):
    def __init__(self, past_db_path: str | Path):
        """Initialize the Linker with a path to the past database."""
        if isinstance(past_db_path, str):
            past_db_path = Path(past_db_path)
        self.past_db_path = past_db_path
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
        # TODO : refactor to bypass empty message querying,
        # as it is not used in the current implementation.
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
        for current_id, current_message in current_messages:
            current_message = str(current_message)
            current_id = str(current_id)

            tags = message_to_tags.get(current_message, None)
            if tags is None:
                logger.trace(
                    f"No tags found for message ID {current_id} with content: {current_message[:50]}..."  # noqa: E501
                )
                continue  # Skip if no tags found for the message, it will be tagged on future runs
            current_to_tags[current_id] = []
            current_to_tags[current_id].extend(tags.copy())

        logger.debug(f"Linked {len(current_to_tags)} current messages to tags")
        logger.debug(f"Among these, {len(current_empty_messages)} are likely empty messages")

        return current_to_tags

    def pipeline(self) -> Dict[str, List[str]]:
        """
        Main pipeline method to link past messages to tags and return the mapping.

        This method orchestrates the linking of past messages to tags and returns a dictionary
        mapping message content to lists of tags.

        :return: A dictionary mapping message content to lists of tags.
        :rtype: Dict[str, List[str]]
        """
        logger.info("Starting the linking pipeline")
        logger.info("Linking past messages to tags")
        message_to_tags = self.link_past_to_tags()
        logger.info("Linking current messages to tags")
        current_to_tags = self.link_current_to_tags(message_to_tags)
        logger.info("Linking pipeline completed")
        return current_to_tags
    
    def close(self) -> None:
        """
        Close any resources held by the Linker.

        This method should be called when the Linker is no longer needed.
        """
        pass


class LegacyLinker(Linker):
    """
    LegacyLinker is a subclass of Linker that handles legacy metafiles
    and links past messages to tags using a pair of CSV files.

    It is designed to work with legacy databases that do not use the
    modern ChromaDB structure.

    **This class is intended to be used for legacy data migration and
    CANNOT be used for new databases.**

    It reads tags from a pair of CSV files, where the first file contains
    message IDs and their associated tags, and the second file contains
    blacklisted message IDs with a specific tag.
    It provides methods to retrieve the taglist, fetch past messages,
    and link past messages to tags.
    """

    # For now, I'm keeping this class in the codebase, in case. You know.

    def __init__(
        self,
        legacy_db_path: str | Path,
        legacy_metafiles_path: Tuple[str, str] | Tuple[Path, Path],
    ) -> None:
        # Legacy metafile is a pair of CSV files, not a proper database.
        # first string is the path to the CSV file with tags, second is the blacklist
        raise NotImplementedError(
            "This class was deprecated in favor of SQL-based metafiles handling."
        )
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
        raise NotImplementedError(
            "This class was deprecated in favor of SQL-based metafiles handling."
        )
        logger.debug("Loading legacy taglist from metafiles")
        legacy_taglist = []
        with open(self.legacy_metafiles_path[0], "r") as tagfile:
            csv_reader = reader(tagfile, delimiter=",")
            for row in csv_reader:
                legacy_taglist.append(
                    [row[0], ";".join(row[1:])]
                )  # If parasitic "," separator after second column, just concatenate to the rest
        with open(self.legacy_metafiles_path[1], "r") as blacklistfile:
            for line in blacklistfile:
                legacy_taglist.append([line[:-1], BLACKLIST_TAG])
                # Remove newline character and add a specific tag for blacklisted items

        return np.array(legacy_taglist, dtype=str)

    def get_past_messages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy override for previous databases that did not include an empty_messages metadata flag.
        """
        raise NotImplementedError(
            "This class was deprecated in favor of SQL-based metafiles handling."
        )
        client = chromadb.PersistentClient(path=str(self.past_db_path))
        collection = client.get_collection("messages")  # TODO : do not hardcode, inelegant,
        # although legacy database has a consistent naming scheme.
        logger.debug("Fetching past messages from the legacy database")
        past_messages_get = collection.get(include=["documents"])  # type: ignore
        past_messages_ids = np.array(past_messages_get["ids"], dtype=str)
        past_messages_docs = np.array(past_messages_get["documents"], dtype=str)
        past_messages = np.column_stack((past_messages_docs, past_messages_ids))
        logger.debug(f"Fetched {len(past_messages)} past messages from the legacy database")

        past_empty_messages = np.array([], dtype=str)  # Legacy databases do not have empty messages

        return past_messages, past_empty_messages

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
        raise NotImplementedError(
            "This class was deprecated in favor of SQL-based metafiles handling."
        )
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

            message_to_tags[message] = list(set(message_to_tags[message]))  # Remove duplicates

        logger.debug(f"Linked tags to {len(message_to_tags)} unique text messages")
        return message_to_tags


class MetafileWriter:
    """
    Class to handle generating and writing the new metafile contents.

    This class is responsible for creating the metafile content based on the linked tags
    and writing it to the proper database.
    """

    def __init__(self, creation_mode: bool = False) -> None:
        """
        Initialize the MetafileWriter.

        :param creation_mode: If True, expects to work on a new database.
                              If False, expects to append to an existing database.
        """
        self.creation_mode = creation_mode
        self.dynamic_tags_db = get_paths().dynamic_tags_db

        if self.dynamic_tags_db.exists() == self.creation_mode:
            logger.debug("Creation mode mismatch with dynamic tags database existence.")
            # If creation mode is True, the database should not exist.
            # If creation mode is False, the database should exist.
            if self.creation_mode:
                logger.error(
                    f"Dynamic tags database {self.dynamic_tags_db} already exists, but creation mode is set to True."  # noqa: E501
                )
                raise FileExistsError(
                    f"Dynamic tags database at {self.dynamic_tags_db} already exists."
                )
            else:
                logger.error(
                    f"Dynamic tags database {self.dynamic_tags_db} does not exist, but creation mode is set to False."  # noqa: E501
                )
                raise FileNotFoundError(
                    f"Dynamic tags database at {self.dynamic_tags_db} does not exist."
                )
        else:
            self.sqlite_connection = sqlite3.connect(self.dynamic_tags_db)
            self.sqlite_cursor = self.sqlite_connection.cursor()
            logger.debug(f"Connected writer to dynamic tags database at {self.dynamic_tags_db}")

    def create_dynamic_tags_table(self) -> None:
        """
        Create the dynamic tags table in the database.

        This method creates a table with the following structure:
        - id: `INTEGER` `PRIMARY KEY` `AUTOINCREMENT` (automatic, not accessed, is part of the virtual table structure)
        - message_id: `TEXT`
        - tags: `TEXT`
        """  # noqa: E501
        assert self.creation_mode, "Cannot create dynamic tags table in non-creation mode."
        logger.info("Creating dynamic tags table in the database")
        # Create the dynamic_tags table with FTS5 for full-text search capabilities
        self.sqlite_cursor.execute(
            """CREATE VIRTUAL TABLE dynamic_tags USING fts5(message_id, tags);"""
        )
        self.sqlite_connection.commit()
        logger.info("Dynamic tags table created successfully")

    def write_single_tagline(
        self,
        message_id: str,
        tags: List[str],
        check_for_duplicates: bool = False,
    ) -> None:
        """
        Write a single tagline to the dynamic tags database.

        :param message_id: The ID of the message.
        :param tags: A list of tags associated with the message.
        """
        if check_for_duplicates:
            # Check if the message ID already exists in the database
            existing_tags = self.sqlite_cursor.execute(
                "SELECT * FROM dynamic_tags WHERE message_id = ?;", (message_id,)
            ).fetchone()
            if existing_tags:
                logger.error(
                    f"Message ID {message_id} already exists in the dynamic tags database, yet writing operation was attempted."  # noqa: E501
                )
                raise RuntimeError(
                    f"Attempted to write over {message_id}, but it already exists in the dynamic tags database."  # noqa: E501
                )

        # Join tags into a single string
        tags.sort()  # Sort tags for consistency
        # TODO : document somewhere that tags are sorted in alphabetical order
        # before being written to database
        tags_str = ";".join(tags)
        self.sqlite_cursor.execute(
            "INSERT INTO dynamic_tags (message_id, tags) VALUES (?, ?);",
            (message_id, tags_str),
        )
        self.sqlite_connection.commit()
        logger.debug(f"Tags for message ID {message_id} written successfully")

    def write_tags_dict(self, current_to_tags: Dict[str, List[str]]) -> None:
        """
        Write a dictionary of message IDs to tags into the dynamic tags database.

        :param current_to_tags: A dictionary mapping message IDs to lists of tags.
        """
        if not self.creation_mode:
            logger.warning(
                "In current version, mass writing is assumed to be creation mode exclusive. Are you sure of what you are doing?"  # noqa: E501
            )
            # raise RuntimeError(
            #     "Mass writing to the dynamic tags database is only allowed in creation mode."
            # )
            # TODO : lighten the restriction on mass writing to the dynamic tags database,
            # it is useful in other cases than creation mode.

        logger.info("Writing tags to the dynamic tags database")

        # Prepare data for executemany: sort tags and join them into a string
        tag_rows = [
            (message_id, ";".join(sorted(tags))) for message_id, tags in current_to_tags.items()
        ]
        # Insert all tags into the dynamic_tags table
        self.sqlite_cursor.executemany(
            "INSERT INTO dynamic_tags (message_id, tags) VALUES (?, ?);",
            tag_rows,
        )
        logger.debug(f"Inserted {len(tag_rows)} tag rows into the dynamic tags database")
        # Commit the changes to the database
        self.sqlite_connection.commit()
        logger.info("All tags written to the dynamic tags database successfully")

    def close(self) -> None:
        """
        Close the connection to the dynamic tags database.

        This method should be called when the writer is no longer needed.
        """
        if self.sqlite_connection:
            self.sqlite_connection.close()
            logger.trace("Dynamic tags database connection closed")
        else:
            logger.warning("Dynamic tags database connection was already closed")

    # def __del__(self):
    #     """
    #     Destructor to ensure the database connection is closed when the object is deleted.
    #     """
    #     # TODO : prefer deterministic closing via explicit closer method calls

    #     self.close()
    #     logger.debug(
    #         f"MetafileWriter instance {self.__repr__()} deleted, database connection closed"
    #     )

    def deduplicate_table(self) -> None:
        """
        Deduplicate the dynamic tags table by removing duplicate entries.

        This method removes duplicate entries from the dynamic_tags table based on the message_id.
        It keeps the first occurrence of each message_id and removes subsequent duplicates.
        """
        # TODO : understand why this method exectued so slowly, and optimize it.
        logger.info("Deduplicating the dynamic tags table")
        self.sqlite_cursor.execute(
            """
            DELETE FROM dynamic_tags
            WHERE rowid NOT IN (
                SELECT rowid
                FROM dynamic_tags dt
                WHERE rowid = (
                    SELECT rowid
                    FROM dynamic_tags
                    WHERE message_id = dt.message_id
                    ORDER BY LENGTH(tags) DESC, rowid ASC
                    LIMIT 1
                )
            );
            """
        )
        self.sqlite_connection.commit()
        logger.info("Dynamic tags table deduplicated successfully")


class MetafileQuerier:
    # TODO : docstrings.
    # TODO : more complete implementation of the querier.

    def __init__(self) -> None:
        self.dynamic_tags_db = get_paths().dynamic_tags_db

        assert (
            self.dynamic_tags_db.exists()
        ), "Dynamic tags database do not exist. Check configuration."

        self.sqlite_connection = sqlite3.connect(self.dynamic_tags_db)
        self.sqlite_cursor = self.sqlite_connection.cursor()
        logger.debug(f"Connected querier to dynamic tags database at {self.dynamic_tags_db}")

    def get_all_tagged_ids(self) -> List[str]:
        tagged_ids = [
            row[0]
            for row in self.sqlite_cursor.execute(
                """SELECT message_id FROM dynamic_tags"""
            ).fetchall()
        ]
        return tagged_ids
    
    def match_ids(self, ids: List[str]) -> List[str]:
        """
        Given a list of message IDs, return the subset that exists in the dynamic tags database.

        :param ids: A list of message IDs to check.
        :return: A list of message IDs that exist in the dynamic tags database.
        """
        self.sqlite_cursor.execute("DROP TABLE IF EXISTS _temp_ids;")
        self.sqlite_cursor.execute(
            "CREATE TEMPORARY TABLE _temp_ids (candidate_id TEXT PRIMARY KEY);"
        )

        self.sqlite_cursor.executemany(
            "INSERT OR IGNORE INTO _temp_ids (candidate_id) VALUES (?);",
            [(message_id,) for message_id in ids],
        )

        matched_ids = [
            row[0]
            for row in self.sqlite_cursor.execute(
                """
                SELECT ti.candidate_id
                FROM _temp_ids ti
                INNER JOIN dynamic_tags dt ON ti.candidate_id = dt.message_id;
                """
            ).fetchall()
        ]

        self.sqlite_cursor.execute("DROP TABLE IF EXISTS _temp_ids;")
        return matched_ids

    def close(self) -> None:
        """
        Close the connection to the dynamic tags database.

        This method should be called when the querier is no longer needed.
        """
        if self.sqlite_connection:
            self.sqlite_connection.close()
            logger.trace("Dynamic tags database connection closed")
        else:
            logger.warning("Dynamic tags database connection was already closed")

    # def __del__(self):
    #     """
    #     Destructor to ensure the database connection is closed when the object is deleted.
    #     """
    #     # TODO : prefer deterministic closing via explicit closer method calls

    #     self.close()
    #     logger.debug(
    #         f"MetafileQuerier instance {self.__repr__()} deleted, database connection closed"
    #     )


if __name__ == "__main__":
    from backend.utils.log_setup import LoggerSetup
    from backend.utils.path_utils import get_paths

    LoggerSetup.configure_logger(console_level="DEBUG")

    logger.info("Starting Metafiles process...")
    logger.warning("""You are running the metafiles handling script directly.
                   This is intended for debugging purposes only.

                   The script IS SUSCPTIBLE to write, read, and modify actual databases.
                   It is not recommended to run this script in production environments.

                   This script is not intended for production use.
                   Users stay advised.""")

    # Debug run
    # TODO : at some point, ask for user input to proceed OR remove the debug run script.

    # # TEST THE LEGACY LINKER'S BEHAVIOR
    # from pprint import pprint

    # # Get paths from the environment variables
    # paths = get_paths()

    # legacy_chroma = paths.legacy_chroma_dir / "V1 postprocess_chroma"
    # legacy_metafiles = (paths.messages_categories, paths.blacklist_categories)

    # legacy_linker = LegacyLinker(
    #     legacy_db_path=legacy_chroma,
    #     legacy_metafiles_path=legacy_metafiles,
    # )

    # logger.info("Running the legacy linker pipeline")
    # current_to_tags = legacy_linker.pipeline()
    # logger.info("Legacy linker pipeline completed")
    # logger.info("Current to tags mapping:")
    # with open("data/text_output_streams/current_to_tags_output.txt", "w") as f:
    #     pprint(current_to_tags, stream=f)
    # logger.info("Finished running the legacy linker")

    # # TEST THE METAFILE WRITER'S BEHAVIOR
    # legacy_linker = LegacyLinker(
    #     legacy_db_path=legacy_chroma,
    #     legacy_metafiles_path=legacy_metafiles,
    # )

    # # Get paths from the environment variables
    # paths = get_paths()
    # legacy_chroma = paths.legacy_chroma_dir / "V1 postprocess_chroma"
    # legacy_metafiles = (paths.messages_categories, paths.blacklist_categories)

    # logger.info("Running the legacy linker pipeline")
    # current_to_tags = legacy_linker.pipeline()
    # logger.info("Legacy linker pipeline completed")

    # metafile_writer = MetafileWriter(creation_mode=True)
    # logger.info("Creating dynamic tags table")
    # metafile_writer.create_dynamic_tags_table()
    # logger.info("Writing tags to the dynamic tags database")
    # metafile_writer.write_tags_dict(current_to_tags)
    # logger.info("Finished writing tags to the dynamic tags database")
    # logger.info("Metafiles process completed successfully.")
    # print(metafile_writer.sqlite_cursor.execute("SELECT * FROM dynamic_tags;").fetchmany(10))
    # metafile_writer = None
    # # Explicitly delete the metafile writer instance to trigger the GC and destructor
    # # Check if the destructor is called and the database connection is closed
    # logger.info("MetafileWriter instance deleted, database connection should be closed now.")
    # logger.info("Exiting the script.")

    # # JUST PEEK INTO THE DYNAMIC TAGS DATABASE
    # from pprint import pprint

    # metafile_writer = MetafileWriter(creation_mode=False)
    # logger.info("Peeking into the dynamic tags database")
    # dynamic_tags = metafile_writer.sqlite_cursor.execute(
    #     """SELECT *
    #     FROM dynamic_tags
    #     WHERE tags NOT LIKE 'none';"""
    # ).fetchall()
    # # # Query for tags that start with '##' and end with '##' (whatever in the middle)
    # # dynamic_tags = metafile_writer.sqlite_cursor.execute(
    # #     """
    # #     SELECT *
    # #     FROM dynamic_tags
    # #     WHERE tags GLOB '##*##';
    # #     """
    # # ).fetchall()
    # logger.info("Dynamic tags database content:")
    # with open("data/text_output_streams/dynamic_tags_output.txt", "w") as f:
    #     pprint(dynamic_tags, stream=f)
    # metafile_writer = None
    # logger.info("MetafileWriter set to None.")
    # logger.info("Exiting the script.")

    # # DEDUPLICATE THE DYNAMIC TAGS TABLE
    # metafile_writer = MetafileWriter()
    # logger.info("Deduplicating the dynamic tags table")
    # metafile_writer.deduplicate_table()
    # logger.info("Dynamic tags table deduplicated successfully")
    # del metafile_writer
    # logger.info("MetafileWriter instance deleted, database connection should be closed now.")
    # logger.info("Exiting the script.")

    # COUNT ENTRIES IN THE DYNAMIC TAGS TABLE
    metafile_querier = MetafileQuerier()
    logger.info("Counting entries in the dynamic tags table")
    count = metafile_querier.sqlite_cursor.execute(
        """SELECT COUNT(*) FROM dynamic_tags;"""
    ).fetchone()[0]
    logger.info(f"Dynamic tags table contains {count} entries")
    metafile_querier = None
