"""Wrappers for the ChromaDB access, as endpoints with safe methods"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional, Set, Tuple, Union

import chromadb
import chromadb.api
import chromadb.api.types
import numpy as np
import requests
from loguru import logger

from backend.loaders.conversation_loader import ConversationLoader
from backend.loaders.conversation_parser import ConversationParser, ParsedMessage
from backend.models.app_models import CommandValue, JobStatus, UpdaterJobInfo
from backend.utils import hash_utils as hash_utils
from backend.utils.path_utils import get_paths

Metadata = Mapping[
    str, Union[str, int, float, bool]
]  # Mimic chromadb.Metadata type for easier type hinting

ChromaInclude = chromadb.api.types.IncludeEnum


class UpsertMessageArgs(NamedTuple):
    """A named tuple for upserting messages into ChromaDB."""

    ids: str
    documents: str
    metadatas: Metadata


class UpsertMultipleMessagesArgs(NamedTuple):
    """A named tuple for upserting multiple messages into ChromaDB."""

    ids: List[str]
    documents: List[str]
    metadatas: List[Metadata]


class UpsertableTitle:
    """A class representing a conversation that can be naturally upserted
    into the ChromaDB `conversations` collection."""

    def __init__(
        self,
        title: str,
    ):
        """
        Initialize the UpsertableTitle with an ID and title.
        :param conversation_id: The unique identifier for the conversation
        :param title: The title of the conversation
        """
        self.title = title
        self.conversation_id = hash_utils.hash_set_length(title.encode("utf-8"), length=16)

    def __repr__(self) -> str:
        return f"UpsertableTitle(conversation_id={self.conversation_id}, title={self.title})"

    def cast(self) -> Dict[str, List[str]]:
        """
        Cast the UpsertableTitle to a proper argument for upserting into ChromaDB.
        The output is an unpackable object `data` such that
        it can be used directly in the `upsert` method :
        `conversation_collection.upsert(**data)`,
        """
        return {
            "ids": [self.conversation_id],
            "documents": [self.title],
        }


class UpsertableMessages:
    """A class representing messages that can be naturally upserted
    into the ChromaDB `messages` collection."""

    def __init__(
        self,
        parser: ConversationParser,
    ):
        self.title = parser.title
        self.messages = parser.messages
        self.conversation_id = hash_utils.hash_set_length(self.title.encode("utf-8"), length=16)

    def __repr__(self) -> str:
        return f"UpsertableMessages(title={self.title}, messages={self.messages})"

    def cast_single_message(
        self,
        message: ParsedMessage,
    ) -> UpsertMessageArgs:
        """
        Cast a single message to a proper argument for upserting into ChromaDB.
        :param message: The ParsedMessage instance to cast
        :return: A dictionary containing the message ID, content, and metadata
        """
        # TODO : make the metadata a NamedTuple, or a TypedDict,
        # essentially anything that requires every key to be present or something.
        # For consistency, you know.
        message_id = hash_utils.hash_message(str(message))
        logger.trace(f"Casting message: {message_id}")

        compatible_metadata = message.metadata.to_dict()
        compatible_metadata["conv_id"] = self.conversation_id
        compatible_metadata["empty_or_non_text"] = (
            message.content == "" or message.content == "[non-text content]"
        )  # This is a boolean flag to indicate if the message is empty or non-text content.
        # It's a workaround for the fact that ChromaDB does not support directly
        # filtering out documents based on string equality.
        # It also seems like document-based filtering is computationally expensive.

        return UpsertMessageArgs(
            ids=message_id,
            documents=message.content,
            metadatas=compatible_metadata,
        )

    def cast(self) -> UpsertMultipleMessagesArgs:
        """
        Cast the UpsertableMessages to a proper argument for upserting into ChromaDB.
        The output is an unpackable object `data` such that
        it can be used directly in the `upsert` method :
        `messages_collection.upsert(**data)`,
        """
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Metadata] = []

        for message in self.messages:
            casted_message = self.cast_single_message(message)
            ids.append(casted_message.ids)
            documents.append(casted_message.documents)
            metadatas.append(casted_message.metadatas)

        return UpsertMultipleMessagesArgs(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )


class UpsertableConversation:
    """A class representing a conversation that can be naturally upserted
    into the ChromaDB respective collections."""

    def __init__(
        self,
        conversation: ConversationParser,
    ):
        """
        Initialize the UpsertableConversation with a parsed conversation.
        :param conversation: The ConversationParser instance containing the parsed conversation
        """
        self.conversation = conversation
        self.title = UpsertableTitle(conversation.title)
        self.messages = UpsertableMessages(conversation)

    def cast(self) -> Tuple[Dict[str, List[str]], UpsertMultipleMessagesArgs]:
        """
        Cast the UpsertableConversation to a proper argument for upserting into ChromaDB.
        :return: A tuple containing the conversation data and messages data
        """
        logger.trace(f"Casting UpsertableConversation: {self.title.title}")
        return self.title.cast(), self.messages.cast()


@dataclass(frozen=True)
class CollectionsNames:
    """
    A dataclass to hold the names of the ChromaDB collections.

    We do NOT want any tinkering with the collection names.
    """

    # TODO : store default collection names in a config file, rather than hardcoding them here.

    conversations: str = "conversations"
    messages: str = "messages"

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the CollectionsNames instance to a dictionary.

        :return: A dictionary representation of the collection names
        """
        return {
            "conversations": self.conversations,
            "messages": self.messages,
        }

    def missing_keys(self, provided: Dict[str, str]) -> Set[str]:
        """
        Check which keys are missing from the provided dictionary
        compared to the default collection names.

        :param provided: The dictionary of provided collection names
        :return: A list of missing keys
        """
        return set(self.to_dict().keys()).difference(provided.keys())


COLLECTIONS_NAMES = CollectionsNames()


def connect(creator_mode: bool = False) -> chromadb.api.ClientAPI:
    """
    Connect to the ChromaDB persistent client.
    This function retrieves the database path and initializes the client.

    :rtype: chromadb.api.ClientAPI
    """
    DB_PATH = get_paths().chroma_db_path
    exists = DB_PATH.exists()

    if not exists and not creator_mode:
        # We are trying to connect as readers to a database that does not exist.
        # Log critical (for VERY unexpected behavior, possibly corrupted storage),
        # then raise an error (for error at runtime or invalid config).
        logger.critical(
            f"ChromaDB under path {DB_PATH} does not exist. Recommended checking env variables or creating a new database first."  # noqa: E501
        )
        raise FileNotFoundError(
            f"ChromaDB path {DB_PATH} does not exist. Please check the configuration."
        )
    elif not exists and creator_mode:
        # We are trying to connect as creators to a database that does not exist.
        # This is expected behavior, so we log info, and warn about the `parents=True` flag.
        logger.info(f"ChromaDB under path {DB_PATH} does not exist. Creating a new database.")
        logger.warning(
            "User be advised, database creation includes creating missing path parent directories."
        )
        DB_PATH.mkdir(
            parents=True, exist_ok=False
        )  # Create the directory. It will raise an error if it already exists,
        #    but it shouldn't already exist if we're here.
    elif exists and creator_mode:
        # We are trying to connect as creators to a database that already exists.
        # This is unexpected behavior, so we log and raise an error.
        logger.critical(
            f"ChromaDB under path {DB_PATH} already exists. Recommended checking env variables or using the upsert script."  # noqa: E501
        )
        raise FileExistsError(
            f"ChromaDB path {DB_PATH} already exists. Please check the configuration."
        )
    else:
        # We are trying to connect as readers to a database that already exists.
        # This is expected behavior, so we log info.
        logger.info(f"Connecting to existing ChromaDB at {DB_PATH}.")

    # Initialize the ChromaDB persistent client
    # After previous checks, the database path should exist.
    logger.debug(f"Initializing ChromaDB client with path: {DB_PATH}")
    return chromadb.PersistentClient(path=str(DB_PATH))


class ChromaUpserter:
    """A wrapper class to properly upsert conversations into ChromaDB."""

    # TODO : find a way for the script to flag or skip conversations
    # that are already in the database in entirety

    def __init__(
        self, client: Optional[chromadb.api.ClientAPI] = None, api_endpoint: Optional[str] = None
    ) -> None:
        """
        Initialize the ChromaUpserter with the ChromaDB client and collections.

        The database path is retrieved, checked for existence,
        and the client instantiated AT RUNTIME.

        If the database path does not exist,
        a critical error is logged and a FileNotFoundError is raised.
        """
        if client is None:
            client = connect()

        self.client = client

        self.conv_collection_name = COLLECTIONS_NAMES.conversations
        self.mess_collection_name = COLLECTIONS_NAMES.messages

        self.conv_collection = self.client.get_collection(self.conv_collection_name)
        self.mess_collection = self.client.get_collection(self.mess_collection_name)

        self.api_endpoint = api_endpoint

        self.conversation_loader = ConversationLoader()

    def post_status_update(
        self,
        status: JobStatus,
        updated_conversations: List[str],
        post_command: CommandValue = CommandValue.DEFAULT,
    ) -> CommandValue:
        logger.trace("Posting status update to the API endpoint.")

        if self.api_endpoint is not None:
            # If an API endpoint is provided, send a POST request to update the job status
            updater = UpdaterJobInfo(
                status=status,
                updated_conversations=updated_conversations,
            )
            try:
                logger.trace(
                    f"Sending POST request to {self.api_endpoint} with data: {updater.model_dump()}"
                )  # noqa: E501
                response = requests.post(
                    self.api_endpoint,
                    json=updater.model_dump(),
                )
                response.raise_for_status()  # Raise an error for bad responses

                logger.debug(
                    f"Successfully updated job status at {self.api_endpoint}."  # noqa: E501
                )
            except requests.RequestException as e:
                logger.error(
                    f"Failed to update job status at {self.api_endpoint}: {e}"  # noqa: E501
                )
                raise requests.RequestException(
                    f"Failed to update job status at {self.api_endpoint}: {e}"  # noqa: E501
                ) from e

            command: str = response.json()["command"]
            logger.debug(f"Received command from API: {command}")
            return CommandValue(command)
        else:
            # If no API endpoint is provided, do nothing and return the default command value
            logger.trace("No API endpoint provided, skipping status update.")
            return CommandValue.DEFAULT

    def upsert_conversation(self, conversation: ConversationParser) -> CommandValue:
        """
        Upsert a single conversation into the ChromaDB collections.

        :param conversation: The ConversationParser instance containing the parsed conversation
        """
        # TODO : check if the upsert updates the conversation.
        # If it doesn't, we've likely reached a point of the loop where
        # the conversations are already in the database.
        logger.debug(f"Upserting conversation: {conversation.title}")

        upsertable_conv = UpsertableConversation(conversation)
        conv_data, mess_data = upsertable_conv.cast()

        logger.debug(f"Upserting conversation: {upsertable_conv.title.title}")
        self.conv_collection.upsert(ids=conv_data["ids"], documents=conv_data["documents"])

        logger.debug(f"Upserting messages for conversation: {upsertable_conv.title.title}")
        self.mess_collection.upsert(
            ids=mess_data.ids,
            documents=mess_data.documents,
            metadatas=mess_data.metadatas,
        )

        logger.success(f"Conversation '{upsertable_conv.title.title}' upserted successfully.")

        command = self.post_status_update(
            status=JobStatus.RUNNING,
            updated_conversations=[upsertable_conv.title.title],
        )
        logger.debug(
            f"Posted status update for conversation '{upsertable_conv.title.title}' to {self.api_endpoint}."  # noqa: E501
        )

        return command

    def upsert_all_conversations(self) -> None:
        """
        Upsert all conversations from the ConversationLoader into the ChromaDB collections.
        This method iterates through all conversations and upserts them one by one.
        """
        logger.info("Starting upsert of all conversations.")
        self.post_status_update(
            status=JobStatus.RUNNING,
            updated_conversations=[],
        )

        processed_titles: List[str] = []
        for conversation in self.conversation_loader:
            logger.debug(f"Processing conversation: {conversation.title}")
            command = self.upsert_conversation(conversation)
            processed_titles.append(conversation.title)
            if command == CommandValue.ABORT:
                logger.warning("Aborting upsert process as per command from API.")
                self.post_status_update(
                    status=JobStatus.ABORTED,
                    updated_conversations=processed_titles,
                    post_command=CommandValue.ABORT,
                )
                return

        logger.success("All conversations have been upserted successfully.")
        self.post_status_update(
            status=JobStatus.COMPLETED,
            updated_conversations=[conv.title for conv in self.conversation_loader],
        )


class ChromaQuerier:
    def __init__(self, client: Optional[chromadb.api.ClientAPI] = None) -> None:
        """
        Initialize the ChromaQuerier with the ChromaDB client and collections.

        The database path is retrieved, checked for existence,
        and the client instantiated AT RUNTIME.
        """
        if client is None:
            client = connect()

        self.client = client

        self.conv_collection_name = COLLECTIONS_NAMES.conversations
        self.mess_collection_name = COLLECTIONS_NAMES.messages

        self.conv_collection = self.client.get_collection(self.conv_collection_name)
        self.mess_collection = self.client.get_collection(self.mess_collection_name)

    def _fully_query(
        self,
        query_text: chromadb.Documents,
        where_condition: Optional[chromadb.Where] = None,
        where_document_condition: Optional[chromadb.WhereDocument] = None,
        include: Optional[chromadb.Include] = None,
        n_results: int = 100,
    ) -> chromadb.QueryResult:
        """
        A flexible wrapper for querying the ChromaDB messages collection.

        :param query_text: The text to query against the collection
        :param where_condition: Optional condition to filter results
        :param where_document_condition: Optional condition to filter documents
        :param include: Optional include parameters for the query
        :param n_results: Optional number of results to return
        :return: The query result from the ChromaDB collection
        """
        logger.debug(f"Full querying for text: {query_text}")

        if include is None:
            res = self.mess_collection.query(
                query_texts=query_text,
                where=where_condition,
                where_document=where_document_condition,
                n_results=n_results,
            )
        else:
            res = self.mess_collection.query(
                query_texts=query_text,
                where=where_condition,
                where_document=where_document_condition,
                include=include,
                n_results=n_results,
            )

        logger.debug("Query completed successfully.")

        return res

    def quick_query(
        self,
        query_text: str | List[str],
        n_results: int = 10,
    ) -> List[str]:
        """
        A quick query method that returns the most relevant messages
        from the ChromaDB messages collection in order.

        Aims to query for NON-EMPTY messages only (not quite implemented yet).
        Also queries for documents that are not `[non-text content]` (not implemented either).

        :param query_text: The text to query against the collection
        :param n_results: The number of results to return (default is 10)
        :return: The most relevant message from the collection
        """
        if isinstance(query_text, str):
            query_text = [query_text]

        logger.debug(f"Quick querying for text: {query_text}")
        query_res = self._fully_query(
            query_text=query_text,
            n_results=n_results,
            where_condition={"empty_or_non_text": False},  # Filter out empty or non-text messages
        )

        results = []
        if query_res["documents"] is None:
            logger.error("No documents found for the given query.")
            raise ValueError("No documents found for the given query.")

        results.extend(query_res["documents"][0])
        # It returns a list of lists, but since we query only one text,
        # we take the first (and only) list.

        return results

    def get_all_nonempty_messages(self) -> np.ndarray:
        """
        Retrieve all messages from the ChromaDB messages collection.

        This method filters out empty or non-text messages by checking the `empty_or_non_text`
        metadata field.
        It returns a 2D numpy array where the first column contains message IDs
        and the second column contains the corresponding message content.

        :return: A 2D numpy array with message IDs and content.

                 - On the first coordinate, the message ID.
                 - On the second coordinate, the message content.

        :rtype: np.ndarray
        :raises ValueError: If no documents are retrieved or if there is a mismatch
            between the number of IDs and documents retrieved from the messages collection.
        """
        logger.debug("Retrieving all messages from the ChromaDB messages collection.")
        query_res = self.mess_collection.get(
            where={"empty_or_non_text": False},
            include=[ChromaInclude.documents],
        )

        ids_list = query_res["ids"]
        documents_list = query_res["documents"]
        if documents_list is None:
            logger.critical("No documents retrieved from messages collection.")
            raise ValueError("No documents retrieved from messages collection.")
        if len(ids_list) != len(documents_list):
            logger.critical(
                "Mismatch between number of IDs and documents retrieved from messages collection."
            )
            raise ValueError(
                "Mismatch between number of IDs and documents retrieved from messages collection."
            )

        logger.debug(f"Retrieved {len(ids_list)} messages from the collection.")

        # Create a 2D numpy array with IDs and documents
        ids_arr = np.array(ids_list, dtype=str)
        documents_arr = np.array(documents_list, dtype=str)
        messages_array = np.column_stack((ids_arr, documents_arr))

        return messages_array

    def get_all_empty_messages(self) -> np.ndarray:
        """
        Retrieve all empty messages from the ChromaDB messages collection.

        This method filters messages that are marked as empty or non-text content
        by checking the `empty_or_non_text` metadata field.
        It returns a 1D numpy array containing the IDs of the empty messages.

        :return: A 1D numpy array with message IDs of empty messages.
        :rtype: np.ndarray
        """
        logger.debug("Retrieving all empty messages from the ChromaDB messages collection.")
        query_res = self.mess_collection.get(
            where={"empty_or_non_text": True},
            include=[],
        )

        ids_list = query_res["ids"]
        logger.debug(f"Retrieved {len(ids_list)} empty messages from the collection.")

        return np.array(ids_list, dtype=str)

    def count_collections(self) -> Dict[str, int]:
        """
        Count the number of documents in each collection.

        :return: A dictionary with collection names as keys and their document counts as values.
        :rtype: Dict[str, int]
        """
        logger.debug("Counting documents in each ChromaDB collection.")
        counts = {
            self.conv_collection_name: self.conv_collection.count(),
            self.mess_collection_name: self.mess_collection.count(),
        }
        logger.debug(f"Document counts: {counts}")
        return counts


class ChromaCreator:
    """
    ChromaCreator is a utility class for initializing and managing ChromaDB collections with configurable embedding functions, collection names, and database paths.
    This class facilitates the creation of ChromaDB collections by allowing the user to specify custom embedding functions (either a single function or a tuple of two), override default collection names, and set a custom database path. It ensures that all required collection names are provided and that embedding functions are correctly configured. The class provides methods to create the ChromaDB client and its collections, handling all necessary setup and validation.
    Attributes:
        collection_names (Dict[str, str]): Mapping of collection keys to their names.
        db_path (Path): Path to the ChromaDB database.
        embedding_functions (Tuple[chromadb.EmbeddingFunction, chromadb.EmbeddingFunction]): Tuple containing embedding functions for the collections.
    Methods:
        __init__(...): Initializes the ChromaCreator with embedding functions, collection names, and database path.
        _create_collections(client): Creates the required ChromaDB collections using the provided client.
        create(): Instantiates the ChromaDB client and creates the collections.
    """  # noqa: E501

    def __init__(
        self,
        embedding_function: chromadb.EmbeddingFunction
        | Tuple[chromadb.EmbeddingFunction, chromadb.EmbeddingFunction]
        | None = None,
        nondefault_collection_names: Dict[str, str] | None = None,
        nondefault_db_path: str | Path | None = None,
    ) -> None:
        """
        Initializes the class with embedding functions, collection names, and database path.
        Args:
            embedding_function (chromadb.EmbeddingFunction | Tuple[chromadb.EmbeddingFunction, chromadb.EmbeddingFunction] | None, optional):
                The embedding function(s) to use. Can be a single embedding function, a tuple of two embedding functions, or None.
                If a single function is provided, it will be used for both roles. If None, embedding functions must be set later.
            nondefault_collection_names (Dict[str, str] | None, optional):
                A dictionary mapping required collection names. If None, defaults are loaded from COLLECTIONS_NAMES.
                Raises ValueError if required keys are missing.
            nondefault_db_path (str | Path | None, optional):
                The path to the database. If None, uses the default path from get_paths().chroma_db_path.
                If a string is provided, it is converted to a Path object.
        Raises:
            ValueError: If nondefault_collection_names does not contain all required keys,
                or if embedding_function is a tuple but does not contain exactly two functions.
        """  # noqa: E501

        # Handle collection names
        if nondefault_collection_names is None:
            nondefault_collection_names = COLLECTIONS_NAMES.to_dict()
        elif isinstance(nondefault_collection_names, dict) and not set(
            COLLECTIONS_NAMES.to_dict().keys()
        ).issubset(nondefault_collection_names.keys()):
            logger.error(
                f"Provided nondefault_collection_names does not contain required keys: {COLLECTIONS_NAMES.missing_keys(nondefault_collection_names)}"  # noqa: E501
            )
            raise ValueError(
                f"Provided nondefault_collection_names does not contain required keys: {COLLECTIONS_NAMES.missing_keys(nondefault_collection_names)}"  # noqa: E501
            )
        self.collection_names = nondefault_collection_names

        # Handle database path
        if nondefault_db_path is None:
            nondefault_db_path = get_paths().chroma_db_path
        elif isinstance(nondefault_db_path, str):
            nondefault_db_path = Path(nondefault_db_path)
        self.db_path = nondefault_db_path

        # Handle embedding functions
        if not isinstance(embedding_function, tuple):
            self.embedding_functions = (embedding_function, embedding_function)
        elif len(embedding_function) != 2:
            logger.error(
                "Provided embedding_function must be a single function or a tuple of two functions."
            )
            raise ValueError(
                "Provided embedding_function must be a single function or a tuple of two functions."
            )
        else:
            self.embedding_functions = embedding_function

    def _create_collections(self, client: chromadb.api.ClientAPI) -> None:
        """
        Create the ChromaDB collections with the specified names.
        :param client: The ChromaDB client to use for creating collections
        """
        logger.info("Creating ChromaDB collections...")
        for i, key in enumerate(COLLECTIONS_NAMES.to_dict().keys()):
            collection_to_create = self.collection_names[key]
            client.create_collection(
                name=collection_to_create,
                embedding_function=self.embedding_functions[i],
            )

    def create(self) -> chromadb.api.ClientAPI:
        """
        Create the ChromaDB client and collections.
        :return: The ChromaDB client with the created collections
        """
        logger.info(f"Connecting to new ChromaDB at {self.db_path}...")
        client = connect(creator_mode=True)

        # Create collections if they do not exist
        self._create_collections(client)

        logger.success("ChromaDB client and collections created successfully.")
        return client


if __name__ == "__main__":
    from backend.utils.log_setup import LoggerSetup

    LoggerSetup.configure_logger()

    ANSI_CYAN = "\033[96m"
    ANSI_RESET = "\033[0m"

    logger.info("Starting ChromaDB upsert process...")
    logger.warning("""You are running the ChromaDB upsert script directly.
                   This is intended for debugging purposes only.
                   This WILL also upsert all conversations.

                   This script is not intended for production use.
                   Users stay advised.""")

    # Debug run
    # TODO : at some point, ask for user input to proceed OR remove the debug run script.

    # # CHECK THE CURRENT COLLECTIONS' LENGTHS
    # querier = ChromaQuerier()
    # counts = querier.count_collections()
    # logger.info("Current collections' lengths:")
    # for collection_name, count in counts.items():
    #     logger.info(f"{collection_name}: {count} documents")

    # # Find and display the number of empty messages
    # empty_messages = querier.get_all_empty_messages()
    # logger.info(f"Number of empty messages: {len(empty_messages)}")

    # logger.info("You can now use the ChromaDB client to query or manipulate the data.")
    # logger.info("ChromaDB client is ready for use.")

    # # QUICKLY QUERY THE DATABASE FOR A GIVEN TEXT INPUT
    # querier = ChromaQuerier()
    # query_text = None

    # if query_text is None:
    #     query_text = str(
    #         input("Enter the text to query against the ChromaDB messages collection: ")
    #     )

    # results = querier.quick_query(
    #     query_text=query_text, n_results=int(input("Enter the number of results to return: "))
    # )

    # for i, result in enumerate(results):
    #     print(
    #         "{color}[{rank:>3}]{uncolor} {content}\n".format(
    #             color=ANSI_CYAN,
    #             rank=i + 1,
    #             uncolor=ANSI_RESET,
    #             content=result,
    #         )
    #     )
    # logger.success("ChromaDB query process completed successfully.")

    # # CREATE A NEW CHROMADB PERSISTENT DATABASE
    # creator = ChromaCreator()
    # client = creator.create()
    # upserter = ChromaUpserter(client=client)
    # upserter.upsert_all_conversations()
    # logger.info("ChromaDB upsert process completed successfully.")
    # logger.info("You can now use the ChromaDB client to query or manipulate the data.")
    # logger.info("ChromaDB client is ready for use.")
