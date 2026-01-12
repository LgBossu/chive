"""Wrappers for the ChromaDB access, as endpoints with safe methods"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

import chromadb
import chromadb.api
import chromadb.api.configuration
import chromadb.api.types

# numpy removed: avoid building large arrays in-memory from Chroma queries
import requests
from loguru import logger

from backend.loaders.conversation_loader import ConversationLoader
from backend.loaders.conversation_parser import ConversationParser
from backend.models.app_models import CommandValue, JobStatus, QueryDatabaseModel, UpdaterJobInfo
from backend.models.data_models import (
    ChromaDocument,
    ChromaEmbedding,
    ChromaInclude,
    Metadata,
    UpsertableConversation,
)
from backend.models.message_node import MessageNode
from backend.utils import hash_utils as hash_utils
from backend.utils.path_utils import get_paths


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

    def close(self) -> None:
        """
        Close the ChromaDB client connection.
        """
        logger.debug("Closing ChromaDB client connection.")
        del self.client 
        # ChromaDB PersistentClient (ClientAPI) does not have a close method. We dereference the API instead.
        # TODO : future versions might use other clients that expose explicit closing.
        # Make sure to be aware and update in consequence.

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
        query: QueryDatabaseModel,
    ) -> chromadb.QueryResult:
        """
        A flexible wrapper for querying the ChromaDB messages collection.

        :param query: The `QueryDatabaseModel` containing all query parameters
        :return: The query result from the ChromaDB collection
        """
        logger.trace(f"Full querying for text: {query.query_text}")

        query_dict = query.cast_to_query_args()
        try:
            res: chromadb.QueryResult = self.mess_collection.query(**query_dict)
        except RuntimeError as e:
            logger.error(f"Failed to query ChromaDB: {e}")
            logger.warning("Trying a query without metadata filtering.")
            query_dict["where"] = None  # Remove metadata filtering
            query_dict["where_document"] = None  # Remove document-specific filtering
            res: chromadb.QueryResult = self.mess_collection.query(**query_dict)

        logger.trace("Query completed successfully.")

        return res

    def _flatten_query_documents(
        self,
        query_res: chromadb.QueryResult,
    ) -> List[str]:
        """
        Parse the query result from the ChromaDB collection.

        :param query_res: The query result from the ChromaDB collection
        :param flatten: If True, flattens the result to a single list of document texts
        :return: A list of document texts from the query result
        :raises ValueError: If no documents are found in the query result
        """
        logger.debug("Parsing query result.")

        if query_res["documents"] is None:
            logger.error("No documents found in query result.")
            raise ValueError("No documents found in query result.")

        # Extract document texts from the query result
        res = []
        for doc in query_res["documents"]:
            res.extend(doc)

        return res

    # TODO : go over and fix error propagation from unpack_result :
    # certain methods expect a ValueError to be raised on empty results,
    # unpack_result raises a ValueError on missing fields,
    # but not on empty results.
    def _unpack_result(
        self,
        query_res: Union[chromadb.QueryResult, chromadb.api.types.GetResult],
        batch_index: int = 0,
    ) -> List[Tuple[str, ChromaDocument, ChromaEmbedding, Metadata]]:
        """
        Unpack the query result from the ChromaDB collection into a list of tuples.

        Each tuple contains:
        - message ID
        - document content
        - embeddings
        - metadata

        To account for the fact that the query result may contain multiple batches,
        the `batch_index` parameter is used to specify which batch to unpack.

        :param query_res: The query result from the ChromaDB collection
        :param batch_index: The index of the batch to unpack (default is 0)
        :return: A list of tuples containing the unpacked query result
        :raises ValueError: If the query result does not contain the expected fields
        """
        logger.trace("Unpacking query result.")

        if not query_res["ids"]:
            return []
        if not query_res["documents"] or not query_res["metadatas"] or not query_res["embeddings"]:
            raise ValueError("Failed to retrieve messages contents, metadata or embeddings.")
        try:
            if isinstance(query_res["ids"][0], str):
                # If the IDs are strings, we can directly zip them (GetResult case)
                iter_result = zip(
                    query_res["ids"],
                    query_res["documents"],
                    query_res["embeddings"],
                    query_res["metadatas"],
                )
            else:
                # If the IDs are lists (batches), we need to index into them
                # to get the specific batch we want (QueryResult case)
                if batch_index >= len(query_res["ids"]):
                    raise IndexError("Batch index out of range for query result.")
                iter_result = zip(
                    query_res["ids"][batch_index],
                    query_res["documents"][batch_index],
                    query_res["embeddings"][batch_index],
                    query_res["metadatas"][batch_index],
                )
        except KeyError as e:
            logger.error(f"Failed to parse query result: {e}")
            raise ValueError("Missing required fields in the query result.") from e
        except Exception as e:
            logger.error(f"Error processing query result: {e}")
            raise ValueError("Error processing query result.") from e

        return list(iter_result)  # type: ignore
        # This type ignore is exceptional, as compliance with the type checker
        # is difficult : GetResult and QueryResult are not types, but TypedDicts,
        # and the type checker does not understand that we are unpacking them correctly.
        # TODO : find a way to make this static type compliant.

    def _cast_query_res_to_message_nodes(
        self,
        query_res: Union[chromadb.QueryResult, chromadb.api.types.GetResult],
    ) -> List[MessageNode]:
        """
        Parse the query result from the ChromaDB collection into MessageNode objects.

        The MessageNodes are created "raw", without linking or processing.

        :param query_res: The query result from the ChromaDB collection
        :return: A list of MessageNode objects created from the query result
        :raises ValueError: If no documents are found in the query result
        """
        logger.debug("Casting query result to MessageNode objects.")

        iter_result = self._unpack_result(query_res)

        return [MessageNode(*item) for item in iter_result]

    def query_sorted_nodes(
        self,
        query: QueryDatabaseModel,
    ) -> Dict[str, List[MessageNode]]:
        """
        Query the ChromaDB messages collection and cast the query result to MessageNodes,
        to build the ambient message trees and highlight the queried messages.

        :param query: The QueryDatabaseModel containing all query parameters
        :return: A list of MessageTree objects created from the query result
        :raises ValueError: If no documents are found in the query result
        """
        logger.debug(f"Querying for trees with query: {query.query_text}")

        query_res = self._fully_query(query)

        try:
            message_nodes = self._cast_query_res_to_message_nodes(query_res)
        except Exception as e:
            logger.error(f"Error casting query result to MessageNodes: {e}")
            raise e

        # A dictionary to sort the messages by their conversation ID
        sorter: Dict[str, List[MessageNode]] = dict()

        for node in message_nodes:
            conv_id = node.metadata.get("conv_id")
            assert isinstance(conv_id, str), "Conversation ID must be set to a string."
            if conv_id not in sorter:
                sorter[conv_id] = []
            sorter[conv_id].append(node)

        return sorter

    def quick_query(
        self,
        query_text: str,
        n_results: int = 25,
    ) -> List[str]:
        """
        A quick query method that returns the most relevant messages
        from the ChromaDB messages collection in order,
        for a single query text.

        Queries for text, non-empty messages only

        :param query_text: The text to query against the collection
        :param n_results: The number of results to return (default to 25)
        :return: The most relevant message from the collection
        """
        logger.debug(f"Quick querying for text: {query_text}")

        query_model = QueryDatabaseModel(
            query_text=query_text,
            num_results=n_results,
        )

        query_res = self._fully_query(query_model)

        try:
            results = self._flatten_query_documents(query_res)
        except ValueError as e:
            logger.error(f"Error parsing query result: {e}")
            raise ValueError("No documents found in the query result.") from e

        return results

    def get_conversation_by_id(
        self,
        conversation_id: str,
    ) -> List[MessageNode]:
        """
        Retrieve all messages belonging to a specific conversation from the ChromaDB messages collection.

        :param conversation_id: The unique identifier of the conversation to retrieve.

        :return: A list of MessageNode objects representing the messages in the conversation.
                 Returns an empty list if no messages are found for the given conversation ID.

        Raises:
            Any exceptions raised by the underlying ChromaDB API or data casting methods.

        Logs:
            Logs the retrieval attempt with the provided conversation ID at the debug level.
        """
        logger.debug(f"Retrieving conversation for ID: {conversation_id}")
        query_res: chromadb.api.types.GetResult = self.mess_collection.get(
            where={"conversation_id": conversation_id},
            include=[ChromaInclude.documents, ChromaInclude.metadatas, ChromaInclude.embeddings],
        )

        if not query_res["ids"]:
            raise ValueError(f"No conversation found with ID: {conversation_id}")

        if not query_res["documents"] or not query_res["metadatas"] or not query_res["embeddings"]:
            raise ValueError("Failed to retrieve messages contents, metadata or embeddings.")

        try:
            iterable_result = zip(
                query_res["ids"],
                query_res["documents"],
                query_res["embeddings"],
                query_res["metadatas"],
            )
            return [MessageNode(*item) for item in iterable_result]
        except Exception as e:
            logger.error(f"Error processing get result of conversation contents by ID: {e}")
            raise ValueError("Error processing get result of conversation contents by ID.") from e


    def get_conversation_title(self, conversation_id: str) -> str:
        """
        Retrieve the title of a conversation by its ID from the ChromaDB conversations collection.

        :param conversation_id: The unique identifier of the conversation to retrieve the title for.
        :return: The title of the conversation as a string.
        :raises ValueError: If no conversation is found with the given ID.
        """
        logger.debug(f"Retrieving title for conversation ID: {conversation_id}")
        query_res = self.conv_collection.get(
            ids=[conversation_id],
        )
        if not query_res["documents"]:
            logger.error(f"No conversation found with ID: {conversation_id}")
            raise ValueError(f"No conversation found with ID: {conversation_id}")

        return query_res["documents"][0]

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
    
    def stream_messages(
            self,
            batch_size: int = 10000,
            include_content: bool = False,
            include_metadata: bool = False,
            offset: int = 0,
        ) -> Iterable[chromadb.api.types.GetResult]:
        """
        Stream messages from the ChromaDB messages collection in batches.

        :param batch_size: The number of messages to retrieve in each batch (default is 5000)
        :param include_content: Whether to include message content in the output (default is False)
        :param include_metadata: Whether to include metadata in the output (default is False)
        :return: An iterable of dictionaries containing message IDs,
                 and optionally content and metadata.
        """
        include_fields = []
        if include_content:
            include_fields.append(ChromaInclude.documents)
        if include_metadata:
            include_fields.append(ChromaInclude.metadatas)

        while True:
            query_res: chromadb.GetResult = self.mess_collection.get(
                limit=batch_size,
                offset=offset,
                include=include_fields,
            )
            
            offset += batch_size
            yield query_res

            if len(query_res["ids"]) < batch_size:
                # Reached the end of the collection
                break
    

    def close(self) -> None:
        """
        Close the ChromaDB client connection.
        """
        logger.debug("Closing ChromaDB client connection.")
        del self.client 
        # ChromaDB PersistentClient (ClientAPI) does not have a close method. We dereference the API instead.
        # TODO : future versions might use other clients that expose explicit closing.
        # Make sure to be aware and update in consequence.


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
        hnsw_params: Optional[Dict[str, int | str]] = None,
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
            hnsw_params (Dict[str, int] | None, optional): HNSW index parameters (e.g.:
              {'hnsw:space':'cosine', 'hnsw:M': 32, 'hnsw:ef_construction': 400})
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

        # Handle HNSW parameters
        self.hnsw_params = hnsw_params or {}

    def _create_collections(self, client: chromadb.api.ClientAPI) -> None:
        """
        Create the ChromaDB collections with the specified names.
        :param client: The ChromaDB client to use for creating collections
        """
        logger.info("Creating ChromaDB collections...")
        for i, key in enumerate(COLLECTIONS_NAMES.to_dict().keys()):
            collection_to_create = self.collection_names[key]
            # # Merge HNSW params into metadata
            # configuration = self.hnsw_params
            client.create_collection(
                name=collection_to_create,
                embedding_function=self.embedding_functions[i],
                metadata=self.hnsw_params,  # Pass config with HNSW params
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

    LoggerSetup.configure_logger(console_level="TRACE")

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

    # QUICKLY QUERY THE DATABASE FOR A GIVEN TEXT INPUT
    querier = ChromaQuerier()
    query_text = None

    if query_text is None:
        query_text = str(
            input("Enter the text to query against the ChromaDB messages collection: ")
        )

    results = querier.quick_query(query_text=query_text)

    for i, result in enumerate(results):
        print(
            "{color}[{rank:>3}]{uncolor} {content}\n".format(
                color=ANSI_CYAN,
                rank=i + 1,
                uncolor=ANSI_RESET,
                content=result,
            )
        )
    logger.success("ChromaDB query process completed successfully.")

    # # CREATE A NEW CHROMADB PERSISTENT DATABASE
    # hnsw_params = {
    #     "hnsw:space": "l2",
    #     "hnsw:construction_ef": 1024,
    #     "hnsw:M": 128,
    #     "hnsw:search_ef": 512,
    # }

    # creator = ChromaCreator(hnsw_params=hnsw_params)
    # client = creator.create()
    # upserter = ChromaUpserter(client=client)
    # upserter.upsert_all_conversations()
    # logger.info("ChromaDB upsert process completed successfully.")
    # logger.info("You can now use the ChromaDB client to query or manipulate the data.")
    # logger.info("ChromaDB client is ready for use.")
