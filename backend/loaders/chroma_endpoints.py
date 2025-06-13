"""Wrappers for the ChromaDB access, as endpoints with safe methods"""

from typing import Dict, List, Mapping, NamedTuple, Tuple, Union

import chromadb
from loguru import logger

from backend.loaders.conversation_loader import ConversationLoader
from backend.loaders.conversation_parser import ConversationParser, ParsedMessage
from backend.utils import hash_utils as hash_utils
from backend.utils.path_utils import get_paths

DB_PATH = get_paths().chroma_db_path

if not DB_PATH.exists():
    logger.warning(
        f"ChromaDB path {DB_PATH} does not exist. Recommended checking env variables or creating a new database."  # noqa: E501
    )
    # DB_PATH.mkdir(parents=True, exist_ok=True)
    raise FileNotFoundError(
        f"ChromaDB path {DB_PATH} does not exist. Please check the configuration."
    )

client = chromadb.PersistentClient(path=str(DB_PATH))


Metadata = Mapping[str, Union[str, int, float, bool]]  # Mimic chromadb.Metadata type


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
        message_id = hash_utils.hash_message(str(message))
        compatible_metadata = message.metadata.to_dict()
        compatible_metadata["conv_id"] = self.conversation_id
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


class ChromaUpserter:
    """A wrapper class to properly upsert conversations into ChromaDB."""

    # TODO : find a way for the script to flag or skip conversations
    # that are already in the database in entirety

    def __init__(self) -> None:
        """
        Initialize the ChromaUpserter with the ChromaDB client and collections.
        """
        self.client = client

        self.conv_collection_name = "conversations"
        self.mess_collection_name = "messages"  # TODO : store names properly in env or something

        self.conv_collection = self.client.get_collection(self.conv_collection_name)
        self.mess_collection = self.client.get_collection(self.mess_collection_name)

        self.conversation_loader = ConversationLoader()

    def upsert_conversation(self, conversation: ConversationParser) -> None:
        """
        Upsert a single conversation into the ChromaDB collections.
        :param conversation: The ConversationParser instance containing the parsed conversation
        """
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

    def upsert_all_conversations(self) -> None:
        """
        Upsert all conversations from the ConversationLoader into the ChromaDB collections.
        This method iterates through all conversations and upserts them one by one.
        """
        logger.info("Starting upsert of all conversations.")
        for conversation in self.conversation_loader:
            self.upsert_conversation(conversation)
        logger.success("All conversations have been upserted successfully.")


class ChromaQuerier:
    pass


class ChromaCreator:
    pass


if __name__ == "__main__":
    from backend.utils.log_setup import LoggerSetup

    LoggerSetup.configure_logger()

    logger.info("Starting ChromaDB upsert process...")
    logger.warning("""You are running the ChromaDB upsert script directly.
                   This is intended for debugging purposes only.
                   This WILL also upsert all conversations.

                   This script is not intended for production use.
                   Users stay advised.""")

    # Load Chroma collections and display the number of entries in each
    collection_names = client.list_collections()
    for name in collection_names:
        col = client.get_collection(name)
        count = col.count()
        logger.info(f"Collection '{name}' has {count} entries.")

    # Example usage
    upserter = ChromaUpserter()
    upserter.upsert_all_conversations()
    logger.info("ChromaDB upsert completed.")
