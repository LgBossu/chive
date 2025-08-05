from typing import Dict, List, Mapping, NamedTuple, Tuple, Union

import chromadb
import chromadb.api
import chromadb.api.configuration
import chromadb.api.types
from loguru import logger

from backend.loaders.conversation_parser import ConversationParser, ParsedMessage
from backend.utils import hash_utils

Metadata = Mapping[
    str, Union[str, int, float, bool]
]  # Mimic chromadb.Metadata type for easier type hinting

ChromaDocument = chromadb.api.types.Document
ChromaInclude = chromadb.api.types.IncludeEnum
ChromaEmbedding = Union[chromadb.api.types.Embedding, chromadb.api.types.PyEmbedding]


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
        # URGENT TODO : make the metadata a TypedDict, like, why didn't I do it before?
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
