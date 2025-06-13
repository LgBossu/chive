from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

from loguru import logger


class JSONKeys(Enum):
    """
    Enum class for the JSON keys.
    """

    TITLE = "title"
    MAPPING = "mapping"
    MESSAGE = "message"
    CONTENT = "content"
    AUTHOR = "author"
    ROLE = "role"
    CONTENT_TYPE = "content_type"
    CONTENT_PARTS = "parts"
    CUR_ID = "id"
    PARENT_ID = "parent"
    CHILDREN_IDS = "children"
    TIMESTAMP = "create_time"


class ContentTypes(Enum):
    """
    Enum class for the content types.
    """

    TEXT = "text"
    CODE = "code"
    MULTIMODAL_TEXT = "multimodal_text"
    TETHER_BROWSING_DISPLAY = "tether_browsing_display"
    TETHER_QUOTE = "tether_quote"


# Define the accepted content types
accepted_content_types = [ContentTypes.TEXT.value, ContentTypes.CODE.value]

# Operations to format text for each accepted content type
format_text_content = {
    ContentTypes.TEXT.value: lambda content: "".join(content["parts"]),
    ContentTypes.CODE.value: lambda content: content["text"],
}


@dataclass
class MessageMetadata:
    """
    Dataclass to hold metadata for a message.
    """

    role: str
    cur_id: str
    parent_id: str | None
    children_ids: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Union[str, int, float, bool]]:
        """
        Convert the metadata to a dictionary.
        :return: A dictionary representation of the metadata

        The type hinting matches that of chromadb.Metadata,
        which is a mapping of string keys to values of various types.
        This function is critical for hashing messages in the pipeline,
        as the hash argument is this method's output.
        :rtype: Dict[str, Union[str, int, float, bool]]
        """
        return {
            "role": self.role,
            "cur_id": self.cur_id,
            "parent_id": self.parent_id if self.parent_id is not None else str(None),
            "children_ids": str(self.children_ids),
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return str(self.to_dict())


@dataclass
class ParsedMessage:
    content: str
    metadata: MessageMetadata

    def __str__(self) -> str:
        """
        String representation of the ParsedMessage.
        :return: A string representation of the message content and metadata

        This function is critical for hashing messages in the pipeline,
        as the hash argument is this method's output.
        """
        return self.content + str(self.metadata)


class ConversationParser:
    def __init__(self, conversation_data: Dict[str, Any]) -> None:
        """
        Initialize the parser with raw conversation data.
        Immediately parse the conversation into structured attributes.
        """
        self._conversation_data = conversation_data
        self._title: str | None = None
        self._messages: list[ParsedMessage] | None = None

    def _retrieve_key(self, jsonkey: JSONKeys) -> Any:
        """
        Retrieve a value from the conversation data using the specified JSON key.
        Raises ValueError if the key is not found.
        :param jsonkey: The JSON key to retrieve
        :return: The value associated with the JSON key
        :raises ValueError: If the key is not found in the conversation data
        """
        try:
            return self._conversation_data[jsonkey.value]
        except KeyError as e:
            logger.warning(
                f"No value found under {jsonkey.value} found in conversation data, parsing cannot proceed."  # noqa: E501
            )
            raise ValueError(f"Key {jsonkey.value} missing in conversation data") from e

    def _parse_message_content(self, content: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Parse the content of a message and return its text representation.
        :param content: The content dictionary of the message
        :return: A tuple containing a boolean indicating success and the text content
        """
        content_type = content.get(JSONKeys.CONTENT_TYPE.value, "")
        if content_type not in accepted_content_types:
            return False, "[non-text content]"
        return True, format_text_content[content_type](content)

    def _parse_individual_message(
        self, message_data: Dict[str, Any]
    ) -> None | Tuple[bool, ParsedMessage]:
        """
        Parse an individual message from the conversation data.
        :param message_data: The message data to parse
        :return: A tuple containing the content and metadata of the message
        """
        if not isinstance(message_data, dict):
            logger.error("Message data must be a dictionary")
            raise ValueError("Message data must be a dictionary")

        message = message_data.get(JSONKeys.MESSAGE.value, None)

        # Handle the None case
        if message is None:
            return None

        # Extract metadata and content.
        try:
            cur_id = message_data[JSONKeys.CUR_ID.value]
            parent_id = message_data[JSONKeys.PARENT_ID.value]
            children_ids = message_data[JSONKeys.CHILDREN_IDS.value].copy()
        except KeyError as e:
            logger.error(f"Missing key in message data: {e}")
            raise ValueError(f"Missing key in message data: {e}") from e

        try:
            role = message[JSONKeys.AUTHOR.value][JSONKeys.ROLE.value]
            timestamp = message_data[JSONKeys.TIMESTAMP.value]
        except KeyError as e:
            logger.error(f"Missing key in message author or timestamp: {e}")
            raise ValueError(f"Missing key in message author or timestamp: {e}") from e

        accepted_content, content = self._parse_message_content(
            message.get(JSONKeys.CONTENT.value, {})
        )

        metadata = MessageMetadata(
            role=role,
            cur_id=cur_id,
            parent_id=parent_id,
            children_ids=children_ids,
            timestamp=timestamp,
        )
        parsed_message = ParsedMessage(content=content, metadata=metadata)

        return (accepted_content, parsed_message)

    def parse_conversation(self) -> None:
        logger.debug("Beginning to parse conversation")

        # Retrieve title
        self._title = self._retrieve_key(JSONKeys.TITLE)
        assert isinstance(self._title, str), "Title must be a string"
        logger.debug(f"Parsing conversation titled: {self._title.encode('utf-8', 'replace')}")

        # Retrieve messages mapping as a dictionary
        messages_dict = self._retrieve_key(JSONKeys.MAPPING)
        assert isinstance(messages_dict, dict), "Messages mapping must be a dictionary"
        logger.debug("Successfully retrieved messages mapping.")

        # Keep track of total messages and messages of accepted content types
        total, accepted = 0, 0

        # Initialize the messages list
        self._messages = []

        # Process each message in the conversation.
        for key, message_data in messages_dict.items():
            # Process the message in all generality
            parsing_result = self._parse_individual_message(message_data)
            total += 1  # Increment total messages processed

            if parsing_result is None:
                # In the None case, we skip the message (nothing relevant to append)
                logger.debug(f"Message {key} is None or not found, skipping.")
                continue
            else:
                # In the case of a valid message, we unpack the result
                success, parsed_message = parsing_result
                if not success:
                    logger.debug(f"Message {key} has non-accepted content type.")

                self._messages.append(parsed_message)
                accepted += success  # Increment accepted messages if content is valid

        logger.debug(f"Processed {total} messages, {accepted} with accepted content type.")

    @property
    def title(self) -> str:
        """
        Returns the conversation title.
        """
        if self._title is None:
            logger.error("Title has not been set. Ensure conversation data is valid.")
            raise ValueError("Title has not been set. Ensure conversation data is valid.")
        assert isinstance(self._title, str), "Title must be a string"
        return self._title

    @property
    def messages(self) -> List[ParsedMessage]:
        """
        Returns the structured messages.
        """
        if self._messages is None:
            logger.error("Messages have not been parsed. Ensure conversation data is valid.")
            raise ValueError("Messages have not been parsed. Ensure conversation data is valid.")
        return self._messages


if __name__ == "__main__":
    # Example usage; in practice you might load a JSON file.
    example_data = {
        "title": "Test Conversation",
        "mapping": {
            "1": {
                "id": "1",
                "parent": None,
                "children": [],
                "create_time": "2025-06-10T12:00:00Z",
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hello, world!"]},
                },
            }
        },
    }

    parser = ConversationParser(example_data)
    logger.info(f"Conversation title: {parser.title}")
    for parsed_message in parser.messages:
        logger.info(f"Message: {parsed_message.content} | Metadata: {parsed_message.metadata}")
