from enum import Enum
from typing import Any, Dict, List, Tuple

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


class ConversationParser:
    def __init__(self, conversation_data: Dict[str, Any]) -> None:
        """
        Initialize the parser with raw conversation data.
        Immediately parse the conversation into structured attributes.
        """
        self.conversation_data = conversation_data
        self._title: str = "Untitled"
        self._messages: List[Tuple[str, Dict[str, Any]]] = []
        self._parse_conversation()

    def _parse_conversation(self) -> None:
        logger.debug("Beginning to parse conversation")
        # Retrieve title
        self._title = self.conversation_data.get(JSONKeys.TITLE.value, "Untitled")
        logger.debug(f"Parsing conversation titled: {self._title}")

        messages_dict = self.conversation_data.get(JSONKeys.MAPPING.value, {})
        total, accepted = 0, 0

        # Process each message in the conversation.
        for key, message_data in messages_dict.items():
            message = message_data.get(JSONKeys.MESSAGE.value)
            if message is None:
                logger.debug(f"Message {key} is None")
                continue

            # Extract metadata and content.
            role = message.get(JSONKeys.AUTHOR.value, {}).get(JSONKeys.ROLE.value, "unknown")
            cur_id = message_data.get(JSONKeys.CUR_ID.value)
            parent_id = message_data.get(JSONKeys.PARENT_ID.value)
            children_ids = message_data.get(JSONKeys.CHILDREN_IDS.value, [])
            timestamp = message_data.get(JSONKeys.TIMESTAMP.value, "no-timestamp")

            success, content = self._extract_text_content(message.get(JSONKeys.CONTENT.value, {}))
            self._messages.append(
                (
                    content,
                    {
                        "role": role,
                        "cur_id": cur_id,
                        "parent_id": parent_id,
                        "children_ids": children_ids,
                        "timestamp": timestamp,
                    },
                )
            )
            total += 1
            accepted += int(success)

        logger.debug(f"Processed {total} messages, {accepted} with accepted content type.")

    def _extract_text_content(self, content: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Extract text content based on the content type.
        """
        content_type = content.get(JSONKeys.CONTENT_TYPE.value, "")
        if content_type not in accepted_content_types:
            return False, "[non-text content]"
        return True, format_text_content[content_type](content)

    @property
    def title(self) -> str:
        """
        Returns the conversation title.
        """
        return self._title

    @property
    def messages(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Returns the structured messages.
        """
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
    for content, meta in parser.messages:
        logger.info(f"Message: {content} | Metadata: {meta}")
