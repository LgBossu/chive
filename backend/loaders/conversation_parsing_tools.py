from enum import Enum

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


# Wanted exchange format declaration
role_succession = ["user", "assistant"]

# Define the accepted content types
accepted_content_types = [ContentTypes.TEXT.value, ContentTypes.CODE.value]
# Tailored methods to format text for each accepted content type
format_text_content = {
    ContentTypes.TEXT.value: lambda content: "".join(content["parts"]),
    ContentTypes.CODE.value: lambda content: content["text"],
}


def extract_text_content(content: dict) -> tuple[bool, str]:
    """
    Extract text content from the JSON data.
    :param content: The content dictionary
    :return: The extracted text content
    """
    # Get the content type
    content_type = content[JSONKeys.CONTENT_TYPE.value]

    # Check if the content type is accepted
    if content_type not in accepted_content_types:
        return False, "[non-text content]"
    else:
        # Get the textual accepeted content
        return True, format_text_content[content_type](content)


def parse_conversation(conversation: dict) -> tuple[str, list[list[str | dict]]]:
    """
    Parse a conversation from the JSON data.
    :param conversation: The conversation data
    :return: The title and the messages list
    """
    logger.debug("Beginning to parse conversation")
    # Resolve title
    title = conversation[JSONKeys.TITLE.value]
    logger.debug(
        f"Parsing in progress for conversation with title: {title.encode('utf-8', 'replace')}"  # noqa: E501
    )

    # Assign messages
    messages_dict = conversation[JSONKeys.MAPPING.value]

    # Initialize processed exchanges list
    messages = []

    # Tracking amount of messages processed
    total = 0
    acceptable = 0

    # Process the messages
    for key in messages_dict.keys():
        # Get message data
        message = messages_dict[key][JSONKeys.MESSAGE.value]
        cur_id = messages_dict[key][JSONKeys.CUR_ID.value]
        parent_id = messages_dict[key][JSONKeys.PARENT_ID.value]
        children_ids = messages_dict[key][JSONKeys.CHILDREN_IDS.value].copy()
        timestamp = messages_dict[key].get(JSONKeys.TIMESTAMP.value, "no-timestamp")

        is_acceptable = False

        if message is not None:
            # Get author's role
            role = message[JSONKeys.AUTHOR.value][JSONKeys.ROLE.value]
            # Get message content
            is_acceptable, content = extract_text_content(message[JSONKeys.CONTENT.value])

            messages.append(
                [
                    content,
                    {
                        "role": role,
                        "cur_id": cur_id,
                        "parent_id": parent_id,
                        "children_ids": children_ids,
                        "timestamp": timestamp,
                    },
                ]
            )

        else:
            logger.debug(f"Message {key} is None")

        total += 1
        acceptable += is_acceptable

    logger.debug(f"Processed {total} messages, {acceptable} of which had accepted content type.")

    return title, messages


# def parse_json_file(file_path: str):
#     """
#     Parse a JSON file.
#     :param file_path: The file path
#     :return: The parsed conversation
#     """
#     logger.debug(f"Beginning to parse JSON file: {file_path}")
#     with open(file_path, "r") as file:
#         data = json.load(file)
#     for conv in data:
#         title, messages = parse_conversation(conv)
