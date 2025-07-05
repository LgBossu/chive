import json
from typing import Any, Dict, List

from loguru import logger

from backend.loaders.conversation_parser import ConversationParser
from backend.utils.path_utils import get_paths


class ConversationLoader:
    def __init__(self):
        logger.trace("Initializing ConversationLoader...")
        SOURCE_JSON_PATH = get_paths().source_conversations_path
        logger.trace("Source JSON path loaded")
        self.source_path = SOURCE_JSON_PATH

        with open(SOURCE_JSON_PATH, "r", encoding="utf-8") as file:
            try:
                self.source_json: List[Dict[str, Any]] = json.load(file)
            except json.JSONDecodeError as e:
                logger.critical(f"Failed to decode JSON from {SOURCE_JSON_PATH}: {e}")
                raise ValueError(f"Invalid JSON format in {SOURCE_JSON_PATH}") from e

        logger.trace("Source JSON loaded successfully")

        self._conversations: List[ConversationParser] = [
            ConversationParser(conversation) for conversation in self.source_json
        ]

        logger.trace(f"Loaded {len(self._conversations)} conversations from JSON.")

        self._uncategorized_conversations = list(range(len(self._conversations)))

        logger.info("ConversationLoader initialized.")

    def initialize_idx(self, idx: int) -> None:
        """
        Initialize the conversation at the given index.
        This method can be used to set up any necessary state or resources for the conversation.
        :param idx: The index of the conversation to initialize
        """
        if idx < 0 or idx >= len(self._conversations):
            logger.error(f"Index {idx} is out of bounds for conversations list.")
            return

        logger.debug(f"Initializing conversation at index {idx}.")
        try:
            self._conversations[idx].parse_conversation()
            logger.debug(f"Initialized conversation at index {idx}.")
            self._uncategorized_conversations.remove(idx)
            # The above approach is acceptable, as the list of conversations is relatively small.
        except Exception as e:
            logger.error(f"Failed to initialize conversation at index {idx}: {e}")
            raise RuntimeError(f"Initialization failed for conversation at index {idx}") from e

    def get_conversation(self, idx: int) -> ConversationParser:
        """
        Retrieve the conversation at the given index.
        :param idx: The index of the conversation to retrieve
        :return: The ConversationParser instance for the specified index
        :raises IndexError: If the index is out of bounds
        """
        if idx < 0 or idx >= len(self._conversations):
            logger.error(f"Index {idx} is out of bounds for conversations list.")
            raise IndexError(f"Index {idx} is out of bounds for conversations list.")

        if idx in self._uncategorized_conversations:
            logger.warning(f"Conversation at index {idx} is not initialized yet.")
            self.initialize_idx(idx)

        return self._conversations[idx]

    @property
    def conversations(self) -> List[ConversationParser]:
        """
        Get the list of all conversations.
        :return: A list of ConversationParser instances

        WARNING: ConversationParsers are mutable objects.
        As such, do not modify or call methods on them if you don't know what you're doing.
        """
        return [conv for conv in self._conversations if conv is not None]

    @property
    def uncategorized_conversations(self) -> List[int]:
        """
        Get the list of indices of uncategorized conversations.
        :return: A list of indices of uncategorized conversations
        """
        return self._uncategorized_conversations.copy()

    def __iter__(self):
        """
        Make the ConversationLoader iterable.
        As each conversation is accessed through iteration, ensure it is parsed.
        """
        logger.trace("Starting iteration over conversations.")
        for idx, conversation in enumerate(self._conversations):
            logger.trace(f"Iterating over conversation at index {idx}.")
            if idx in self._uncategorized_conversations:
                logger.debug(f"Parsing conversation at index {idx} during iteration.")
                self.initialize_idx(idx)
            yield conversation

    def __len__(self) -> int:
        """
        Get the number of conversations loaded.
        :return: The number of conversations
        """
        return len(self._conversations)

    def __getitem__(self, idx: int) -> ConversationParser:
        """
        Get the conversation at the specified index.
        :param idx: The index of the conversation to retrieve
        :return: The ConversationParser instance for the specified index
        :raises IndexError: If the index is out of bounds
        """
        if idx < 0 or idx >= len(self._conversations):
            logger.error(f"Index {idx} is out of bounds for conversations list.")
            raise IndexError(f"Index {idx} is out of bounds for conversations list.")

        if idx in self._uncategorized_conversations:
            logger.warning(f"Conversation at index {idx} is not initialized yet.")
            self.initialize_idx(idx)

        return self._conversations[idx]
