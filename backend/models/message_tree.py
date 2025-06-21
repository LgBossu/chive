from typing import List, Optional, Tuple

from backend.models.message_node import MessageNode


class MessageTree:
    """A class representing a tree structure of messages in a conversation."""

    def __init__(self) -> None:
        """Initializes an empty message tree."""
        pass

    def _get_message_by_id(self, message_id: str) -> MessageNode:  # type: ignore # TODO: Implement, then suppress type:ignore  # noqa: E501
        """Retrieves a message node by its identifier.

        Args:
            message_id (str): The identifier of the message node to retrieve.

        Returns:
            MessageNode: The matching MessageNode.

        Raises:
            ValueError: If no message node with the given ID is found.
        """
        pass

    def find_message(self, message_id: str) -> Optional[MessageNode]:
        """Searches for and returns the message node with the given identifier.
        If the message node is not found, returns None.
        This method is a wrapper around `_get_message_by_id` to handle the case
        where the message node might not exist, returning None instead of raising
        an exception.
        This is useful for cases where the existence of the message node is uncertain,
        and you want to avoid exceptions in the control flow.

        Args:
            message_id (str): The identifier of the message node to search for.

        Returns:
            Optional[MessageNode]: The found message node, or None if not found.
        """
        try:
            return self._get_message_by_id(message_id)
        except ValueError:
            return None

    def remove_message(self, message_id: str) -> None:
        """Removes a message node from the tree by its identifier.

        Args:
            message_id (str): The identifier of the message node to be removed.
        """
        match = self._get_message_by_id(message_id)
        if match is None:
            raise ValueError(f"Message with ID {message_id} not found.")
        if match is not None:
            match.delete()

    def traverse_depth_first(self) -> List[MessageNode]:  # type: ignore # TODO: Implement, then suppress type:ignore  # noqa: E501
        """Traverses the tree in depth-first order.

        Returns:
            List[MessageNode]: A list of message nodes following a depth-first traversal.
        """
        pass

    def traverse_breadth_first(self) -> List[MessageNode]:  # type: ignore # TODO: Implement, then suppress type:ignore  # noqa: E501
        """Traverses the tree in breadth-first order.

        Returns:
            List[MessageNode]: A list of message nodes following a breadth-first traversal.
        """
        pass

    def traverse_main_branch(self) -> List[MessageNode]:  # type: ignore # TODO: Implement, then suppress type:ignore  # noqa: E501
        """Traverses the main branch of the tree, which is defined as the longest path
        from the root to a leaf node.

        Returns:
            List[MessageNode]: A list of message nodes following the main branch traversal.
        """
        pass

    def get_all_nodes(self) -> List[MessageNode]:  # type: ignore # TODO: Implement, then suppress type:ignore  # noqa: E501
        """Returns a list of all message nodes in the tree.
        Flattens the tree structure into a single list of message nodes.

        - TODO: detail the specs. Ideally, we want to list the main branch first,
        then all successive branches from longest to shortest remaining branches.
        - TODO: ideally, we'd want a list like that with separator flags to indicate
        the start of a new branch, and the end of a branch.
        Maybe return a list of indices ALONG with the nodes, to indicate
        where the branches start anew.

        Returns:
            List[MessageNode]: A list containing all the message nodes in the tree.
        """
        pass

    def count_nodes(self) -> int:  # type: ignore # TODO: Implement, then suppress type:ignore  # noqa: E501
        """Counts the total number of message nodes in the tree.

        Returns:
            int: The total count of message nodes.
        """
        pass

    def format_for_display(self) -> Tuple[str, List[str]]:  # type: ignore # TODO: Implement, then suppress type:ignore  # noqa: E501
        """Formats the message tree for display.

        Returns:
            Tuple[str, List[str]]: A tuple containing the title of the tree and a list of
            formatted strings representing each message node.
        """
        pass
