from typing import List, Optional, Tuple, Union

from backend.loaders.chroma_endpoints import ChromaQuerier
from backend.models.message_node import MessageNode


class MessageNodeSet:
    """A set of message nodes that can be used to build a message tree.

    We must check that all nodes are :
    - from the same conversation
    - connected in a tree structure.
    """

    def __init__(self, nodes: List[MessageNode]) -> None:
        """Initializes the MessageNodeSet with a list of message nodes."""
        if not nodes:
            raise ValueError("The list of nodes cannot be empty.")
        self.nodes = nodes
        self.nodes.sort(key=lambda node: node.id)  # Sort nodes by ID for binary search efficiency.
        self.conversation_id = nodes[0].metadata.get("conversation_id", None)

        if not self.conversation_id:
            raise ValueError("Conversation ID missing from a node (first node).")
        if len(self.nodes) != len({node.id for node in self.nodes}):
            raise ValueError("Nodes must be unique within the set (duplicate IDs found).")

        self.root = None

    def _get_message_by_id(self, message_id: str) -> MessageNode:
        """Retrieves a message node by its identifier.

        Args:
            message_id (str): The identifier of the message node to retrieve.

        Returns:
            MessageNode: The matching MessageNode.

        Raises:
            ValueError: If no message node with the given ID is found.
        """
        # Perform binary search to find the message node by ID
        left, right = 0, len(self.nodes) - 1
        while left <= right:
            mid = (left + right) // 2
            mid_id = self.nodes[mid].id
            if mid_id == message_id:
                return self.nodes[mid]
            elif mid_id < message_id:
                left = mid + 1
            else:
                right = mid - 1
        raise ValueError(f"Message with ID {message_id} not found.")

    def _check_conversation_id(self) -> None:
        for node in self.nodes:
            if node.metadata.get("conversation_id") != self.conversation_id:
                raise ValueError("All nodes must have the same conversation ID.")

    def _check_connected(self) -> None:
        """Checks if the nodes form a connected tree structure."""
        # TODO : check for correctness of the method
        all_nodes = {node.id for node in self.nodes}
        all_children = {child_id for node in self.nodes for child_id in node.metadata_children_ids}
        all_parents = {
            node.metadata_parent_id for node in self.nodes if node.metadata_parent_id is not None
        }

        root_candidates = all_nodes - all_children
        if len(root_candidates) != 1:
            raise ValueError("There must be exactly one root node in the tree.")
        self.root = self._get_message_by_id(next(iter(root_candidates)))

        connected_ids = all_parents | all_children | root_candidates
        if not all_nodes.issubset(connected_ids):
            raise ValueError(
                "The nodes do not form a connected tree structure (orphan nodes exist)."
            )

    def _check_acyclic(self) -> None:
        """Checks if the nodes form an acyclic structure."""
        visited = set()
        stack = set()

        def visit(node: MessageNode) -> None:
            if node.id in stack:
                raise ValueError("The tree contains a cycle.")
            if node.id not in visited:
                visited.add(node.id)
                stack.add(node.id)
                for child_id in node.metadata_children_ids:
                    child_node = self._get_message_by_id(child_id)
                    visit(child_node)
                stack.remove(node.id)

        assert (
            self.root is not None
        ), "Root node must be set before checking acyclic structure. Make sure you called _check_connected() first."  # noqa: E501
        visit(self.root)

    def validate(self) -> None:
        """Validates the set of message nodes to ensure they form a valid tree."""
        self._check_conversation_id()  # Ensure all nodes have the same conversation ID
        self._check_connected()  # Ensure the nodes are connected with a unique root
        self._check_acyclic()  # Ensure the graph is acyclic

    @property
    def is_valid(self) -> bool:
        """Checks if the set of message nodes is valid."""
        try:
            self.validate()
            return True
        except ValueError:
            return False

    @property
    def root_node(self) -> MessageNode:
        """Returns the root node of the message tree, if it exists."""
        if self.root is None:
            raise ValueError("Root node is not set. Make sure to validate the node set first.")
        return self.root


class MessageTree:
    """A class representing a tree structure of messages in a conversation."""

    def __init__(
        self,
        source: Union[MessageNode, MessageNodeSet],
        highlights: List[str] = [],
        querier: Optional[ChromaQuerier] = None,
    ) -> None:
        """Initializes an empty message tree.

        We can build a message tree by :
        - passing a single message node
            - the constructor will then fetch from database all messages of matching conversation ID
        - passing a list or set of message nodes from the same conversation

        The constructor validates the assumed tree structure of the input nodes, and then connects them
        into a proper tree structure.

        Args:
            source (Union[MessageNode, MessageNodeSet]): The source message node or set of message nodes.
            highlights (List[str], optional): A list of message IDs to highlight in the tree. Defaults to [].
            querier (Optional[ChromaQuerier]): An optional querier to fetch additional messages from the database. If not provided, a new querier will be initialized at runtime if needed (that is, if `source` is a `MessageNode` type).
        Raises:
            ValueError: If the source nodes do not form a single connected tree structure.
        """  # noqa: E501
        if isinstance(source, MessageNode):
            conv_id = source.metadata.get("conversation_id", None)
            assert isinstance(
                conv_id, str
            ), "Conversation ID missing from the source node or is mistyped. Cannot infer matching nodes."  # noqa: E501

            if querier is None:
                querier = ChromaQuerier()

            all_nodes_list = [
                MessageNode(*yielded_data)
                for yielded_data in querier.get_conversation_by_id(conv_id)
            ]

            source = MessageNodeSet(all_nodes_list)

        assert isinstance(source, MessageNodeSet), "Source must be a MessageNode or MessageNodeSet."
        source.validate()

        self.root: MessageNode = source.root_node
        self.nodes: list[MessageNode] = []

        # List to store message IDs that were returned by the query
        self.highlights = highlights.copy()

        def _connect_and_add(node: MessageNode) -> None:
            """Recursively connects the node to its children and adds it to the nodes list."""
            self.nodes.append(node)
            for child_id in node.metadata_children_ids:
                child_node = source._get_message_by_id(child_id)
                node.add_next(child_node)
                child_node.set_prev(node)
                _connect_and_add(child_node)

        _connect_and_add(self.root)

        assert (
            len(self.nodes) == len(source.nodes)
        ), (
            "The number of nodes in the tree does not match the source nodes."
        )  # TODO: remove this sanity check if it is not needed.

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
            if match.id == self.root.id:
                raise NotImplementedError("Cannot remove the root node from the tree.")
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
