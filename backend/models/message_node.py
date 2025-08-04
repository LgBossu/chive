from json import JSONDecodeError
from json import loads as json_loads
from typing import Dict, List, Optional, Union

from backend.loaders.chroma_endpoints import ChromaEmbedding, Metadata

# TODO : document properly


class MessageNode:
    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation of the MessageNode."""
        return {
            "id": self._id,
            "content": self._content,
            "embeddings": (
                list(self._embeddings)
                if hasattr(self._embeddings, "__iter__")
                else self._embeddings
            ),
            "metadata": self._metadata,
            "prev": self.prev.id if self.prev else None,
            "next": [n.id for n in self.next],
        }
    def __init__(
        self,
        id: str,
        content: str,
        embeddings: ChromaEmbedding,
        metadata: Union[Metadata, Dict[str, Union[str, float, int, bool]]],
    ) -> None:
        self._id: str = id
        self._content: str = content
        self._embeddings: ChromaEmbedding = embeddings
        if not isinstance(metadata, dict):
            metadata = dict(metadata)
        self._metadata: Dict[str, Union[str, float, int, bool]] = metadata or {}
        # Metadata can include any additional information about the message.
        # For example, it could include timestamps, user IDs, etc.
        self.prev: Optional["MessageNode"] = None
        # Previous node in the conversation tree.
        self.next: List["MessageNode"] = []
        # Next nodes in the conversation tree.

    @property
    def id(self) -> str:
        """Returns the ID of the message node."""
        return self._id

    @property
    def content(self) -> str:
        """Returns the content of the message node."""
        return self._content

    # TODO : fix below to provide a proper embedding property accounting for various embedding types supported by ChromaDB.  # noqa: E501
    # @property
    # def embeddings(self) -> List[float]:
    #     """Returns the embeddings of the message node."""
    #     return self._embeddings.copy()

    @property
    def metadata(self) -> Dict[str, Union[str, float, int, bool]]:
        """Returns the metadata of the message node."""
        return self._metadata.copy()

    @property
    def metadata_parent_id(self) -> Optional[str]:
        """Returns the parent ID from the metadata, if it exists."""
        res = self._metadata.get("parent_id", None)
        if res is None or not isinstance(res, str):
            raise ValueError("The 'parent_id' in metadata is unset or not a string.")
        return res

    @property
    def metadata_children_ids(self) -> List[str]:
        """Returns the children IDs from the metadata, if they exist."""
        string_children = self._metadata.get("children", "[]")
        if not isinstance(string_children, str):
            raise ValueError("The 'children' in metadata is not a string.")

        try:
            children_ids = json_loads(string_children)
        except JSONDecodeError:
            raise ValueError("The 'children' in metadata is not a valid JSON string.")

        if not isinstance(children_ids, list):
            raise ValueError("The 'children' in metadata is not a list.")
        return children_ids

    @property
    def branches_out(self) -> bool:
        """Returns True if the message node has multiple successors, False otherwise."""
        return len(self.next) > 1

    @property
    def is_leaf(self) -> bool:
        """Returns True if the message node is a leaf node (no successors), False otherwise."""
        return len(self.next) == 0

    @property
    def empty_or_non_text(self) -> bool:
        """Returns True if the message node is flagged as empty, False otherwise."""
        is_empty_flag = self._metadata.get("empty_or_non_text", None)
        if is_empty_flag is None:
            raise ValueError("The 'empty_or_non_text' flag is not set in the metadata.")
        assert isinstance(
            is_empty_flag, bool
        ), f"The 'empty_or_non_text' flag must be a boolean, and was not recognized as such (got {type(is_empty_flag)})."  # noqa: E501
        return is_empty_flag

    def set_prev(self, prev_node: "MessageNode") -> None:
        """Sets the previous node in the conversation tree."""
        self.prev = prev_node

    def add_next(self, next_node: "MessageNode") -> None:
        """Adds a next node in the conversation tree."""
        self.next.append(next_node)

    def remove_next(self, next_node: "MessageNode") -> None:
        """Removes a next node from the conversation tree."""
        if next_node in self.next:
            self.next.remove(next_node)
        else:
            raise ValueError("The specified next node is not in the conversation tree.")

    def reset_next(self) -> None:
        """Resets the next nodes in the conversation tree."""
        self.next = []

    def length(self) -> int:
        """Returns the length of the conversation tree starting from this node."""
        if not hasattr(self, "next") or not self.next:
            return 1
        return 1 + max(next_node.length() for next_node in self.next)
        # The method returns the length of the longest branch starting from this node.

    def count_branches_lengths(self) -> Union[int, List[int]]:
        """Counts the lengths of all branches in the conversation tree."""
        # The method returns a list of lengths if there are multiple branches,
        # or a single integer if there is only one branch.
        if not hasattr(self, "next"):
            return 0
        elif len(self.next) == 1:
            return 1 + self.next[0].length()
        else:
            return [1 + next_node.length() for next_node in self.next]

    def get_longest_next(self) -> Optional["MessageNode"]:
        """Returns the main next node in the conversation tree."""
        if not hasattr(self, "next") or not self.next:
            return None
        branches_lengths = self.count_branches_lengths()
        if isinstance(branches_lengths, int):
            return self.next[0]
        elif isinstance(branches_lengths, list) and branches_lengths:
            max_length_index = branches_lengths.index(max(branches_lengths))
            return self.next[max_length_index]
        else:
            return None

    # def __del__(self) -> None:
    #     """Deletes GC-wise."""

    def delete(self) -> None:
        """Deletes the message node from the tree,
        and rewire the previous and next nodes together,
        effectively removing it from the conversation tree."""
        if self.prev is not None:
            # If there is a previous node, remove oneself from its next list.
            self.prev.remove_next(self)
        for next_node in self.next:
            # If there are next nodes, rewire them to the previous node.
            if self.prev is not None:
                self.prev.add_next(next_node)
            next_node.prev = self.prev  # This statement works whether prev is None or not.

        # Clear the current node's connections.
        # Reset all connections to allow garbage collection.
        self.prev = None
        self.next = []

    def __repr__(self) -> str:
        return f"MessageNode(id={self._id}, content={self._content[:20]}...)"

    def __hash__(self) -> int:
        """
        Tree agnostic hash function.

        Returns the ID of the message node as its hash.
        We do not consider the spot of the node in the tree : if two notes have the same ID,
        they have the same inner values and are considered equal.
        """
        return self._id.__hash__()  # Id is already a hash actually.

    def __eq__(self, other: object) -> bool:
        """
        Equality check based on ID.

        Two MessageNode instances are considered equal if their IDs are the same.
        This is a tree agnostic equality check, meaning it does not consider the position
        of the node in the conversation tree, only the ID.

        :param: The object to compare with.

        :return: True if the IDs are the same, False otherwise.
        :rtype: bool
        """
        if not isinstance(other, MessageNode):
            return False  # A message node is only equal to another message node.
        return self._id == other._id
