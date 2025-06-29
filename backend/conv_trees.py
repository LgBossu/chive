import json
from collections.abc import Callable

import graphviz
import numpy as np
from loguru import logger
from tqdm import tqdm

# TODO : type hint properly and clean up


class MessageNode:
    def __init__(
        self,
        message_id,
        content: str,
        embeddings: np.ndarray,
        metadata: dict[str, str | float | int | bool] | None = None,
        table_id: str | None = None,
    ):
        self.id = message_id
        self.content = content
        self.embeddings = embeddings
        self.metadata = metadata or dict()
        self.predecessors = set()  # Messages leading to this one
        self.successors = set()  # Messages following this one
        self.table_id = table_id

    def get_table_id(self) -> str:
        """Returns the table ID of the message."""
        if self.table_id is None:
            raise ValueError("Table ID is not set.")
        return self.table_id

    def add_predecessor(self, node):
        self.predecessors.add(node)

    def add_successor(self, node):
        self.successors.add(node)

    def get_neighbors(self) -> set:
        """Returns both predecessors and successors."""
        return self.predecessors.union(self.successors)

    def __repr__(self):
        return f"MessageNode({self.id})"


def get_branch_length(node: MessageNode) -> int:
    """Recursively calculates the length of the longest branch starting from
    a node."""
    if not node.successors:
        return 1
    return 1 + max([get_branch_length(successor) for successor in node.successors])


class ConversationTrees:
    def __init__(self):
        # self.id = conv_id
        # self.title = title
        self.nodes: dict[str, MessageNode] = {}

    def add_message(self, message: MessageNode):
        self.nodes[message.id] = message

    def bulk_add_messages(self, messages: list[MessageNode]):
        for message in messages:
            self.add_message(message)
            # if message.metadata["conv_id"] == self.id:
            #     self.add_message(message)
            #     logger.trace(f"Added message {message.id} to conversation {self.id}")
            # else:
            #     logger.error(
            #         f"Message {message.id} does not belong in conversation {self.id}"
            #     )

    def link_messages(self, parent_id, child_id):
        """Links a parent message to a child message."""
        if (parent_id in self.nodes) and (child_id in self.nodes):
            self.nodes[parent_id].add_successor(self.nodes[child_id])
            self.nodes[child_id].add_predecessor(self.nodes[parent_id])
            logger.trace(f"Linked {parent_id} to {child_id}")
        else:
            logger.trace(
                f"Failed to link {parent_id} to {child_id} : {parent_id in self.nodes} -> {child_id in self.nodes}"  # noqa: E501
            )

    def link_all_messages(self):
        for node in self.nodes.values():
            # logger.debug(f"Metadata: {node.metadata["children_ids"]}")
            try:
                for child_id in json.loads(
                    node.metadata["children_ids"].replace("'", '"')  # type: ignore
                ):
                    self.link_messages(node.id, child_id)
            except KeyError as e:
                logger.error(f"Error on {node.id} with metadata {node.metadata}")
                logger.info(node.content)
                raise e
            parent_id = node.metadata["parent_id"]
            self.link_messages(parent_id, node.id)

    def connected_graph(self) -> tuple[bool, list[set[MessageNode]]]:
        """Returns whether the graph is connected and its components."""
        visited, components = set(), []
        for node in self.nodes.values():
            if node not in visited:
                component = set()
                queue = [node]
                while queue:
                    current = queue.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        queue.extend(current.get_neighbors())
                components.append(component)
        return len(components) == 1, components

    def equivalence_class(
        self, conv_ids_table: dict[str, str], message_id: str
    ) -> tuple[str, set[MessageNode]]:
        """Returns all connected messages as an 'equivalence class'."""
        _, components = self.connected_graph()
        if message_id:
            for component in components:
                if message_id in {node.id for node in list(component)}:
                    component_title = conv_ids_table[message_id]
                    return component_title, component

    def equivalence_classes(
        self, conv_ids_table: dict[str, str]
    ) -> list[tuple[str, set[MessageNode]]]:
        """Returns all connected messages as an 'equivalence class'."""
        _, components = self.connected_graph()
        equivalence_classes = []
        for component in components:
            component_title = conv_ids_table[list(component)[0].metadata["conv_id"]]
            equivalence_classes.append((component_title, component))
        return equivalence_classes

    def plot_graph(
        self,
        conv_ids_table: dict[str, str] | None = None,
        tag_function: Callable[[str], str] | None = None,
        color_function: Callable[[str], str] = lambda x: "#ffffff",
    ) -> graphviz.Digraph:
        graph = graphviz.Digraph(format="png")
        for node in tqdm(self.nodes.values()):
            if node.predecessors:
                if tag_function:
                    graph.node(
                        node.id,
                        f"[{node.metadata["role"]}] {tag_function(node.get_table_id())}",  # noqa: E501
                        style="filled",
                        fillcolor=color_function(node.get_table_id()),
                    )
                else:
                    graph.node(node.id, node.metadata["role"])
            elif conv_ids_table:
                graph.node(node.id, conv_ids_table[node.metadata["conv_id"]], shape="box")
            else:
                graph.node(node.id, node.metadata["conv_id"], shape="box")
            for neighbor in node.successors:
                graph.edge(node.id, neighbor.id)
        return graph

    def visualize(
        self,
        conv_ids_table: dict[str, str] | None = None,
        tag_function: Callable[[str], str] | None = None,
        color_function: Callable[[str], str] = lambda x: "#ffffff",
        view: bool = False,
    ):
        """Visualizes the conversation tree using Graphviz."""
        graph = self.plot_graph(
            conv_ids_table=conv_ids_table,
            tag_function=tag_function,
            color_function=color_function,
        )
        graph.render(
            "conversations",
            outfile="output_graphs/conversations_trees.pdf",
            directory="output_graphs",
            cleanup=True,
            view=view,
        )


####################################################################################


class SingleConversationTree(ConversationTrees):
    def __init__(self, conv_id: str, title: str):
        self.id = conv_id
        self.title = title
        super().__init__()

    def bulk_add_messages(self, messages: list[MessageNode]):
        for message in messages:
            if message.metadata["conv_id"] == self.id:
                self.add_message(message)
                logger.trace(f"Added message {message.id} to conversation {self.id}")
            else:
                logger.error(f"Message {message.id} does not belong in conversation {self.id}")

    def visualize(
        self,
        tag_function: Callable[[str], str] | None = None,
    ):
        """Visualizes the conversation tree using Graphviz."""
        graph = self.plot_graph(conv_ids_table={self.id: self.title}, tag_function=tag_function)
        graph.render(
            f"conversation_{self.id}",
            outfile=f"output_graphs/conversation_tree_{self.title}.pdf",
            directory="output_graphs",
            cleanup=True,
            view=True,
        )

    def get_ordered_nodes(self) -> list[MessageNode]:
        """Returns a list of nodes ordered such that successors follow predecessors,
        and branches are ordered by shortest to longest."""

        visited = set()
        ordered_nodes = []

        def dfs(node: MessageNode):  # AI GENERATED FUNCTION
            """Depth-first search to order nodes."""
            if node in visited:
                return
            visited.add(node)
            if len(node.successors) > 1:
                successors = sorted(node.successors, key=get_branch_length)
            else:
                successors = list(node.successors)
            for successor in successors:
                dfs(successor)
            ordered_nodes.append(node)

        # Start DFS from nodes with no predecessors (roots)
        roots = [node for node in self.nodes.values() if not node.predecessors]
        if len(roots) > 1:
            logger.warning(f"Unexpected behavior : Multiple roots found for conversation {self.id}")
        for root in roots:
            dfs(root)

        return ordered_nodes[::-1]  # Reverse to ensure predecessors come before successors

    def get_embedding_distance_profile(
        self,
        distance: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y),
    ) -> tuple[list[int], list[float]]:
        """Computes the embedding distance profile for the conversation tree."""

        x = list(range(len(self.nodes) - 1))
        ordered_nodes = self.get_ordered_nodes()
        y = [distance(ordered_nodes[i].embeddings, ordered_nodes[i + 1].embeddings) for i in x]  # noqa: E501

        return x, y
