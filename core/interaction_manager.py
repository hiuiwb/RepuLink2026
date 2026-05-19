# interaction_manager.py

import pandas as pd
import numpy as np
from typing import Dict, List

class InteractionManager:
    def __init__(self, interactions: pd.DataFrame, nodes: List, node_to_index: Dict):
        """
        Initializes the InteractionManager with the initial interactions DataFrame,
        list of nodes, and node-to-index mapping.
        
        The interactions DataFrame is expected to have columns:
        ["src", "dst", "rating", "timestamp", "month"].
        
        Additionally, two dense matrices (pos_counts and neg_counts) are computed
        to maintain the count of positive and negative interactions between nodes.
        """
        self.interactions = interactions.copy()
        self.nodes = nodes
        self.node_to_index = node_to_index

        # Create dense matrices for positive and negative counts
        N = len(nodes)
        self.pos_counts = np.zeros((N, N), dtype=np.int32)
        self.neg_counts = np.zeros((N, N), dtype=np.int32)
        self._initialize_counts()

    def _initialize_counts(self) -> None:
        """Initializes the pos_counts and neg_counts matrices based on the existing interactions."""
        for _, row in self.interactions.iterrows():
            src = row["src"]
            dst = row["dst"]
            rating = row["rating"]
            if src in self.node_to_index and dst in self.node_to_index:
                i = self.node_to_index[src]
                j = self.node_to_index[dst]
                if rating >= 1:
                    self.pos_counts[i, j] += 1
                elif rating <= -1:
                    self.neg_counts[i, j] += 1

    def add_interaction(self, src: str, dst: str, rating: int, timestamp: float, month: int) -> None:
        """
        Adds a new interaction record to the history and updates the corresponding
        positive or negative feedback count.
        """
        if src not in self.nodes or dst not in self.nodes:
            print(f"Warning: One of the nodes '{src}' or '{dst}' is not in the network.")
        
        new_record = {
            "src": src,
            "dst": dst,
            "rating": rating,
            "timestamp": timestamp,
            "month": month
        }
        # Append new record using pd.concat (since DataFrame.append is deprecated)
        new_row = pd.DataFrame([new_record])
        self.interactions = pd.concat([self.interactions, new_row], ignore_index=True)
        print(f"Added interaction: {new_record}")

        # Update feedback counts
        if src in self.node_to_index and dst in self.node_to_index:
            i = self.node_to_index[src]
            j = self.node_to_index[dst]
            if rating >= 1:
                self.pos_counts[i, j] += 1
            elif rating <= -1:
                self.neg_counts[i, j] += 1

    def display_interactions_for_node(self, node_id: str) -> None:
        """Displays all interaction records for which the given node is either source or destination."""
        if node_id not in self.nodes:
            print(f"Error: Node '{node_id}' is not in the network.")
            return
        
        filtered = self.interactions[(self.interactions["src"] == node_id) |
                                     (self.interactions["dst"] == node_id)]
        if filtered.empty:
            print(f"No interactions found for node '{node_id}'.")
        else:
            print(f"Interactions for node '{node_id}':")
            print(filtered.to_string(index=False))

    def display_all_interactions(self) -> None:
        """Displays all recorded interactions."""
        if self.interactions.empty:
            print("No interactions recorded.")
        else:
            print("All interactions:")
            print(self.interactions.to_string(index=False))

    def display_positive_counts(self) -> None:
        """Displays the positive feedback counts matrix."""
        print("Positive Feedback Counts:")
        print(self.pos_counts)

    def display_negative_counts(self) -> None:
        """Displays the negative feedback counts matrix."""
        print("Negative Feedback Counts:")
        print(self.neg_counts)

    def resize(self, new_nodes: List, new_node_to_index: Dict) -> None:
        """
        Resizes the interaction count matrices and updates node info to include new nodes.

        Args:
            new_nodes (List): The updated list of nodes.
            new_node_to_index (Dict): The updated node-to-index mapping.
        """
        old_N = len(self.nodes)
        new_N = len(new_nodes)
        # Create new matrices with the new dimensions
        new_pos_counts = np.zeros((new_N, new_N), dtype=np.int32)
        new_neg_counts = np.zeros((new_N, new_N), dtype=np.int32)

        # Copy over the old counts to the corresponding positions in the new matrices.
        # We assume that the new mapping contains all old nodes with the same ordering for the old subset.
        for node in self.nodes:
            if node in new_node_to_index and node in self.node_to_index:
                old_i = self.node_to_index[node]
                new_i = new_node_to_index[node]
                # Copy the entire row and column for this node.
                new_pos_counts[new_i, :old_N] = self.pos_counts[old_i, :]
                new_pos_counts[:old_N, new_i] = self.pos_counts[:, old_i]
                new_neg_counts[new_i, :old_N] = self.neg_counts[old_i, :]
                new_neg_counts[:old_N, new_i] = self.neg_counts[:, old_i]

        # Update instance variables
        self.pos_counts = new_pos_counts
        self.neg_counts = new_neg_counts
        self.nodes = new_nodes
        self.node_to_index = new_node_to_index
        print("Interaction matrices resized to include new nodes.")

# Example usage for dynamic resizing:
if __name__ == "__main__":
    # Initial dummy data
    data = {
        "src": ["A", "B"],
        "dst": ["B", "C"],
        "rating": [1, -1],
        "timestamp": [1609459200, 1609545600],
        "month": [1, 1]
    }
    interactions_df = pd.DataFrame(data)
    nodes = ["A", "B", "C"]
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    im = InteractionManager(interactions_df, nodes, node_to_index)
    im.display_all_interactions()
    im.display_positive_counts()
    im.display_negative_counts()

    # Add a new interaction
    im.add_interaction("A", "C", 1, 1609632000, 1)

    # Now suppose we add a new node "D"
    new_nodes = ["A", "B", "C", "D"]  # New list including D
    new_node_to_index = {node: idx for idx, node in enumerate(new_nodes)}
    # Call resize to update the matrices
    im.resize(new_nodes, new_node_to_index)
    
    # After resizing, new interactions involving "D" can be added
    im.add_interaction("D", "A", -1, 1609718400, 1)
    im.display_all_interactions()
    im.display_positive_counts()
    im.display_negative_counts()
