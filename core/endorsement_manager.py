# endorsement_manager.py

import numpy as np
from typing import Dict, List

class EndorsementManager:
    def __init__(self, endorsement_matrix: np.ndarray, nodes: List, node_to_index: Dict):
        """
        Initializes the endorsement manager with a dense endorsement matrix,
        list of nodes, and node-to-index mapping.
        
        Each row of the endorsement_matrix represents the endorsements given by that node,
        and should be normalized so that the sum of the row equals 1 if there are any endorsements.
        
        Args:
            endorsement_matrix (np.ndarray): Dense endorsement matrix (shape: N x N).
            nodes (List): List of node identifiers.
            node_to_index (Dict): Dictionary mapping node identifier to index.
        """
        self.endorsement_matrix = endorsement_matrix.copy()
        self.nodes = nodes.copy()
        self.node_to_index = node_to_index.copy()
        self.normalize_all_rows()

    def normalize_all_rows(self) -> None:
        """
        Normalizes each row of the endorsement matrix so that the sum equals 1.
        If a row is entirely zeros (i.e. no endorsements), it remains unchanged.
        """
        for i in range(self.endorsement_matrix.shape[0]):
            row_sum = np.sum(self.endorsement_matrix[i, :])
            if row_sum > 0:
                self.endorsement_matrix[i, :] /= row_sum

    def add_endorsement(self, from_node: str, to_node: str) -> None:
        """
        Adds an endorsement from 'from_node' to 'to_node'. 
        
        If 'from_node' already endorses some nodes, all existing endorsement weights
        are scaled by n/(n+1) (where n is the current number of endorsements),
        and the new endorsement is assigned a weight of 1/(n+1). This maintains an L₁ norm of 1.
        
        If the endorsement already exists, no change is made.
        """
        if from_node not in self.node_to_index:
            print(f"Error: Node '{from_node}' not found in network.")
            return
        if to_node not in self.node_to_index:
            print(f"Error: Node '{to_node}' not found in network.")
            return
        
        i = self.node_to_index[from_node]
        j = self.node_to_index[to_node]

        # Check if endorsement already exists
        if self.endorsement_matrix[i, j] > 0:
            print(f"Endorsement from '{from_node}' to '{to_node}' already exists.")
            return

        # Count current endorsements given by from_node
        current_indices = np.nonzero(self.endorsement_matrix[i, :])[0]
        n = len(current_indices)

        # Scale existing endorsements if any exist
        if n > 0:
            scale_factor = n / (n + 1)
            self.endorsement_matrix[i, current_indices] *= scale_factor

        # Add the new endorsement with weight 1/(n+1)
        self.endorsement_matrix[i, j] = 1 / (n + 1)
        print(f"Added endorsement from '{from_node}' to '{to_node}'.")
        # (Row now sums to 1.)

    def remove_endorsement(self, from_node: str, to_node: str) -> None:
        """
        Removes the endorsement from 'from_node' to 'to_node' and re-normalizes the row.
        
        Let n be the number of endorsements in that row before removal.
        If n > 1, after removal the remaining endorsements are scaled by n/(n-1);
        if n equals 1, removal results in a row of zeros.
        """
        if from_node not in self.node_to_index:
            print(f"Error: Node '{from_node}' not found in network.")
            return
        if to_node not in self.node_to_index:
            print(f"Error: Node '{to_node}' not found in network.")
            return
        
        i = self.node_to_index[from_node]
        j = self.node_to_index[to_node]

        if self.endorsement_matrix[i, j] == 0:
            print(f"No endorsement from '{from_node}' to '{to_node}' exists.")
            return

        # Count the current endorsements in row i
        current_indices = np.nonzero(self.endorsement_matrix[i, :])[0]
        n = len(current_indices)

        # Remove the endorsement
        self.endorsement_matrix[i, j] = 0

        # Get the remaining endorsements in the row
        new_indices = np.nonzero(self.endorsement_matrix[i, :])[0]
        new_n = len(new_indices)

        if new_n > 0:
            # Scale the remaining endorsements so that the row L1 norm equals 1.
            # The scaling factor is n/(n-1)
            scale_factor = n / (n - 1)
            self.endorsement_matrix[i, new_indices] *= scale_factor
        print(f"Removed endorsement from '{from_node}' to '{to_node}' and re-normalized row.")

    def update_endorsements_with_penalty(self, penalty_signal: np.ndarray) -> None:
        """
        Updates the endorsement matrix using a penalty signal.
        
        For each endorser (each row in the matrix), the weights for each endorsement
        are multiplied by the corresponding penalty signal of the endorsee.
        
        That is, for each row i and for each column j:
            new_weight[i,j] = old_weight[i,j] * penalty_signal[j]
        
        Then, each row is re-normalized (if nonzero) so that its L1 norm is 1.
        
        Args:
            penalty_signal (np.ndarray): A 1D array of length N where each entry is
                                         the penalty signal g(N) for the corresponding node.
        """
        N = self.endorsement_matrix.shape[0]
        # For each row, update each endorsement weight
        for i in range(N):
            # Multiply row i element-wise with the penalty signal vector.
            self.endorsement_matrix[i, :] *= penalty_signal
            row_sum = np.sum(self.endorsement_matrix[i, :])
            if row_sum > 0:
                self.endorsement_matrix[i, :] /= row_sum
        print("Endorsements updated with penalty signal and re-normalized.")

    def update_endorsements_with_reward(self, reward_signal: np.ndarray) -> None:
        """
        Updates the endorsement matrix using a reward signal.
        
        For each endorser (each row), each endorsement weight is multiplied by the
        corresponding reward signal of the endorsee.
        
        That is, for each row i and for each column j:
            new_weight[i,j] = old_weight[i,j] * reward_signal[j]
        
        Then, each row is re-normalized (if nonzero) so that its L1 norm is 1.
        
        Args:
            reward_signal (np.ndarray): A 1D array of length N where each entry is
                                        the reward signal r(P) for the corresponding node.
        """
        N = self.endorsement_matrix.shape[0]
        for i in range(N):
            self.endorsement_matrix[i, :] *= reward_signal
            row_sum = np.sum(self.endorsement_matrix[i, :])
            if row_sum > 0:
                self.endorsement_matrix[i, :] /= row_sum
        print("Endorsements updated with reward signal and re-normalized.")

    def display_endorsements_for_node(self, node_id: str) -> None:
        """
        Displays the normalized endorsements given by 'node_id'.
        
        Prints endorsements in the format:
          from node --> to node: weight.
        """
        if node_id not in self.node_to_index:
            print(f"Error: Node '{node_id}' not found in network.")
            return
        
        i = self.node_to_index[node_id]
        row = self.endorsement_matrix[i, :]
        endorsements = [(self.nodes[j], row[j]) for j in range(len(row)) if row[j] > 0]
        if not endorsements:
            print(f"Node '{node_id}' has not endorsed any node.")
        else:
            print(f"Endorsements from node '{node_id}':")
            for to_node, weight in endorsements:
                print(f"  --> {to_node}: {weight:.4f}")

    def display_all_endorsements(self) -> None:
        """
        Displays the entire normalized endorsement matrix.
        """
        print("Normalized Endorsement Matrix:")
        print(self.endorsement_matrix)


# Example usage (for testing):
if __name__ == "__main__":
    # Dummy data for demonstration:
    # Assume four nodes: A, B, C, D.
    nodes = ["A", "B", "C", "D"]
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    N = len(nodes)
    # Start with an empty endorsement matrix (all zeros)
    endorsement_matrix = np.zeros((N, N), dtype=np.float32)
    
    # Create an instance of EndorsementManager with the empty matrix.
    em = EndorsementManager(endorsement_matrix, nodes, node_to_index)
    
    # Add some endorsements:
    em.add_endorsement("A", "B")
    em.add_endorsement("A", "C")
    em.display_endorsements_for_node("A")
    
    # Suppose we have a penalty signal for nodes, for example:
    # Let’s say the penalty signal is defined by g(N_j) = exp(-β * N_j).
    # For demonstration, assume beta=0.5 and some arbitrary negative feedback counts.
    neg_feedback_counts = np.array([2, 1, 3, 0], dtype=np.float32)  # dummy numbers for each node
    beta = 0.5
    penalty_signal = np.exp(-beta * neg_feedback_counts)
    print("Penalty signal:", penalty_signal)
    
    # Update endorsements with penalty signal.
    em.update_endorsements_with_penalty(penalty_signal)
    em.display_endorsements_for_node("A")
    
    # Similarly, suppose we have a reward signal for nodes, defined by:
    # r(P_j) = 1 - exp(-λ * P_j), with lambda a reward sensitivity coefficient.
    pos_feedback_counts = np.array([5, 2, 1, 0], dtype=np.float32)  # dummy numbers for each node
    lambda_param = 0.3
    reward_signal = 1 - np.exp(-lambda_param * pos_feedback_counts)
    print("Reward signal:", reward_signal)
    
    # Update endorsements with reward signal.
    em.update_endorsements_with_reward(reward_signal)
    em.display_endorsements_for_node("A")
    
    # Display full matrix
    em.display_all_endorsements()
