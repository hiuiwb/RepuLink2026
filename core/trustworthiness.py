# trustworthiness.py

import numpy as np
from typing import Optional

class TrustworthinessCalculator:
    def __init__(self, pos_matrix: np.ndarray, neg_matrix: np.ndarray, epsilon: float = 1e-6):
        """
        Initializes the TrustworthinessCalculator.

        Args:
            pos_matrix (np.ndarray): Dense positive feedback count matrix.
            neg_matrix (np.ndarray): Dense negative feedback count matrix.
            epsilon (float): Small constant added to denominators for numerical stability.
        """
        self.pos_matrix = pos_matrix
        self.neg_matrix = neg_matrix
        self.epsilon = epsilon

    def compute_raw_trust(self) -> np.ndarray:
        """
        Computes the raw trust value matrix T where each element is defined by:
            T_ij = (p_ij - n_ij) / (p_ij + n_ij + epsilon)

        Returns:
            np.ndarray: Raw trust value matrix (same shape as pos_matrix/neg_matrix).
        """
        T = (self.pos_matrix - self.neg_matrix) / (self.pos_matrix + self.neg_matrix + self.epsilon)
        return T

    def normalize_trust(self, T: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalizes the trust matrix locally. For each row i (representing the trust scores given by node i),
        it applies:
        
            C_ij = max(T_ij, 0) / (sum_j max(T_ij, 0) + epsilon)
        
        If no trust (all non-positive) is present in the row, the row is set to zeros.

        Args:
            T (np.ndarray, optional): Trust matrix to be normalized. If None, it is computed via compute_raw_trust().

        Returns:
            np.ndarray: Normalized local trust matrix C.
        """
        if T is None:
            T = self.compute_raw_trust()
        
        N, M = T.shape
        C = np.zeros_like(T)
        # Normalize each row
        for i in range(N):
            # Only consider positive values in each row (clamp negative values to zero)
            row_positive = np.maximum(T[i, :], 0)
            row_sum = np.sum(row_positive)
            if row_sum > 0:
                C[i, :] = row_positive / (row_sum + self.epsilon)
            else:
                C[i, :] = 0  # No positive trust, all remain zero.
        return C

    def compute_local_trust(self) -> np.ndarray:
        """
        Computes the normalized local trust matrix by first computing the raw trust values
        and then performing local normalization.

        Returns:
            np.ndarray: The normalized local trust matrix C.
        """
        T = self.compute_raw_trust()
        C = self.normalize_trust(T)
        return C


# Example usage:
if __name__ == "__main__":
    # Example dummy data for demonstration.
    # Suppose we have a network of 4 nodes, so matrices are 4x4.
    # Let's create positive and negative feedback matrices:
    
    # Positive feedback counts (each element: how many positive interactions from i to j)
    pos = np.array([
        [0, 3, 1, 0],
        [2, 0, 0, 1],
        [0, 4, 0, 2],
        [1, 0, 3, 0]
    ], dtype=np.float32)
    
    # Negative feedback counts (each element: how many negative interactions from i to j)
    neg = np.array([
        [0, 1, 0, 0],
        [0, 0, 2, 0],
        [1, 0, 0, 0],
        [0, 2, 0, 0]
    ], dtype=np.float32)
    
    # Initialize the calculator with these matrices.
    calculator = TrustworthinessCalculator(pos, neg, epsilon=1e-6)
    
    # Compute raw trust values.
    T = calculator.compute_raw_trust()
    print("Raw Trust Matrix T:")
    print(T)
    
    # Compute the normalized local trust matrix.
    C = calculator.compute_local_trust()
    print("Normalized Local Trust Matrix C:")
    print(C)
