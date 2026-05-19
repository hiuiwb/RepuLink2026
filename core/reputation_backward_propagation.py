# reputation_backward_propagation.py

import numpy as np
from core.config import EPSILON
from typing import Optional

class ReputationBackwardPropagation:
    def __init__(self, epsilon: float = EPSILON):
        """
        Initializes the reputation backward propagation updater.
        
        Args:
            epsilon (float): A small constant for numerical stability.
        """
        self.epsilon = epsilon

    def apply_backward_propagation(self, 
                                   forward_reputation: np.ndarray, 
                                   penalty_vector: np.ndarray, 
                                   reward_vector: np.ndarray) -> np.ndarray:
        """
        Integrates the forward reputation with the backward signals by:
        
            R_updated = forward_reputation - penalty_vector + reward_vector
            R_updated_clipped = max(R_updated, 0)   [applied elementwise]
            R_final = R_updated_clipped normalized so that its L1 norm equals 1
        
        Args:
            forward_reputation (np.ndarray): Reputation vector from forward propagation.
            penalty_vector (np.ndarray): Endorsement penalty vector (π).
            reward_vector (np.ndarray): Endorsement reward vector (ρ).
            
        Returns:
            np.ndarray: The final updated reputation vector.
        """
        # Integrate the signals.
        R_updated = forward_reputation - penalty_vector + reward_vector
        
        # Ensure no negative values.
        R_updated = np.maximum(R_updated, 0)
        
        # Normalize the updated reputation vector (L1 norm = 1).
        total = np.sum(R_updated) + self.epsilon
        R_final = R_updated / total
        
        return R_final


# Example usage:
if __name__ == "__main__":
    # For demonstration, assume a network with 4 nodes.
    # Dummy forward reputation vector (e.g., from forward propagation).
    forward_reputation = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    # Dummy penalty vector (π) and reward vector (ρ),
    # for example, computed by backward propagation modules.
    penalty_vector = np.array([0.05, 0.10, 0.02, 0.00], dtype=np.float32)
    reward_vector  = np.array([0.02, 0.03, 0.00, 0.01], dtype=np.float32)

    # Create an instance of the backward propagation updater.
    rbp = ReputationBackwardPropagation()

    # Compute the final reputation vector after backward propagation.
    final_reputation = rbp.apply_backward_propagation(forward_reputation, penalty_vector, reward_vector)
    
    print("Forward Reputation Vector:")
    print(forward_reputation)
    print("Penalty Vector (π):")
    print(penalty_vector)
    print("Reward Vector (ρ):")
    print(reward_vector)
    print("Final Reputation Vector after Backward Propagation:")
    print(final_reputation)
