# reputation_normalisation.py

import numpy as np
from core.config import EPSILON

class ReputationNormalizer:
    def __init__(self, epsilon: float = EPSILON):
        """
        Initializes the ReputationNormalizer.

        Args:
            epsilon (float): A small constant to avoid division-by-zero.
        """
        self.epsilon = epsilon

    def normalize(self, reputation: np.ndarray) -> np.ndarray:
        """
        Normalizes the reputation vector so that all values are nonnegative and the vector sums to 1.
        
        The process is:
          1. Clip any negative values to zero.
          2. Compute the L₁ norm (sum) of the resulting vector.
          3. Divide the vector by its sum (plus epsilon for numerical stability).
        
        Args:
            reputation (np.ndarray): The input reputation vector.

        Returns:
            np.ndarray: The normalized reputation vector (with L₁ norm equal to 1).
        """
        # Clip negative values to 0.
        reputation_clipped = np.maximum(reputation, 0)
        # Compute the sum and normalize.
        total = np.sum(reputation_clipped) + self.epsilon
        normalized_reputation = reputation_clipped / total
        return normalized_reputation


# Example usage:
if __name__ == "__main__":
    # Example reputation vector (possibly coming from backward propagation)
    # Some entries might be negative due to penalty signals.
    reputation = np.array([0.1, -0.05, 0.3, 0.65], dtype=np.float32)
    normalizer = ReputationNormalizer()
    final_reputation = normalizer.normalize(reputation)
    
    print("Input Reputation Vector:")
    print(reputation)
    print("Final Normalized Reputation Vector:")
    print(final_reputation)
