# reputation_forward_propagation.py

import numpy as np
from core.config import EPSILON, ALPHA, MAX_ITER, CONVERGENCE_THRESHOLD
from typing import Optional

class ReputationForwardPropagation:
    def __init__(self, trust_matrix: np.ndarray, endorsement_matrix: np.ndarray):
        """
        Initializes the reputation forward propagation calculator using the
        normalized trustworthiness matrix (C) and normalized endorsement matrix (F).

        The reputation is computed via power iteration according to:
            R(t+1) = ALPHA * (Cᵀ * R(t)) + (1 - ALPHA) * (Fᵀ * R(t))
        where the parameters are imported from config.py.

        Args:
            trust_matrix (np.ndarray): Normalized local trustworthiness matrix (C) of shape (N x N).
            endorsement_matrix (np.ndarray): Normalized endorsement matrix (F) of shape (N x N).
        """
        self.C = trust_matrix
        self.F = endorsement_matrix
        self.alpha = ALPHA
        self.max_iter = MAX_ITER
        self.tol = CONVERGENCE_THRESHOLD
        self.epsilon = EPSILON

    def compute_reputation(self, initial_reputation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the reputation vector using power iteration:
            R(t+1) = ALPHA * (Cᵀ * R(t)) + (1 - ALPHA) * (Fᵀ * R(t))
        The reputation vector is normalized (L1 norm equals 1) at every iteration.
        
        Args:
            initial_reputation (Optional[np.ndarray]): Optional starting reputation vector.
                If not provided, a uniform vector is used.
                
        Returns:
            np.ndarray: Final converged reputation vector.
        """
        N = self.C.shape[0]
        if initial_reputation is None:
            R = np.ones(N, dtype=np.float32) / N
        else:
            R = initial_reputation.copy()
            R = R / (np.sum(R) + self.epsilon)
        
        for iteration in range(self.max_iter):
            # Update using the forward propagation formula.
            R_new = self.alpha * np.dot(self.C.T, R) + (1 - self.alpha) * np.dot(self.F.T, R)
            # Normalize R_new so that its L1 norm is 1.
            norm = np.sum(R_new) + self.epsilon
            R_new = R_new / norm

            # Check convergence using L1 norm difference.
            if np.linalg.norm(R_new - R, 1) < self.tol:
                print(f"Converged in {iteration + 1} iterations.")
                return R_new
            R = R_new

        print("Did not converge within the maximum iterations.")
        return R

# Example usage:
if __name__ == "__main__":
    # Dummy normalized trust matrix (C) and endorsement matrix (F) for a network of 4 nodes.
    C = np.array([
        [0.0, 0.3, 0.7, 0.0],
        [0.2, 0.0, 0.8, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    F = np.array([
        [0.0, 0.6, 0.4, 0.0],
        [0.1, 0.0, 0.9, 0.0],
        [0.3, 0.7, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    # Create the ReputationForwardPropagation instance using config parameters.
    rep_forward = ReputationForwardPropagation(C, F)
    
    # Compute the reputation vector.
    reputation_vector = rep_forward.compute_reputation()
    print("Final Reputation Vector:")
    print(reputation_vector)
