# endorsement_penalty.py

import numpy as np
from core.config import BETA, GAMMA, MAX_ITER, CONVERGENCE_THRESHOLD, EPSILON
from typing import Optional

class EndorsementPenalty:
    def __init__(self, endorsement_matrix: np.ndarray, negative_feedback: np.ndarray):
        """
        Initializes the endorsement penalty calculator.
        
        Args:
            endorsement_matrix (np.ndarray): Normalized endorsement matrix F (shape: N x N).
            negative_feedback (np.ndarray): A 1D array of length N with negative feedback counts for each node.
        """
        self.F = endorsement_matrix
        self.negative_feedback = negative_feedback.astype(np.float32)
        self.beta = BETA
        self.gamma = GAMMA
        self.max_iter = MAX_ITER
        self.tol = CONVERGENCE_THRESHOLD
        self.epsilon = EPSILON

    def compute_penalty_signal(self) -> np.ndarray:
        """
        Computes the base penalty signal vector g(N) where each element is:
            g(N_j) = exp(-beta * N_j)
        
        Returns:
            np.ndarray: Penalty signal vector g(N) of length N.
        """
        gN = np.exp(-self.beta * self.negative_feedback)
        return gN

    def compute_penalty(self) -> np.ndarray:
        """
        Computes the backward endorsement penalty vector π using power iteration,
        based on the formula: π = sum_{k=1 to K} (gamma^k * F^k * (1 - g(N)))
        
        The iterative process is as follows:
            1. Calculate base signal: gN = exp(-beta * N)
            2. Define propagated signal: s_penalty = (1 - gN)
            3. Initialize first term of sum: current_propagated_term = gamma * F @ s_penalty
            4. Initialize pi = current_propagated_term
            5. For k from 2 up to max_iter (or K in formula):
                 current_propagated_term = gamma * F @ (current_propagated_term / gamma)  
                                          # Effectively gamma^k * F^k * s_penalty by iterating F on previous F^{k-1} term
                                          # More directly: temp_signal_component = F @ previous_signal_component
                                          # current_propagated_term = gamma * temp_signal_component
                 current_propagated_term = self.gamma * np.dot(self.F, (current_propagated_term / self.gamma if self.gamma != 0 else current_propagated_term)) # to get F * (gamma^{k-1}F^{k-1}s)
                 # Simpler: iterate on the F*s part and multiply by gamma^k later, or iterate on gamma*F*s
                 # Let's stick to the sum of terms: term_k = (gamma*F)^k * s_penalty
                 # Or, as implemented before for sum gamma^k F^k S:
                 # term_1 = gamma * F @ S
                 # term_k = gamma * F @ term_{k-1}_unscaled_by_gamma
                 # Let temp be (gamma*F)^{k-1} * S. Then next term is gamma*F*temp.

        Returns:
            np.ndarray: The computed penalty vector π (length N).
        """
        gN = self.compute_penalty_signal()  # vector of length N
        signal_to_propagate = 1.0 - gN # This is (1 - g(N))

        # Check if the signal to propagate is effectively zero
        if np.linalg.norm(signal_to_propagate, 1) < self.tol: # Using tol as a general small number
            print("Penalty propagation: (1 - g(N)) signal is near zero. Resulting penalty is zero.")
            return np.zeros_like(signal_to_propagate)

        # Iterative calculation of sum_{k=1 to K} (gamma^k * F^k * signal_to_propagate)
        # Let term_k_base = F^k * signal_to_propagate
        # pi = gamma * term_1_base + gamma^2 * term_2_base + ...

        pi = np.zeros_like(signal_to_propagate)
        current_F_power_s = signal_to_propagate.copy() # This is F^0 * s_penalty

        for k in range(1, self.max_iter + 1): # k from 1 to max_iter
            current_F_power_s = np.dot(self.F, current_F_power_s) # F * (F^{k-1}*s) = F^k * s
            term_k = (self.gamma ** k) * current_F_power_s
            pi += term_k
            
            # Check convergence based on the magnitude of the added term
            if np.sum(np.abs(term_k)) < self.tol:
                print(f"Penalty propagation converged in {k} iterations.")
                return pi
        
        print(f"Penalty propagation did not fully converge within {self.max_iter} iterations.")
        return pi

# Example usage:
if __name__ == "__main__":
    # For demonstration, assume a network with 4 nodes.
    # Create a dummy normalized endorsement matrix F.
    F = np.array([
        [0.0, 0.5, 0.5, 0.0],
        [0.3, 0.0, 0.7, 0.0],
        [0.6, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    # Dummy negative feedback counts for each node (e.g., computed from interactions)
    neg_counts = np.array([2, 1, 3, 0], dtype=np.float32)
    
    # Initialize and compute the penalty.
    ep = EndorsementPenalty(F, neg_counts)
    penalty_vector = ep.compute_penalty()
    
    print("Penalty signal g(N):")
    print(ep.compute_penalty_signal())
    print("Computed Penalty Vector π:")
    print(penalty_vector)
