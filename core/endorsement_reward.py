# endorsement_reward.py

import numpy as np
from core.config import EPSILON, GAMMA, MAX_ITER, CONVERGENCE_THRESHOLD, LAMBDA
from typing import Optional

class EndorsementReward:
    def __init__(self, endorsement_matrix: np.ndarray, positive_feedback: np.ndarray):
        """
        Initializes the endorsement reward calculator.
        
        Args:
            endorsement_matrix (np.ndarray): Normalized endorsement matrix F of shape (N x N).
            positive_feedback (np.ndarray): A 1D array of length N containing positive feedback counts for each node.
        """
        self.F = endorsement_matrix
        self.positive_feedback = positive_feedback.astype(np.float32)
        self.lambda_param = LAMBDA
        self.gamma = GAMMA
        self.max_iter = MAX_ITER
        self.tol = CONVERGENCE_THRESHOLD
        self.epsilon = EPSILON

    def compute_reward_signal(self) -> np.ndarray:
        """
        Computes the reward signal vector r(P) where each element is defined as:
            r(P_j) = 1 - exp(-lambda * P_j)
        
        Returns:
            np.ndarray: Reward signal vector of length N.
        """
        rP = 2 - np.exp(-self.lambda_param * self.positive_feedback)
        return rP

    # def compute_reward(self) -> np.ndarray:
    #     """
    #     Computes the backward endorsement reward vector ρ using an iterative propagation process.
        
    #     The iterative process is defined as:
    #       - Initialize: ρ₁ = gamma * F * r(P)
    #       - For k = 2 to max_iter:
    #             temp = gamma * F * temp
    #             ρ += temp
    #             if ||temp||₁ < tol, break.
        
    #     Returns:
    #         np.ndarray: The computed reward vector ρ (of length N).
    #     """
    #     r = self.compute_reward_signal()  # 1D array of length N
    #     rho = self.gamma * np.dot(self.F, r)  # Initial reward vector
    #     temp = rho.copy()

    #     for iteration in range(1, self.max_iter):
    #         temp = self.gamma * np.dot(self.F, temp)
    #         rho += temp
    #         if np.sum(np.abs(temp)) < self.tol:
    #             print(f"Reward propagation converged in {iteration + 1} iterations.")
    #             return rho
    #     print("Reward propagation did not fully converge within maximum iterations.")
    #     return rho

    def compute_reward(self) -> np.ndarray:
        """
        Computes the backward endorsement reward vector ρ using power iteration,
        based on the formula: ρ = sum_{k=1 to K} (gamma^k * F^k * (r(P) - 1))
        
        The iterative process is as follows:
            1. Calculate base signal: rP = 1 - exp(-lambda * P)
            2. Define propagated signal: s_reward = (rP - 1.0)
            3. Iteratively sum terms: term_k = (gamma^k * F^k * s_reward)

        Returns:
            np.ndarray: The computed reward vector ρ (of length N).
        """
        rP = self.compute_reward_signal()  # vector of length N
        signal_to_propagate = rP - 1.0 # This is (r(P) - 1)

        # Check if the signal to propagate is effectively zero
        # (r(P) is between 0 and 1, so r(P)-1 is between -1 and 0)
        if np.linalg.norm(signal_to_propagate, 1) < self.tol: # Using tol as a general small number
            print("Reward propagation: (r(P) - 1) signal is near zero. Resulting reward is zero.")
            return np.zeros_like(signal_to_propagate)

        # Iterative calculation of sum_{k=1 to K} (gamma^k * F^k * signal_to_propagate)
        rho = np.zeros_like(signal_to_propagate)
        # current_F_power_s will store F^(k-1) * signal_to_propagate in iteration k
        # or more precisely, it stores F_power_prev_s which is F^(k-1) * signal_to_propagate
        # and then current_F_power_s becomes F^k * signal_to_propagate
        
        # Let current_F_power_s be F^{k-1} * signal_to_propagate
        # Initialize for k=1: F^0 * signal_to_propagate = signal_to_propagate
        f_k_minus_1_s = signal_to_propagate.copy() 

        for k in range(1, self.max_iter + 1): # k from 1 to max_iter
            # For term k: we need F^k * signal_to_propagate
            # current_F_power_s = F @ (F^{k-1} * signal_to_propagate)
            current_F_power_s = np.dot(self.F, f_k_minus_1_s) # This is F * (F^{k-1}*s) = F^k * s
            
            term_k = (self.gamma ** k) * current_F_power_s
            rho += term_k
            
            # Update for next iteration: F^{k}*s becomes the new F^{k-1}*s for the next step
            f_k_minus_1_s = current_F_power_s 

            # Check convergence based on the magnitude of the added term
            if np.sum(np.abs(term_k)) < self.tol:
                print(f"Reward propagation converged in {k} iterations.")
                return rho
        
        print(f"Reward propagation did not fully converge within {self.max_iter} iterations.")
        return rho


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
    
    # Dummy positive feedback counts for each node (e.g., computed from interactions).
    pos_counts = np.array([5, 2, 1, 0], dtype=np.float32)
    
    # Initialize and compute the reward vector.
    er = EndorsementReward(F, pos_counts)
    reward_signal = er.compute_reward_signal()
    reward_vector = er.compute_reward()
    
    print("Reward signal r(P):")
    print(reward_signal)
    print("Computed Reward Vector ρ:")
    print(reward_vector)
