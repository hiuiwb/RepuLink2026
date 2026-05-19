# config.py
EPSILON = 1e-6          # To avoid division by zero
ALPHA = 0.5             # Weight between local trust and endorsement contribution
BETA = 0.1              # Parameter for the penalty function
GAMMA = 0.05            # Penalty coefficient for endorsers
MAX_ITER = 1000          # Maximum iterations for convergence
CONVERGENCE_THRESHOLD = 1e-5  # Convergence threshold for the reputation vector
LAMBDA = 0.3            # Define a reward sensitivity coefficient λ.