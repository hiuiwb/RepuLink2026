#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script computes the EigenTrust score from interaction data (e.g., Bitcoin OTC)
based on the definition by Kamvar et al., WWW 2003.
It merges the score with endorsement-based in-degree from a second dataset (e.g., Epinions).

Refactored for modularity, timing, reporting, and corrected EigenTrust computation.
"""

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

# ============ Load Interaction Data & Build Trust Matrix C ============
def load_interaction_and_build_matrix_et(interaction_file, num_pretrusted=5, epsilon=1e-10):
    """
    Loads interaction data, calculates local trust s_ij = sat(i,j) - unsat(i,j),
    builds the normalized matrix C (c_ij = max(s_ij,0) / sum_k(max(s_ik,0))),
    defines pre-trusted peers and the 'p' vector, and handles zero-sum rows.
    Returns C, users, user2idx, N, and p_vec.
    """
    print(f"Loading interaction data from: {interaction_file}")
    df = pd.read_csv(interaction_file, header=None, names=["src", "dst", "rating", "timestamp"])
    print(f"Loaded {len(df)} interactions.")

    # Identify users and map to indices
    users = pd.concat([df['src'], df['dst']]).unique()
    users.sort()
    N = len(users)
    user2idx = { user: idx for idx, user in enumerate(users) }
    idx2user = { idx: user for user, idx in user2idx.items() } # For pre-trusted selection
    print(f"Found {N} unique users.")

    # Calculate local trust scores s_ij = sat(i,j) - unsat(i,j) [cite: 33]
    print("Calculating local trust matrix S (s_ij)...")
    # Use COO sparse format for potentially large datasets if memory becomes an issue
    S_ij = np.zeros((N, N))
    interaction_counts = {} # Store (sat, unsat) counts

    for _, row in df.iterrows():
        # Ensure src and dst are in our user list
        if row['src'] in user2idx and row['dst'] in user2idx:
            i, j = user2idx[row['src']], user2idx[row['dst']]
            rating = row['rating']
            
            pair = (i, j)
            if pair not in interaction_counts:
                interaction_counts[pair] = {'sat': 0, 'unsat': 0}

            # Define sat/unsat based on rating sign [cite: 30, 31]
            if rating > 0:
                interaction_counts[pair]['sat'] += 1
            elif rating < 0:
                interaction_counts[pair]['unsat'] += 1
            # Ignore rating == 0 for s_ij calculation

    for (i, j), counts in interaction_counts.items():
        S_ij[i, j] = counts['sat'] - counts['unsat']

    print(f"Calculated {len(interaction_counts)} non-zero s_ij values.")

    # Define pre-trusted peers and vector p [cite: 76]
    # Simple strategy: choose the first 'num_pretrusted' users alphabetically
    # More robust strategies could be used in practice.
    pretrusted_indices = list(range(min(num_pretrusted, N)))
    p_vec = np.zeros(N)
    if pretrusted_indices:
        p_vec[pretrusted_indices] = 1.0 / len(pretrusted_indices)
        print(f"Defined {len(pretrusted_indices)} pre-trusted peers with uniform distribution.")
        # print(f"Pre-trusted peers (IDs): {[idx2user[idx] for idx in pretrusted_indices]}")
    else:
        # Fallback to uniform distribution if no pre-trusted peers
        print("Warning: No pre-trusted peers defined. Using uniform vector for p.")
        p_vec = np.ones(N) / N

    # Normalize S_ij to get C_ij = max(s_ij, 0) / sum_k max(s_ik, 0) [cite: 50]
    # Handle rows where sum_k max(s_ik, 0) == 0 by setting C[i,:] = p [cite: 81, 82]
    print("Building row-normalized trust matrix C...")
    C = np.zeros((N, N))
    zero_sum_rows = 0
    for i in range(N):
        s_i_positive = np.maximum(S_ij[i, :], 0)
        sum_s_i_positive = np.sum(s_i_positive)

        if sum_s_i_positive > epsilon:
            C[i, :] = s_i_positive / sum_s_i_positive
        else:
            # If sum is zero, peer i trusts according to pre-trusted distribution p [cite: 81, 82]
            C[i, :] = p_vec # Assign the p_vec as the i-th row
            zero_sum_rows += 1

    print(f"C matrix built. {zero_sum_rows} users had zero positive local trust (assigned p_vec row).")

    # C is the matrix used in the EigenTrust paper's Algorithm 2: t = (1-a)C^T*t + a*p [cite: 85]
    return C, users, user2idx, N, p_vec

# ============ Load Endorsement Data & Normalize ============
def load_and_normalize_endorsements_et(endorsement_file, users, user2idx, N):
    """
    Loads endorsement data (Epinions format), counts in-degree for known users,
    and returns a min-max normalized endorsement vector aligned with 'users'.
    (Identical to previous version)
    """
    print(f"Loading endorsement data from: {endorsement_file}")
    try:
        df_endorse = pd.read_csv(endorsement_file, sep='\t', header=None, names=['src', 'dst'])
        print(f"Loaded {len(df_endorse)} endorsement links.")
    except FileNotFoundError:
        print(f"Error: Endorsement file not found at {endorsement_file}")
        print("Returning zero vector for endorsements.")
        return np.zeros(N)
    except Exception as e:
        print(f"Error loading endorsement file: {e}")
        print("Returning zero vector for endorsements.")
        return np.zeros(N)

    endorse_counts = df_endorse["dst"].value_counts()
    endorse_vec = np.zeros(N)
    found_endorsements = 0
    for user, count in endorse_counts.items():
        if user in user2idx:
            endorse_vec[user2idx[user]] = count
            found_endorsements += 1
    print(f"Found endorsements for {found_endorsements} nodes present in the interaction data.")

    min_val, max_val = np.min(endorse_vec), np.max(endorse_vec)
    range_val = max_val - min_val
    endorse_norm = (endorse_vec - min_val) / range_val if range_val > 1e-9 else np.zeros(N)
    print("Endorsement counts normalized.")
    return endorse_norm

# ============ Compute EigenTrust Iteratively (Corrected) ============
def compute_eigentrust_iterative(C, N, p_vec, damping_factor_a=0.15, tol=1e-6, max_iter=100):
    """
    Performs the iterative EigenTrust computation based on Algorithm 2[cite: 85].
    Uses t(k+1) = (1-a)*C.T*t(k) + a*p.
    Returns the final trust vector, list of residuals, and number of iterations.
    """
    print("Starting Corrected EigenTrust iterative computation...")
    print(f"Parameters: a={damping_factor_a:.2f}, max_iter={max_iter}, tol={tol:.1e}")

    # Initialize trust vector with pre-trusted distribution p [cite: 78, 85]
    t = p_vec.copy()
    residuals = []
    num_iterations = 0
    converged = False
    C_transpose = C.T # Pre-compute transpose

    for iteration in range(max_iter):
        num_iterations = iteration + 1
        t_prev = t.copy()

        # EigenTrust Update Rule (Algorithm 2) [cite: 86]
        # t = (1-a) * C.T @ t_prev + a * p_vec
        t = (1 - damping_factor_a) * (C_transpose @ t_prev) + (damping_factor_a * p_vec)

        # *** No explicit normalization needed here ***
        # The formula inherently keeps the sum approx 1 if t_prev sums to 1 and p_vec sums to 1.
        # Optional check: print(f"Sum t: {np.sum(t)}")

        # Calculate residual (L1 norm of the change)
        residual = np.linalg.norm(t - t_prev, ord=1)
        residuals.append(residual)

        # Print progress occasionally
        if num_iterations % 20 == 0 or num_iterations == 1:
            print(f"Iteration {num_iterations}/{max_iter}, Residual (L1 Change): {residual:.4e}")

        # Check for convergence
        if residual < tol:
            print(f"Converged after {num_iterations} iterations (L1 Change < {tol:.1e}).")
            converged = True
            break

    if not converged:
        print(f"Reached maximum iterations ({max_iter}) without converging. Final L1 Change: {residual:.4e}")

    print("Corrected EigenTrust iterative computation finished.")
    # Return the last computed trust vector 't', the list of residuals, and iteration count
    return t, residuals, num_iterations

# ============ Compute Hybrid Score ============
def compute_hybrid_score_et(et_scores, endorse_norm, lambda_weight):
    """Combines EigenTrust scores and normalized endorsement scores. (Unchanged)"""
    print(f"Computing hybrid score with lambda = {lambda_weight:.2f}")
    et_scores = np.asarray(et_scores)
    endorse_norm = np.asarray(endorse_norm)
    hybrid_scores = lambda_weight * et_scores + (1 - lambda_weight) * endorse_norm
    return hybrid_scores

# ============ Save Final Scores ============
def save_scores_et(nodes, scores, output_file, score_column_name="score"):
    """Saves the final scores to a CSV file. (Unchanged)"""
    df_out = pd.DataFrame({
        "dst": nodes,
        score_column_name: scores
    })
    df_out.sort_values(by=score_column_name, ascending=False, inplace=True)
    try:
        df_out.to_csv(output_file, index=False)
        print(f"Scores saved to {output_file}.")
    except Exception as e:
        print(f"Error saving scores to {output_file}: {e}")

# ============ Plot Residual Curve (Generic) ============
def plot_residual_curve_et(residuals, output_fig, algorithm_name="EigenTrust", convergence_tol=1e-6):
    """Plots the convergence curve (residuals vs. iteration). (Unchanged)"""
    if not residuals:
        print("No residuals to plot.")
        return
    print(f"Plotting residual curve to {output_fig}...")
    plt.figure(figsize=(8, 6))
    label_font = FontProperties(weight='bold', size=18)
    title_font = FontProperties(weight='bold', size=20)
    tick_font = FontProperties(weight='normal', size=14)
    legend_font = FontProperties(weight='bold', size=14)
    plt.plot(range(1, len(residuals) + 1), residuals, linestyle='-', linewidth=2, label='L1 Residual (Change)')
    if convergence_tol:
        plt.axhline(y=convergence_tol, color='red', linestyle='--', linewidth=1.5, label=f'Tolerance ({convergence_tol:.1e})')
    plt.yscale('log')
    plt.title(f"{algorithm_name} Convergence Curve", fontproperties=title_font)
    plt.xlabel("Iteration", fontproperties=label_font)
    plt.ylabel("Residual (L1 norm)", fontproperties=label_font)
    ax = plt.gca()
    for label in ax.get_xticklabels(): label.set_fontproperties(tick_font)
    for label in ax.get_yticklabels(): label.set_fontproperties(tick_font)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
    plt.grid(True, which="both", ls=":", linewidth=0.5)
    plt.legend(loc="upper right", prop=legend_font)
    plt.tight_layout()
    try:
        plt.savefig(output_fig, dpi=300, bbox_inches='tight')
        print(f"Residual curve saved to {output_fig}")
    except Exception as e:
        print(f"Error saving plot to {output_fig}: {e}")
    plt.close()

# ============ Main Function ============
def main():
    # --- Configuration ---
    interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_otc.csv"
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
    output_dir = "et_output_corrected" # Use a new directory
    output_file_hybrid = os.path.join(output_dir, "eigentrust_hybrid_corrected.csv")
    output_file_et_only = os.path.join(output_dir, "eigentrust_only_corrected.csv")
    residual_plot_file = os.path.join(output_dir, "eigentrust_residuals_corrected.pdf")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in: {os.path.abspath(output_dir)}")

    # Algorithm Parameters
    lambda_weight = 0.8          # Weight for hybrid score
    num_pretrusted = 5           # Number of pre-trusted peers (e.g., first N alphabetically) [cite: 76]
    damping_factor_a = 0.15      # Damping factor 'a' for pre-trusted peers [cite: 86] (Common PageRank value)
    max_iterations = 100         # Max iterations for EigenTrust (adjust as needed, paper shows <10 is often enough [cite: 108])
    convergence_tolerance = 1e-6 # Convergence threshold for EigenTrust

    # --- Data Loading and Matrix Building ---
    C_matrix, users, user2idx, N, p_vec = load_interaction_and_build_matrix_et(
        interaction_file, num_pretrusted=num_pretrusted
    )
    endorsement_norm_vec = load_and_normalize_endorsements_et(endorsement_file, users, user2idx, N)

    # --- Computation and Timing ---
    print("\nStarting Corrected EigenTrust computation and timing...")
    start_time = time.time() # <-- Start timer

    et_scores_vec, residuals, iterations_run = compute_eigentrust_iterative(
        C_matrix, N, p_vec,
        damping_factor_a=damping_factor_a,
        tol=convergence_tolerance,
        max_iter=max_iterations
    )

    end_time = time.time()   # <-- Stop timer
    runtime_s = end_time - start_time # <-- Calculate runtime

    # --- Hybrid Score Calculation ---
    hybrid_scores_vec = compute_hybrid_score_et(et_scores_vec, endorsement_norm_vec, lambda_weight)

    # --- Results ---
    final_residual = residuals[-1] if residuals else None

    # --- Terminal Output ---
    print("\n" + "="*40)
    print("    Corrected EigenTrust Computation Summary")
    print("="*40)
    print(f"Iterations Run:         {iterations_run}")
    if final_residual is not None:
        print(f"Final Residual (L1):    {final_residual:.6e}")
    else:
         print("Final Residual (L1):    N/A")
    print(f"Computation Runtime (s):{runtime_s:.4f}")
    print(f"Hybrid Lambda:          {lambda_weight:.2f}")
    print(f"Damping Factor (a):     {damping_factor_a:.2f}")
    print(f"# Pre-trusted Peers:    {num_pretrusted}")
    print("="*40 + "\n")

    # --- Saving Results ---
    save_scores_et(users, hybrid_scores_vec, output_file_hybrid, score_column_name="eigentrust_hybrid_score")
    save_scores_et(users, et_scores_vec, output_file_et_only, score_column_name="eigentrust_score")

    # --- Plotting Residuals ---
    plot_residual_curve_et(residuals, residual_plot_file, algorithm_name="EigenTrust (Corrected)", convergence_tol=convergence_tolerance)

    print("Corrected EigenTrust script finished successfully.")

# --- Execution Guard ---
if __name__ == "__main__":
    main()