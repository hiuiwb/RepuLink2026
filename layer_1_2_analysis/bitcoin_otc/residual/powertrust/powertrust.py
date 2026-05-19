#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid PowerTrust implementation combining:
- Iterative PowerTrust scores (leveraging selected power nodes based on Bitcoin OTC stats)
- Normalized endorsement in-degree from Epinions

Refactored for modularity, timing, residual tracking/plotting,
and implementing the iterative PowerTrust algorithm based on the paper.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp # <-- Import SciPy sparse
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

# ============ Load Interaction Data & Build Trust Matrix R ============
def load_interaction_and_build_matrix_pt(interaction_file, epsilon=1e-10):
    """
    Loads interaction data, calculates normalized local trust matrix R,
    where R_ij is the normalized positive rating from i to j.
    Returns R, sorted list of users, user-to-index mapping, and N.
    """
    print(f"Loading interaction data from: {interaction_file}")
    try:
        df = pd.read_csv(interaction_file, header=None, names=["src", "dst", "rating", "timestamp"])
        print(f"Loaded {len(df)} interactions.")
    except FileNotFoundError:
         print(f"Error: Interaction file not found at {interaction_file}")
         return None, None, None, 0
    except Exception as e:
        print(f"Error loading interaction file: {e}")
        return None, None, None, 0

    # Identify all unique users involved
    users = pd.unique(df[["src", "dst"]].values.ravel())
    users.sort()
    N = len(users)
    user2idx = { user: idx for idx, user in enumerate(users) }
    print(f"Found {N} unique users.")

    # Calculate average positive ratings for S_ij matrix
    # Only consider positive ratings for calculating local trust scores
    print("Calculating local trust matrix S (using average positive ratings)...")
    df_pos = df[df['rating'] > 0].copy()
    # Group by source and destination, calculate mean positive rating
    s_ij_df = df_pos.groupby(['src', 'dst'])['rating'].mean().reset_index()

    # Build sparse S matrix (using avg positive rating)
    row_indices = [user2idx[s] for s in s_ij_df['src'] if s in user2idx]
    col_indices = [user2idx[d] for d in s_ij_df['dst'] if d in user2idx]
    # Filter data to match valid indices
    valid_mask = [s in user2idx and d in user2idx for s, d in zip(s_ij_df['src'], s_ij_df['dst'])]
    data = s_ij_df['rating'][valid_mask].values

    S_ij_sparse = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    print(f"Built S matrix with {S_ij_sparse.nnz} non-zero avg positive ratings.")

    # Normalize S_ij rows to get R_ij (stochastic matrix)
    # R_ij = S_ij / sum_k(S_ik)
    print("Building row-normalized trust matrix R...")
    row_sums = np.array(S_ij_sparse.sum(axis=1)).flatten()
    non_zero_rows_mask = row_sums > epsilon
    # Prepare diagonal matrix for division (handling zero rows)
    inv_row_sums = np.zeros_like(row_sums)
    inv_row_sums[non_zero_rows_mask] = 1.0 / row_sums[non_zero_rows_mask]
    inv_row_sums_diag = sp.diags(inv_row_sums, format='csr')

    R = inv_row_sums_diag @ S_ij_sparse
    zero_sum_rows = N - np.sum(non_zero_rows_mask)
    print(f"R matrix built. {zero_sum_rows} users had zero positive ratings outgoing (zero rows in R).")
    # Note: The iterative algorithm needs to handle how rank flows *to* nodes
    # connected from these zero-row nodes if necessary, but PowerTrust's formula
    # V = (1-a)R^T V + a P_power handles this implicitly via R^T and the power node term.

    return R, users, user2idx, N

# ============ Identify Power Nodes ============
def identify_power_nodes_pt(interaction_file, users, user2idx, N,
                            power_ratio=0.05, min_feedback=10):
    """
    Identifies power nodes based on interaction stats (high avg rating, min count).
    Returns a set of power node indices and the number of power nodes (m).
    This is a simplification of the paper's distributed ranking.
    """
    print("Identifying Power Nodes (simplified method)...")
    try:
        df = pd.read_csv(interaction_file, header=None, names=["src", "dst", "rating", "timestamp"])
    except Exception as e:
         print(f"Error reading interaction file for power node identification: {e}")
         return set(), 0

    # Calculate feedback stats (mean rating, count) for each source node
    feedback_stats = df.groupby("src")["rating"].agg(["mean", "count"]).reset_index()
    # Filter by minimum feedback count
    feedback_stats = feedback_stats[feedback_stats["count"] >= min_feedback]
    # Sort by mean rating (descending) to find most reputable sources
    feedback_stats = feedback_stats.sort_values(by="mean", ascending=False)

    # Select top 'power_ratio' % as power nodes
    num_potential_power = len(feedback_stats)
    num_power = int(num_potential_power * power_ratio)
    # Ensure at least one power node if possible and ratio > 0
    if num_power == 0 and power_ratio > 0 and num_potential_power > 0:
        num_power = 1

    # Get the IDs of the power nodes
    power_node_ids = set(feedback_stats["src"].iloc[:num_power])

    # Convert power node IDs to indices
    power_node_indices = {user2idx[nid] for nid in power_node_ids if nid in user2idx}
    m = len(power_node_indices)
    print(f"Selected {m} power nodes based on rating stats (ratio={power_ratio}, min_feedback={min_feedback}).")

    return power_node_indices, m


# ============ Load Endorsement Data & Normalize ============
def load_and_normalize_endorsements_pt(endorsement_file, users, user2idx, N):
    """
    Loads endorsement data, counts in-degree, normalizes.
    (Identical to previous versions)
    """
    print(f"Loading endorsement data from: {endorsement_file}")
    try:
        df_endorse = pd.read_csv(endorsement_file, sep='\t', header=None, names=['src', 'dst'])
        print(f"Loaded {len(df_endorse)} endorsement links.")
    except FileNotFoundError:
        print(f"Error: Endorsement file not found at {endorsement_file}. Returning zeros.")
        return np.zeros(N)
    except Exception as e:
        print(f"Error loading endorsement file: {e}. Returning zeros.")
        return np.zeros(N)

    endorse_counts = df_endorse["dst"].value_counts()
    endorse_vec = np.zeros(N)
    found_endorsements = 0
    for user, count in endorse_counts.items():
        if user in user2idx:
            endorse_vec[user2idx[user]] = count
            found_endorsements += 1
    print(f"Found endorsements for {found_endorsements} nodes present in interaction data.")

    min_val, max_val = np.min(endorse_vec), np.max(endorse_vec)
    range_val = max_val - min_val
    endorse_norm = (endorse_vec - min_val) / (range_val + 1e-9) if range_val > -1e9 else np.zeros(N)
    print("Endorsement counts normalized.")
    return endorse_norm


# ============ Compute PowerTrust Iteratively ============
def compute_powertrust_iterative(R, N, power_node_indices, m, greedy_factor_a=0.15,
                                max_iter=100, tol=1e-6):
    """
    Performs the iterative PowerTrust computation based on Algorithm 3 from the paper.
    v_k = (1-a) * sum(v_j * r_jk) + (a/m if k is power node else 0)
    Tracks residuals.

    Args:
        R (scipy.sparse matrix): Row-normalized trust matrix (R_ij = trust from i to j).
        N (int): Number of nodes.
        power_node_indices (set): Set containing indices of power nodes.
        m (int): Number of power nodes.
        greedy_factor_a (float): Damping factor (alpha in the paper).
        max_iter (int): Maximum iterations.
        tol (float): Convergence tolerance (L1 norm change).

    Returns:
        tuple: (trust_vector, residuals_list, num_iterations)
    """
    print("Starting Iterative PowerTrust computation (Algorithm 3)...")
    print(f"Parameters: a={greedy_factor_a:.2f}, m={m}, max_iter={max_iter}, tol={tol:.1e}")

    if N == 0:
        return np.array([]), [], 0

    v = np.ones(N) / N # Initial trust vector (uniform)
    residuals = []
    num_iterations = 0
    converged = False
    R_transpose = R.T.tocsr() # Transpose for efficient calculation of sum(v_j * r_jk)

    # Create the power node contribution vector (a/m for power nodes, 0 otherwise)
    power_contrib = np.zeros(N)
    if m > 0:
        power_indices_list = list(power_node_indices)
        power_contrib[power_indices_list] = greedy_factor_a / m
    else:
        print("Warning: No power nodes identified (m=0). PowerTrust reduces to EigenTrust/PageRank model.")
        # Fallback: distribute the 'a' factor uniformly like PageRank personalization
        power_contrib = (greedy_factor_a / N) * np.ones(N)


    for iteration in range(max_iter):
        num_iterations = iteration + 1
        v_prev = v.copy()

        # PowerTrust Update Rule (Algorithm 3 variant):
        # v = (1-a) * R^T @ v_prev + P_power_contribution
        # where P_power_contribution is (a/m) for power nodes, 0 otherwise
        v = (1 - greedy_factor_a) * (R_transpose @ v_prev) + power_contrib

        # --- Check Sum (optional, should be close to 1) ---
        # current_sum = np.sum(v)
        # if num_iterations % 20 == 0: print(f"Iter {num_iterations}, Sum V: {current_sum}")
        # If sum deviates significantly, renormalization might be needed,
        # but the formula should theoretically preserve the sum.
        # v = v / current_sum # Optional renormalization

        # Calculate residual (L1 norm of the change)
        residual = np.linalg.norm(v - v_prev, ord=1)
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

    print("Iterative PowerTrust computation finished.")
    return v, residuals, num_iterations

# ============ Compute Hybrid Score ============
def compute_hybrid_score_pt(pt_scores_vec, endorse_norm_vec, lambda_weight):
    """Combines PowerTrust scores and normalized endorsement scores."""
    print(f"Computing hybrid score with lambda = {lambda_weight:.2f}")
    hybrid_scores = lambda_weight * pt_scores_vec + (1 - lambda_weight) * endorse_norm_vec
    return hybrid_scores

# ============ Save Final Scores ============
def save_scores_pt(users, scores, output_file, score_column_name="score"):
    """Saves the final scores (node IDs and scores) to a CSV file."""
    df_out = pd.DataFrame({
        "dst": users,
        score_column_name: scores
    })
    df_out.sort_values(by=score_column_name, ascending=False, inplace=True)
    try:
        df_out.to_csv(output_file, index=False)
        print(f"Scores saved to {output_file}.")
    except Exception as e:
        print(f"Error saving scores to {output_file}: {e}")

# ============ Plot Residual Curve (Generic) ============
def plot_residual_curve_pt(residuals, output_fig, algorithm_name="PowerTrust", convergence_tol=1e-6):
    """Plots the convergence curve (residuals vs. iteration)."""
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
    output_dir = "pt_output_iterative" # Use a new directory
    output_file_hybrid = os.path.join(output_dir, "powertrust_hybrid_iterative.csv")
    output_file_pt_only = os.path.join(output_dir, "powertrust_only_iterative.csv")
    residual_plot_file = os.path.join(output_dir, "powertrust_residuals_iterative.pdf")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in: {os.path.abspath(output_dir)}")

    # Algorithm Parameters
    lambda_weight = 0.8          # Weight for hybrid score
    power_ratio = 0.05           # Top % of nodes (based on rating stats) selected as power nodes
    min_feedback = 10            # Min feedback count for a node to be considered for power node status
    greedy_factor_a = 0.15       # Damping factor 'alpha' in PowerTrust paper
    max_iterations = 100         # Max iterations for PowerTrust
    convergence_tolerance = 1e-6 # Convergence tolerance for PowerTrust

    # --- Data Loading and Matrix Building ---
    R_matrix, users, user2idx, N = load_interaction_and_build_matrix_pt(interaction_file)
    if R_matrix is None:
        print("Exiting due to error loading interaction data.")
        return

    # --- Identify Power Nodes ---
    power_node_indices, m = identify_power_nodes_pt(interaction_file, users, user2idx, N,
                                                    power_ratio=power_ratio, min_feedback=min_feedback)

    # --- Load Endorsements ---
    endorsement_norm_vec = load_and_normalize_endorsements_pt(endorsement_file, users, user2idx, N)

    # --- Computation and Timing ---
    print("\nStarting Iterative PowerTrust computation and timing...")
    start_time = time.time() # <-- Start timer

    pt_scores_vec, residuals, iterations_run = compute_powertrust_iterative(
        R_matrix, N, power_node_indices, m,
        greedy_factor_a=greedy_factor_a,
        max_iter=max_iterations,
        tol=convergence_tolerance
    )

    end_time = time.time()   # <-- Stop timer
    runtime_s = end_time - start_time # <-- Calculate runtime

    # --- Hybrid Score Calculation ---
    hybrid_scores_vec = compute_hybrid_score_pt(pt_scores_vec, endorsement_norm_vec, lambda_weight)

    # --- Results ---
    final_residual = residuals[-1] if residuals else None

    # --- Terminal Output ---
    print("\n" + "="*40)
    print("    Iterative PowerTrust Computation Summary")
    print("="*40)
    print(f"Iterations Run:         {iterations_run}")
    if final_residual is not None:
        print(f"Final Residual (L1):    {final_residual:.6e}")
    else:
         print("Final Residual (L1):    N/A")
    print(f"Computation Runtime (s):{runtime_s:.4f}")
    print(f"Hybrid Lambda:          {lambda_weight:.2f}")
    print(f"Greedy Factor (a):      {greedy_factor_a:.2f}")
    print(f"# Power Nodes (m):      {m}")
    print(f"Power Node Ratio:       {power_ratio:.2f}")
    print(f"Min Feedback Count:     {min_feedback}")
    print("="*40 + "\n")

    # --- Saving Results ---
    save_scores_pt(users, hybrid_scores_vec, output_file_hybrid, score_column_name="hybrid_score")
    save_scores_pt(users, pt_scores_vec, output_file_pt_only, score_column_name="powertrust_score")

    # --- Plotting Residuals ---
    plot_residual_curve_pt(residuals, residual_plot_file, algorithm_name="PowerTrust", convergence_tol=convergence_tolerance)

    print("Iterative PowerTrust script finished successfully.")

# --- Execution Guard ---
if __name__ == "__main__":
    main()