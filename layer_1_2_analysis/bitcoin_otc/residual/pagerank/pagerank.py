#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script computes PageRank scores from positive interactions
(e.g., Bitcoin OTC ratings > 0) using a manual power iteration method
to enable residual tracking.
It merges the score with endorsement-based in-degree from a second dataset (e.g., Epinions).
The final output is a hybrid score.

Refactored for modularity, timing, residual tracking/plotting.
"""

import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp # <-- Import SciPy sparse
import time
import os
import matplotlib.pyplot as plt # <-- Re-import matplotlib
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

# ============ Load Interaction Data & Build Weighted Graph ============
def load_interaction_and_build_graph_pr(interaction_file):
    """
    Loads interaction data and builds a networkx DiGraph where edges
    represent positive interactions (rating > 0), and edge weights
    are the sum of positive ratings between src and dst.
    Returns the graph G, sorted list of users, user-to-index mapping, and N.
    (Identical to previous version)
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

    users = pd.concat([df['src'], df['dst']]).unique()
    users.sort()
    N = len(users)
    user2idx = { user: idx for idx, user in enumerate(users) }
    print(f"Found {N} unique users.")

    print("Building graph G from positive interactions (rating > 0)...")
    G = nx.DiGraph()
    G.add_nodes_from(users) # Add nodes first
    positive_interactions_count = 0
    for _, row in df.iterrows():
        src, dst, rating = row["src"], row["dst"], row["rating"]
        if rating > 0:
            positive_interactions_count += 1
            if G.has_edge(src, dst):
                G[src][dst]['weight'] += rating
            else:
                G.add_edge(src, dst, weight=rating)

    print(f"Graph built using {positive_interactions_count} positive interactions.")
    print(f"Graph stats: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    return G, users, user2idx, N

# ============ Load Endorsement Data & Normalize ============
def load_and_normalize_endorsements_pr(endorsement_file, users, user2idx, N):
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


# ============ Compute PageRank Iteratively (Manual Implementation) ============
def compute_pagerank_iterative(G, N, users_list, alpha=0.85, personalization_vec=None,
                               max_iter=100, tol=1e-6, weight='weight'):
    """
    Computes PageRank using the power iteration method manually.
    Handles dangling nodes. Tracks residuals.

    Args:
        G (nx.DiGraph): Input graph.
        N (int): Number of nodes.
        users_list (list): Ordered list of nodes corresponding to matrix indices.
        alpha (float): Damping factor.
        personalization_vec (np.array, optional): Personalization vector. Defaults to uniform.
        max_iter (int): Maximum iterations.
        tol (float): Convergence tolerance for L1 norm change.
        weight (str): Edge attribute key for weights.

    Returns:
        tuple: (pagerank_vector, residuals_list, num_iterations)
    """
    print("Starting Manual PageRank iterative computation...")
    print(f"Parameters: alpha={alpha:.2f}, max_iter={max_iter}, tol={tol:.1e}")

    if N == 0:
        print("Warning: Graph has no nodes. Returning empty results.")
        return np.array([]), [], 0

    # Create mapping from node ID to index consistent with users_list
    node2idx = {node: i for i, node in enumerate(users_list)}

    # Build sparse adjacency matrix A (rows=source, cols=dest)
    # Ensure matrix dimensions match N, even if some nodes have no edges
    A = nx.to_scipy_sparse_array(G, nodelist=users_list, weight=weight, format='csr')
    if A.shape[0] != N or A.shape[1] != N:
         print(f"Warning: Adjacency matrix shape {A.shape} doesn't match N={N}. Adjusting.")
         # This might happen if users_list contains nodes not actually in G's nodeset after filtering
         # Rebuild adj matrix based only on nodes present in the graph G?
         # For now, let's assume users_list is consistent with G after load_interaction...
         # If issues persist, revisit graph/nodelist handling.
         pass # Assume shape is correct for now

    # Calculate out-degrees (sum of weights for each row)
    out_degree = np.array(A.sum(axis=1)).flatten()

    # Identify dangling nodes (out_degree is close to zero)
    dangling_nodes_mask = out_degree < 1e-12
    dangling_indices = np.where(dangling_nodes_mask)[0]
    non_dangling_indices = np.where(~dangling_nodes_mask)[0]
    print(f"Identified {len(dangling_indices)} dangling nodes.")

    # Create the transition matrix M (stochastic matrix)
    # Inverse of out-degree for non-dangling nodes
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore division by zero for dangling
        inv_out_degree = np.zeros(N)
        inv_out_degree[non_dangling_indices] = 1.0 / out_degree[non_dangling_indices]
        
    inv_out_degree_diag = sp.diags(inv_out_degree, format='csr')
    M = inv_out_degree_diag @ A # M_ij = A_ij / out_degree_i

    # Personalization vector p
    if personalization_vec is None:
        p = np.ones(N) / N
        print("Using uniform personalization vector.")
    else:
        p = personalization_vec
        if not np.isclose(np.sum(p), 1.0):
             print("Warning: Personalization vector does not sum to 1. Normalizing.")
             p = p / np.sum(p)
        print("Using provided personalization vector.")

    # --- Power Iteration ---
    pr = np.ones(N) / N # Initial PageRank vector (uniform)
    residuals = []
    num_iterations = 0
    converged = False
    M_transpose = M.T.tocsr() # Use CSR format for efficient matrix-vector product

    for iteration in range(max_iter):
        num_iterations = iteration + 1
        pr_prev = pr.copy()

        # Calculate contribution from dangling nodes
        dangle_sum = np.sum(pr_prev[dangling_indices])

        # PageRank Update Rule: pr = alpha * (M^T * pr + dangle_contribution * 1/N) + (1-alpha) * p
        # M^T @ pr handles transitions from non-dangling nodes
        # dangle_sum / N distributes dangling rank uniformly
        pr = alpha * (M_transpose @ pr_prev + (dangle_sum / N) * np.ones(N)) + (1 - alpha) * p

        # --- Check Sum (optional, should be close to 1) ---
        # if num_iterations % 20 == 0: print(f"Iter {num_iterations}, Sum PR: {np.sum(pr)}")

        # Calculate residual (L1 norm of the change)
        residual = np.linalg.norm(pr - pr_prev, ord=1)
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

    print("Manual PageRank iterative computation finished.")
    return pr, residuals, num_iterations


# ============ Compute Hybrid Score ============
def compute_hybrid_score_pr(pr_scores_vec, endorse_norm_vec, lambda_weight):
    """Combines PageRank scores and normalized endorsement scores."""
    print(f"Computing hybrid score with lambda = {lambda_weight:.2f}")
    # Ensure vectors are aligned (already should be based on users_list)
    hybrid_scores = lambda_weight * pr_scores_vec + (1 - lambda_weight) * endorse_norm_vec
    return hybrid_scores

# ============ Save Final Scores ============
def save_scores_pr(users, scores, output_file, score_column_name="score"):
    """Saves the final scores (node IDs and scores) to a CSV file."""
    df_out = pd.DataFrame({
        "dst": users, # Column name 'dst' to match previous examples
        score_column_name: scores
    })
    df_out.sort_values(by=score_column_name, ascending=False, inplace=True)
    try:
        df_out.to_csv(output_file, index=False)
        print(f"Scores saved to {output_file}.")
    except Exception as e:
        print(f"Error saving scores to {output_file}: {e}")


# ============ Plot Residual Curve (Generic) ============
def plot_residual_curve_pr(residuals, output_fig, algorithm_name="PageRank", convergence_tol=1e-6):
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
    output_dir = "pr_output_manual" # Use a new directory
    output_file_hybrid = os.path.join(output_dir, "pagerank_hybrid_manual.csv")
    output_file_pr_only = os.path.join(output_dir, "pagerank_only_manual.csv")
    residual_plot_file = os.path.join(output_dir, "pagerank_residuals_manual.pdf") # Add back plot file

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in: {os.path.abspath(output_dir)}")

    # Algorithm Parameters
    lambda_weight = 0.8          # Weight for hybrid score
    pagerank_alpha = 0.85        # Damping factor for PageRank
    pagerank_max_iter = 100      # Max iterations for PageRank
    pagerank_tol = 1e-6          # Convergence tolerance for PageRank

    # --- Data Loading and Graph Building ---
    G, users, user2idx, N = load_interaction_and_build_graph_pr(interaction_file)
    if G is None:
        print("Exiting due to error loading interaction data.")
        return

    endorsement_norm_vec = load_and_normalize_endorsements_pr(endorsement_file, users, user2idx, N)

    # --- Computation and Timing ---
    print("\nStarting Manual PageRank computation and timing...")
    start_time = time.time() # <-- Start timer

    # Using the manual implementation now
    pr_scores_vec, residuals, iterations_run = compute_pagerank_iterative(
        G, N, users, # Pass users list for consistent node ordering
        alpha=pagerank_alpha,
        max_iter=pagerank_max_iter,
        tol=pagerank_tol,
        weight='weight' # Use summed positive ratings as weights
    )

    end_time = time.time()   # <-- Stop timer
    runtime_s = end_time - start_time # <-- Calculate runtime

    # --- Hybrid Score Calculation ---
    hybrid_scores_vec = compute_hybrid_score_pr(pr_scores_vec, endorsement_norm_vec, lambda_weight)

    # --- Results ---
    final_residual = residuals[-1] if residuals else None

    # --- Terminal Output ---
    print("\n" + "="*40)
    print("      Manual PageRank Computation Summary")
    print("="*40)
    print(f"Iterations Run:         {iterations_run}") # Now available
    if final_residual is not None:
        print(f"Final Residual (L1):    {final_residual:.6e}") # Now available
    else:
         print("Final Residual (L1):    N/A")
    print(f"Computation Runtime (s):{runtime_s:.4f}")
    print(f"Hybrid Lambda:          {lambda_weight:.2f}")
    print(f"PageRank Alpha:         {pagerank_alpha:.2f}")
    print(f"PageRank Max Iter:      {pagerank_max_iter}")
    print(f"PageRank Tolerance:     {pagerank_tol:.1e}")
    print("="*40 + "\n")

    # --- Saving Results ---
    save_scores_pr(users, hybrid_scores_vec, output_file_hybrid, score_column_name="hybrid_score")
    save_scores_pr(users, pr_scores_vec, output_file_pr_only, score_column_name="pagerank_score") # Save non-hybrid scores too

    # --- Plotting Residuals ---
    plot_residual_curve_pr(residuals, residual_plot_file, algorithm_name="PageRank", convergence_tol=pagerank_tol) # Call plotting function

    print("Manual PageRank script finished successfully.")

# --- Execution Guard ---
if __name__ == "__main__":
    main()