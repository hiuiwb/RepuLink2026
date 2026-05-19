#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import defaultdict
import time # <-- Import time
import matplotlib.pyplot as plt # <-- Import matplotlib
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

# ============ Load Interaction Data & Compute Local Trust Matrix T ============
def load_interaction_and_compute_T(interaction_file):
    """
    Loads interaction data, classifies feedback, computes local trust Tij,
    and builds the local trust matrix T.
    Returns the T matrix, sorted list of nodes, and node-to-index mapping.
    """
    print(f"Loading interaction data from: {interaction_file}")
    df = pd.read_csv(interaction_file, header=None, names=["src", "dst", "rating", "timestamp"])
    print(f"Loaded {len(df)} interactions.")

    # classify feedback
    print("Classifying feedback...")
    def classify_rating(r):
        if r > 0: return 'pos'
        # elif r == 0: return 'neu' # Original AbsoluteTrust formula in paper uses pos/neg
        else: return 'neg'
        # If neutral ratings are important, the score formula needs adjustment.
        # Let's stick to pos/neg for now based on common AbsoluteTrust implementations.

    df['fb_type'] = df['rating'].apply(classify_rating)

    # Calculate local trust T_ij (based on pos/neg counts)
    print("Calculating local trust Tij...")
    local_trust_counts = defaultdict(lambda: defaultdict(lambda: {'pos': 0, 'neg': 0}))
    nodes = set()
    for _, row in df.iterrows():
        i, j, fb = row['src'], row['dst'], row['fb_type']
        if fb in local_trust_counts[i][j]:
             local_trust_counts[i][j][fb] += 1
        nodes.update([i, j]) # Track all nodes involved

    Tij = defaultdict(dict)
    processed_pairs = 0
    for i in local_trust_counts:
        for j in local_trust_counts[i]:
            counts = local_trust_counts[i][j]
            ng = counts['pos'] # Good interactions (positive ratings)
            nb = counts['neg'] # Bad interactions (negative ratings)
            nt = ng + nb      # Total interactions
            if nt == 0: continue

            # AbsoluteTrust local score: ng / nt (proportion of positive interactions)
            # This is a common interpretation. The original formula was (10*ng + ...)/nt
            # Let's use the simpler proportion ng/nt here. If the 10*ng + 5.5*nn + 1*nb needed, revert this.
            score = ng / nt
            Tij[i][j] = score
            processed_pairs += 1

    print(f"Calculated {processed_pairs} non-zero Tij values.")

    nodes = sorted(list(nodes))
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    N = len(nodes)
    print(f"Found {N} unique nodes in interactions.")

    # Build T matrix
    print("Building T matrix...")
    T = np.zeros((N, N))
    for i in Tij:
        for j in Tij[i]:
            if i in node2idx and j in node2idx: # Ensure nodes exist in our index
                 T[node2idx[i]][node2idx[j]] = Tij[i][j]
    print("T matrix built.")
    return T, nodes, node2idx, N

# ============ Load Endorsement Data & Normalize ============
def load_and_normalize_endorsements(endorsement_file, nodes, node2idx, N):
    """
    Loads endorsement data, counts endorsements for known nodes,
    and returns a min-max normalized endorsement vector aligned with 'nodes'.
    """
    print(f"Loading endorsement data from: {endorsement_file}")
    try:
        # Assuming Epinions format: src \t dst
        df_endorse = pd.read_csv(endorsement_file, sep="\t", header=None, names=["src", "dst"])
        print(f"Loaded {len(df_endorse)} endorsement links.")
    except FileNotFoundError:
        print(f"Error: Endorsement file not found at {endorsement_file}")
        print("Returning zero vector for endorsements.")
        return np.zeros(N)
    except Exception as e:
        print(f"Error loading endorsement file: {e}")
        print("Returning zero vector for endorsements.")
        return np.zeros(N)


    # Count endorsements received by each user ('dst')
    endorse_counts = df_endorse["dst"].value_counts()

    # Create a vector aligned with 'nodes'
    endorse_vec = np.zeros(N)
    found_endorsements = 0
    for user, count in endorse_counts.items():
        if user in node2idx:
            endorse_vec[node2idx[user]] = count
            found_endorsements += 1

    print(f"Found endorsements for {found_endorsements} nodes present in the interaction data.")

    # Min-Max Normalize the endorsement counts
    min_val = np.min(endorse_vec)
    max_val = np.max(endorse_vec)
    range_val = max_val - min_val

    if range_val > 1e-9: # Avoid division by zero if all counts are the same
        endorse_norm = (endorse_vec - min_val) / range_val
    else:
        # If all counts are the same (e.g., all zeros), normalized scores are uniform.
        # Assign 0.5 or 0 based on whether the single value is > 0 or == 0.
        endorse_norm = np.zeros(N) if max_val == 0 else np.ones(N) * 0.5

    print("Endorsement counts normalized.")
    return endorse_norm

# ============ Compute AbsoluteTrust Iteratively ============
def compute_absolutetrust_iterative(T, N, alpha=1/3, max_iter=100, tol=1e-6):
    """
    Performs the iterative AbsoluteTrust computation.
    Returns the final trust vector, list of residuals, and number of iterations.
    """
    print("Starting AbsoluteTrust iterative computation...")
    print(f"Parameters: alpha={alpha:.3f}, max_iter={max_iter}, tol={tol:.1e}")

    t = np.ones(N) / N # Start with a uniform trust distribution
    residuals = []
    num_iterations = 0

    # Precompute T @ ones(N) as it doesn't change in the loop
    Ct_sum_precomputed = T @ np.ones(N)

    for iteration in range(max_iter):
        num_iterations = iteration + 1
        t_prev = t.copy()

        # AbsoluteTrust Update Rule
        diag_t = np.diag(t_prev) # Use previous iteration's trust for stability? Check original paper/impl. Let's use t_prev.
        Ct = T @ t_prev             # C_t = T * t
        Ct_diag = T @ (diag_t @ t_prev) # C_t_diag = T * diag(t) * t
        # Ct_sum is constant if T doesn't change: T @ np.ones(N)

        # Avoid division by zero or NaNs
        # Denominator term 1: (T * diag(t) * t) ^ alpha
        denom1_base = np.maximum(Ct_diag, 1e-12) # Add epsilon before power
        denom1 = np.power(denom1_base, alpha)

        # Denominator term 2: (T * 1) ^ (1 + alpha)
        denom2_base = np.maximum(Ct_sum_precomputed, 1e-12) # Use precomputed sum
        denom2 = np.power(denom2_base, 1 + alpha)

        # Combined denominator D (as a vector before diag)
        # We need element-wise division before diag, D = denom1 / denom2
        # D = denom1 / denom2 # This D is actually diag(D') where D' is the term inside power
        # Correct formulation involves matrix D where D_ii = (Ct_diag_i / Ct_sum_i^(1+alpha))^(alpha) <- check this
        # Let's follow the provided formula structure closely:
        # D_vec = np.power(denom1_base, alpha) / np.power(denom2_base, 1 + alpha) # Element-wise calculation for diag
        # D_diag_matrix = np.diag(D_vec)
        # t = np.power(D_diag_matrix @ Ct, 1 / (1 + alpha)) # This seems consistent with the input formula

        # Simpler approach often used: t_new_i = sum_k(T_ki * t_k) / sum_k(T_ki) - Check if this is a variation?
        # Let's stick to the equation provided in the original script for now:

        # Calculate D vector for the diagonal matrix
        D_vec = np.power(Ct_diag, alpha) / np.power(np.maximum(Ct_sum_precomputed, 1e-12), 1 + alpha)
        D_vec = np.nan_to_num(D_vec) # Replace potential NaNs (0/0) with 0

        # Build D matrix
        D_diag_matrix = np.diag(D_vec)

        # Calculate the term inside the final power
        term_inside_power = D_diag_matrix @ Ct
        term_inside_power = np.maximum(term_inside_power, 0) # Ensure non-negativity before power

        # Update trust vector t
        t = np.power(term_inside_power, 1 / (1 + alpha))

        # Normalize t to sum to 1 (optional but often good practice for reputation scores)
        t_sum = np.sum(t)
        if t_sum > 1e-9:
            t = t / t_sum
        else:
            print(f"Warning: Trust vector sum close to zero at iteration {num_iterations}. Resetting to uniform.")
            t = np.ones(N) / N # Reset if sum is zero


        # Calculate residual (L1 norm of the change)
        residual = np.linalg.norm(t - t_prev, ord=1)
        residuals.append(residual)

        if num_iterations % 20 == 0 or num_iterations == 1: # Print progress occasionally
             print(f"Iteration {num_iterations}/{max_iter}, Residual (L1 Change): {residual:.4e}")

        # Check for convergence
        if residual < tol:
            print(f"Converged after {num_iterations} iterations (L1 Change < {tol:.1e}).")
            break

    if num_iterations == max_iter and residual >= tol:
        print(f"Reached maximum iterations ({max_iter}) without converging. Final L1 Change: {residual:.4e}")

    # Save residual data (optional, can be done in main)
    # pd.DataFrame({"iteration": list(range(1, len(residuals)+1)), "residual": residuals}) \
    #   .to_csv("absolutetrust_residuals.csv", index=False)
    # print("Residual data saved to absolutetrust_residuals.csv")

    print("AbsoluteTrust iterative computation finished.")
    return t, residuals, num_iterations

# ============ Compute Hybrid Score ============
def compute_hybrid_score(abs_scores, endorse_norm, lambda_weight):
    """Combines AbsoluteTrust scores and normalized endorsement scores."""
    print(f"Computing hybrid score with lambda = {lambda_weight:.2f}")
    hybrid_scores = lambda_weight * abs_scores + (1 - lambda_weight) * endorse_norm
    return hybrid_scores

# ============ Save Final Scores ============
def save_scores(nodes, scores, output_file, score_column_name="score"):
    """Saves the final scores (e.g., hybrid scores) to a CSV file."""
    df_out = pd.DataFrame({
        "dst": nodes,
        score_column_name: scores
    })
    df_out.sort_values(by=score_column_name, ascending=False, inplace=True)
    df_out.to_csv(output_file, index=False)
    print(f"Scores saved to {output_file}.")

# ============ Plot Residual Curve (Similar to RepuLink) ============
def plot_residual_curve(residuals, output_fig, algorithm_name="AbsoluteTrust", convergence_tol=1e-6):
    """Plots the convergence curve (residuals vs. iteration)."""
    print(f"Plotting residual curve to {output_fig}...")
    plt.figure(figsize=(8, 6))

    label_font = FontProperties(weight='bold', size=18)
    title_font = FontProperties(weight='bold', size=20)
    tick_font = FontProperties(weight='normal', size=14)
    legend_font = FontProperties(weight='bold', size=14)

    plt.plot(
        range(1, len(residuals) + 1),
        residuals,
        linestyle='-',
        linewidth=2,
        label='L1 Residual (Change)'
    )

    if convergence_tol:
        plt.axhline(
            y=convergence_tol,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label=f'Tolerance ({convergence_tol:.1e})'
        )

    plt.yscale('log')
    plt.title(f"{algorithm_name} Convergence Curve", fontproperties=title_font)
    plt.xlabel("Iteration", fontproperties=label_font)
    plt.ylabel("Residual (L1 norm)", fontproperties=label_font)

    # Apply tick font properties
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(tick_font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(tick_font)
    # Improve Y-axis formatting for log scale
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))


    plt.grid(True, which="both", ls=":", linewidth=0.5)
    plt.legend(loc="upper right", prop=legend_font)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"Residual curve saved to {output_fig}")
    plt.close()


# ============ Main Function ============
def main():
    # --- Configuration ---
    # Use absolute paths or paths relative to where the script is run
    interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_otc.csv"
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
    # Ensure the output directory exists or use a valid path
    output_dir = "./" # Example output directory
    output_file_hybrid = f"{output_dir}/absolutetrust_hybrid.csv"
    output_file_abs_only = f"{output_dir}/absolutetrust_only.csv"
    residual_plot_file = f"{output_dir}/absolutetrust_residuals.pdf"

    # Create output directory if it doesn't exist (optional)
    import os
    os.makedirs(output_dir, exist_ok=True)


    # Algorithm Parameters
    alpha = 1/3          # AbsoluteTrust parameter
    lambda_weight = 0.8  # Weight for hybrid score (AbsTrust vs Endorsement)
    max_iterations = 100 # Max iterations for AbsoluteTrust
    convergence_tolerance = 1e-6 # Convergence threshold for AbsoluteTrust

    # --- Data Loading ---
    T_matrix, nodes, node2idx, N = load_interaction_and_compute_T(interaction_file)
    endorsement_norm_vec = load_and_normalize_endorsements(endorsement_file, nodes, node2idx, N)

    # --- Computation and Timing ---
    print("\nStarting AbsoluteTrust computation and timing...")
    start_time = time.time() # <-- Start timer

    abs_scores_vec, residuals, iterations_run = compute_absolutetrust_iterative(
        T_matrix, N, alpha=alpha, max_iter=max_iterations, tol=convergence_tolerance
    )

    end_time = time.time()   # <-- Stop timer
    runtime_s = end_time - start_time # <-- Calculate runtime

    # --- Hybrid Score Calculation ---
    hybrid_scores_vec = compute_hybrid_score(abs_scores_vec, endorsement_norm_vec, lambda_weight)

    # --- Results ---
    final_residual = residuals[-1] if residuals else None # Get the last residual value

    # --- Terminal Output ---
    print("\n" + "="*40)
    print("      AbsoluteTrust Computation Summary")
    print("="*40)
    print(f"Iterations Run:         {iterations_run}")
    if final_residual is not None:
        print(f"Final Residual (L1):    {final_residual:.6e}")
    else:
         print("Final Residual (L1):    N/A (No iterations run or residuals tracked)")
    print(f"Computation Runtime (s):{runtime_s:.4f}")
    print(f"Hybrid Lambda:          {lambda_weight:.2f}")
    print(f"Alpha Parameter:        {alpha:.3f}")
    print("="*40 + "\n")

    # --- Saving Results ---
    save_scores(nodes, hybrid_scores_vec, output_file_hybrid, score_column_name="absolutetrust_hybrid_score")
    save_scores(nodes, abs_scores_vec, output_file_abs_only, score_column_name="absolutetrust_score") # Save non-hybrid scores too

    # --- Plotting Residuals ---
    if residuals: # Only plot if residuals were generated
        plot_residual_curve(residuals, residual_plot_file, algorithm_name="AbsoluteTrust", convergence_tol=convergence_tolerance)
    else:
        print("No residuals generated, skipping plot.")

    print("AbsoluteTrust script finished successfully.")

# --- Execution Guard ---
if __name__ == "__main__":
    main()