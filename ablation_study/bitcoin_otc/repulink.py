#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of RepuLink with Backward Endorsement Penalty
and Reward Propagation (BEPP & BERP).

Combines forward reputation calculation based on interactions (C)
and endorsements (F) with a backward correction mechanism that
propagates penalty/reward signals through the endorsement network.
"""

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

# ============ Load Interaction Data ============
def load_interaction_data(input_file, epsilon=1e-12):
    """
    Loads interaction data, builds the normalized interaction matrix C,
    and returns C, the raw interaction DataFrame, users list, user-to-index map, and N.
    """
    print(f"Loading interaction data from: {input_file}")
    try:
        # Load raw data
        df_inter = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
        print(f"Loaded {len(df_inter)} interactions.")
    except FileNotFoundError:
        print(f"Error: Interaction file not found at {input_file}")
        return None, None, None, None, 0
    except Exception as e:
        print(f"Error loading interaction file: {e}")
        return None, None, None, None, 0

    # Identify users and map to indices
    users = pd.unique(df_inter[["src", "dst"]].values.ravel())
    users.sort()
    N = len(users)
    user2idx = {u: i for i, u in enumerate(users)}
    print(f"Found {N} unique users.")

    # Build interaction matrix T (raw scores)
    T = np.zeros((N, N))
    for _, row in df_inter.iterrows():
        # Ensure users are in the map (should be true by construction)
        if row["src"] in user2idx and row["dst"] in user2idx:
            i, j = user2idx[row["src"]], user2idx[row["dst"]]
            # Accumulate ratings (can be positive or negative)
            T[i, j] += row["rating"]
            # Note: The paper normalizes T to C based on positive ratings only.
            # T_ij = (p_ij - n_ij) / (p_ij + n_ij + eps) -> C_ij = max(T_ij,0) / sum(max(T_ik,0))
            # Let's follow the paper's C definition (Eq 6)

    # Build normalized interaction matrix C (based on positive ratings)
    print("Building row-normalized interaction matrix C...")
    C = np.zeros((N, N))
    for i in range(N):
        # Consider only positive contributions for normalization sum
        row_positive_sum = np.sum(np.maximum(T[i, :], 0))
        if row_positive_sum > epsilon:
            # Normalize only the positive contributions
            C[i, :] = np.maximum(T[i, :], 0) / row_positive_sum
        else:
            # If a user gave no positive ratings, their row in C is zero
            C[i, :] = 0
    print("C matrix built.")

    # Return C, the original df, users, mapping, and count
    return C, df_inter, users, user2idx, N

# ============ Load Endorsement Data ============
def load_endorsement_data(endorsement_file, user2idx, N, epsilon=1e-12):
    """
    Loads endorsement data (e.g., Epinions format: src \t dst),
    builds the binary endorsement matrix E, and the row-normalized
    endorsement matrix F. Returns F.
    """
    print(f"Loading endorsement data from: {endorsement_file}")
    try:
        df_endorse = pd.read_csv(endorsement_file, sep='\t', header=None, names=['src', 'dst'])
        print(f"Loaded {len(df_endorse)} endorsement links.")
    except FileNotFoundError:
        print(f"Error: Endorsement file not found at {endorsement_file}")
        print("Warning: Returning zero matrix for F.")
        return np.zeros((N, N))
    except Exception as e:
        print(f"Error loading endorsement file: {e}")
        print("Warning: Returning zero matrix for F.")
        return np.zeros((N, N))

    # Build binary endorsement matrix E
    E = np.zeros((N, N))
    valid_endorsements = 0
    for _, row in df_endorse.iterrows():
        if row["src"] in user2idx and row["dst"] in user2idx:
            i, j = user2idx[row["src"]], user2idx[row["dst"]]
            E[i, j] = 1 # Binary endorsement
            valid_endorsements += 1
    print(f"Found {valid_endorsements} valid endorsements among known users.")

    # Build row-normalized endorsement matrix F (Eq 3)
    print("Building row-normalized endorsement matrix F...")
    F = np.zeros((N, N))
    zero_sum_rows = 0
    for i in range(N):
        row_sum = E[i, :].sum()
        if row_sum > epsilon:
            F[i, :] = E[i, :] / row_sum
        else:
            # If node i endorses no one, their row in F is zero
            F[i, :] = 0
            zero_sum_rows += 1
    print(f"F matrix built. {zero_sum_rows} users had no outgoing endorsements (zero rows in F).")
    return F

# ============ Column Normalization (for original RepuLink W) ============
def column_normalize(W, epsilon=1e-12):
    """Normalizes columns of a matrix to sum to 1."""
    W_norm = W.copy()
    for j in range(W_norm.shape[1]):
        col_sum = np.sum(W_norm[:, j])
        if col_sum > epsilon:
            W_norm[:, j] /= col_sum
        else:
             # Handle zero columns if necessary (e.g., distribute uniformly)
             # W_norm[:, j] = 1.0 / W_norm.shape[0]
             pass # Keep as zero for now
    return W_norm

# ============ Compute RepuLink Hybrid (Forward Propagation) ============
def compute_repulink_forward(C, F, N, alpha=0.8, tol=1e-6, max_iter=1000, use_col_norm=True):
    """
    Computes the initial RepuLink reputation vector using forward propagation.
    R(t+1) = W @ R(t), where W = alpha*C.T + (1-alpha)*F.T
    Optionally applies column normalization to W.
    Returns the reputation vector r and the list of residuals.
    """
    print("Starting RepuLink Forward Propagation...")
    print(f"Parameters: alpha={alpha:.2f}, max_iter={max_iter}, tol={tol:.1e}, use_col_norm={use_col_norm}")

    # Construct the combined propagation matrix W
    # Note: C and F are row-stochastic, so C.T and F.T are column-stochastic
    W = alpha * C.T + (1 - alpha) * F.T

    # Optional: Column normalize W (as in original script, prevents blow-up if W isn't stochastic)
    # If C.T and F.T are already column-stochastic, W is also column-stochastic,
    # and this normalization might not be strictly necessary but doesn't hurt.
    if use_col_norm:
        print("Applying column normalization to W...")
        W = column_normalize(W) # Ensures W is column-stochastic

    # Power Iteration
    r = np.ones(N) / N # Initial uniform reputation
    residuals = []
    converged = False

    for iteration in range(max_iter):
        r_prev = r.copy()
        r_new = W @ r_prev # Forward propagation step

        # Ensure non-negativity and normalization (already guaranteed if W is col-stochastic)
        # r_new = np.maximum(r_new, 0)
        # r_new = r_new / (np.sum(r_new) + 1e-12) # Sum should be 1 if W is col-stochastic

        r = r_new # Update reputation vector

        # Calculate residual (L1 norm of the change)
        residual = np.linalg.norm(r - r_prev, ord=1)
        residuals.append(residual)

        # Print progress occasionally
        if (iteration + 1) % 100 == 0 or iteration == 0:
            print(f"Forward Iteration {iteration + 1}/{max_iter}, Residual (L1 Change): {residual:.4e}")

        # Check for convergence
        if residual < tol:
            print(f"Forward propagation converged after {iteration + 1} iterations.")
            converged = True
            break

    if not converged:
        print(f"Forward propagation reached max iterations ({max_iter}) without converging. Final L1 Change: {residual:.4e}")

    print("RepuLink Forward Propagation finished.")
    return r, residuals # Return final vector and residuals list

# ============ Backward Propagation ============
def backward_propagation(F, signal, N, gamma, max_iter=20, tol=1e-6):
    """
    Propagates a signal (penalty or reward) backward through the
    endorsement network F using power iteration.
    phi = gamma*F*s + gamma^2*F^2*s + ...
    Returns the propagated signal vector (pi or rho).
    """
    if np.linalg.norm(signal, 1) < 1e-12: # No signal to propagate
        print("Backward Propagation: Input signal is zero. Returning zero vector.")
        return np.zeros(N)

    print(f"Starting Backward Propagation (gamma={gamma:.2f}, max_iter={max_iter})...")
    propagated_signal = np.zeros(N)
    current_term = signal.copy() # Starts with s
    residuals = []

    for iteration in range(max_iter):
        # Calculate next term: gamma * F @ current_term
        # Note: F is row-stochastic, used directly here as per Eq 16/20
        next_term = gamma * (F @ current_term)

        # Add to the result
        propagated_signal += next_term

        # Calculate residual (change in the term being added)
        # Use L1 norm of the term itself, stop when terms become small
        term_norm = np.linalg.norm(next_term, 1)
        residuals.append(term_norm)

        # Update current term for next iteration
        current_term = next_term

        if (iteration + 1) % 5 == 0 or iteration == 0:
             print(f"  Backward Iter {iteration + 1}/{max_iter}, Term Norm (L1): {term_norm:.4e}")

        # Check for convergence (stop when added terms are negligible)
        if term_norm < tol:
            print(f"  Backward propagation converged after {iteration + 1} iterations (term norm < {tol:.1e}).")
            break

    if iteration + 1 == max_iter and term_norm >= tol:
         print(f"  Backward propagation reached max iterations ({max_iter}). Final term norm: {term_norm:.4e}")

    print("Backward Propagation finished.")
    return propagated_signal

# ============ Apply Correction and Normalize ============
def apply_correction(r, penalty, reward, epsilon=1e-12):
    """Applies penalty and reward correction, clips, and normalizes."""
    print("Applying backward propagation corrections...")
    r_corrected = r - penalty + reward
    # Clip negative values to zero
    r_clipped = np.maximum(r_corrected, 0)
    # Normalize the final vector to sum to 1
    sum_clipped = np.sum(r_clipped)
    if sum_clipped > epsilon:
        r_normalized = r_clipped / sum_clipped
    else:
        print("Warning: Corrected reputation sum is near zero. Returning uniform distribution.")
        r_normalized = np.ones_like(r) / len(r) # Fallback to uniform
    print("Correction and normalization applied.")
    return r_normalized

# ============ Save Final Scores ============
def save_scores(users, scores, output_file, score_column_name="repulink_score"):
    """Saves the final scores to a CSV file."""
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

# ============ Plot Residual Curve (for Forward Propagation) ============
def plot_residual_curve(residuals, output_fig, algorithm_name="RepuLink Forward", convergence_tol=1e-6):
    """Plots the convergence curve (residuals vs. iteration) for the forward step."""
    if not residuals:
        print("No forward residuals to plot.")
        return
    print(f"Plotting forward propagation residual curve to {output_fig}...")
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
    output_dir = "repulink_output_bepp" # Define an output directory
    output_file_final = os.path.join(output_dir, "repulink_bepp_final.csv")
    output_file_forward = os.path.join(output_dir, "repulink_forward_only.csv")
    residual_plot_file = os.path.join(output_dir, "repulink_forward_residuals.pdf")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in: {os.path.abspath(output_dir)}")

    # Algorithm Parameters
    alpha = 0.8                  # Weight for interaction vs endorsement in forward step
    forward_max_iter = 1000      # Max iterations for forward RepuLink
    forward_tol = 1e-6           # Convergence tolerance for forward RepuLink
    use_column_norm = True       # Whether to column-normalize W in forward step

    gamma = 0.8                  # Discount factor for backward propagation (adjust as needed)
    backward_max_iter = 50       # Max iterations for backward propagation (paper suggests small K)
    backward_tol = 1e-6          # Convergence tolerance for backward propagation terms
    epsilon = 1e-12              # Small constant for normalization stability

    # --- Start Timer ---
    overall_start_time = time.time()

    # --- Load Data ---
    C, df_inter, users, user2idx, N = load_interaction_data(interaction_file, epsilon=epsilon)
    if C is None:
        print("Exiting due to error loading interaction data.")
        return
    F = load_endorsement_data(endorsement_file, user2idx, N, epsilon=epsilon)

    # --- Forward Propagation ---
    print("\n--- Running Forward Propagation ---")
    forward_start_time = time.time()
    r_forward, forward_residuals = compute_repulink_forward(
        C, F, N, alpha=alpha, tol=forward_tol, max_iter=forward_max_iter, use_col_norm=use_column_norm
    )
    forward_runtime = time.time() - forward_start_time
    # Save forward-only results
    save_scores(users, r_forward, output_file_forward, score_column_name="repulink_forward_score")
    # Plot forward residuals
    plot_residual_curve(forward_residuals, residual_plot_file, algorithm_name="RepuLink Forward", convergence_tol=forward_tol)


    # --- Backward Propagation ---
    print("\n--- Running Backward Propagation ---")
    backward_start_time = time.time()

    # Calculate initial penalty signal (sum of abs negative ratings received)
    neg_signal = np.zeros(N)
    for _, row in df_inter.iterrows():
        if row["rating"] < 0 and row["dst"] in user2idx:
            j = user2idx[row["dst"]]
            neg_signal[j] += abs(row["rating"]) # Use absolute value of neg rating
    print(f"Calculated initial negative signal (L1 norm: {np.linalg.norm(neg_signal, 1):.2f})")
    penalty = backward_propagation(F, neg_signal, N, gamma, backward_max_iter, backward_tol)

    # Calculate initial reward signal (sum of positive ratings received)
    pos_signal = np.zeros(N)
    for _, row in df_inter.iterrows():
        if row["rating"] > 0 and row["dst"] in user2idx:
            j = user2idx[row["dst"]]
            pos_signal[j] += row["rating"]
    print(f"Calculated initial positive signal (L1 norm: {np.linalg.norm(pos_signal, 1):.2f})")
    reward = backward_propagation(F, pos_signal, N, gamma, backward_max_iter, backward_tol)

    backward_runtime = time.time() - backward_start_time

    # --- Apply Correction ---
    r_final = apply_correction(r_forward, penalty, reward, epsilon=epsilon)

    # --- Stop Timer ---
    overall_runtime = time.time() - overall_start_time

    # --- Terminal Output ---
    final_residual_fwd = forward_residuals[-1] if forward_residuals else None
    print("\n" + "="*45)
    print("  RepuLink with Backward Prop. Summary")
    print("="*45)
    print(f"Forward Propagation Runtime (s): {forward_runtime:.4f}")
    print(f"Forward Iterations Run:          {len(forward_residuals)}")
    if final_residual_fwd is not None:
        print(f"Forward Final Residual (L1):     {final_residual_fwd:.6e}")
    else:
        print(f"Forward Final Residual (L1):     N/A")
    print("-" * 45)
    print(f"Backward Propagation Runtime (s):{backward_runtime:.4f}")
    print(f"Backward Discount (gamma):       {gamma:.2f}")
    print("-" * 45)
    print(f"Total Runtime (s):               {overall_runtime:.4f}")
    print(f"Alpha (Forward):                 {alpha:.2f}")
    print("="*45 + "\n")

    # --- Saving Final Results ---
    save_scores(users, r_final, output_file_final, score_column_name="repulink_bepp_score")

    print("RepuLink with Backward Propagation script finished successfully.")

# --- Execution Guard ---
if __name__ == "__main__":
    main()
