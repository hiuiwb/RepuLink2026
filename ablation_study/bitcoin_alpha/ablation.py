#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ablation study script for RepuLink with Backward Propagation.

Calculates and saves reputation scores under four conditions:
1. Baseline (Forward Propagation Only)
2. Forward + Penalty Only (BEPP Effect)
3. Forward + Reward Only (BERP Effect)
4. Full Model (Forward + Penalty + Reward)

This allows analysis of the individual contributions of the
backward penalty and reward mechanisms.
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
    (Identical to the version in repulink_backward_prop)
    """
    print(f"Loading interaction data from: {input_file}")
    try:
        df_inter = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
        print(f"Loaded {len(df_inter)} interactions.")
    except FileNotFoundError:
        print(f"Error: Interaction file not found at {input_file}")
        return None, None, None, None, 0
    except Exception as e:
        print(f"Error loading interaction file: {e}")
        return None, None, None, None, 0

    users = pd.unique(df_inter[["src", "dst"]].values.ravel())
    users.sort()
    N = len(users)
    user2idx = {u: i for i, u in enumerate(users)}
    print(f"Found {N} unique users.")

    # Build interaction matrix T (raw scores) - needed for C calculation
    T = np.zeros((N, N))
    for _, row in df_inter.iterrows():
        if row["src"] in user2idx and row["dst"] in user2idx:
            i, j = user2idx[row["src"]], user2idx[row["dst"]]
            T[i, j] += row["rating"]

    # Build normalized interaction matrix C (based on positive ratings)
    print("Building row-normalized interaction matrix C...")
    C = np.zeros((N, N))
    for i in range(N):
        row_positive_sum = np.sum(np.maximum(T[i, :], 0))
        if row_positive_sum > epsilon:
            C[i, :] = np.maximum(T[i, :], 0) / row_positive_sum
        else:
            C[i, :] = 0
    print("C matrix built.")

    return C, df_inter, users, user2idx, N

# ============ Load Endorsement Data ============
def load_endorsement_data(endorsement_file, user2idx, N, epsilon=1e-12):
    """
    Loads endorsement data, builds the binary endorsement matrix E,
    and the row-normalized endorsement matrix F. Returns F.
    (Identical to the version in repulink_backward_prop)
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

    E = np.zeros((N, N))
    valid_endorsements = 0
    for _, row in df_endorse.iterrows():
        if row["src"] in user2idx and row["dst"] in user2idx:
            i, j = user2idx[row["src"]], user2idx[row["dst"]]
            E[i, j] = 1
            valid_endorsements += 1
    print(f"Found {valid_endorsements} valid endorsements among known users.")

    print("Building row-normalized endorsement matrix F...")
    F = np.zeros((N, N))
    zero_sum_rows = 0
    for i in range(N):
        row_sum = E[i, :].sum()
        if row_sum > epsilon:
            F[i, :] = E[i, :] / row_sum
        else:
            F[i, :] = 0
            zero_sum_rows += 1
    print(f"F matrix built. {zero_sum_rows} users had no outgoing endorsements (zero rows in F).")
    return F

# ============ Column Normalization (Optional) ============
def column_normalize(W, epsilon=1e-12):
    """Normalizes columns of a matrix to sum to 1."""
    W_norm = W.copy()
    for j in range(W_norm.shape[1]):
        col_sum = np.sum(W_norm[:, j])
        if col_sum > epsilon:
            W_norm[:, j] /= col_sum
    return W_norm

# ============ Compute RepuLink Hybrid (Forward Propagation) ============
def compute_repulink_forward(C, F, N, alpha=0.8, tol=1e-20, max_iter=1000, use_col_norm=True):
    """
    Computes the initial RepuLink reputation vector using forward propagation.
    (Identical to the version in repulink_backward_prop)
    """
    print("Starting RepuLink Forward Propagation...")
    print(f"Parameters: alpha={alpha:.2f}, max_iter={max_iter}, tol={tol:.1e}, use_col_norm={use_col_norm}")
    W = alpha * C.T + (1 - alpha) * F.T
    if use_col_norm:
        print("Applying column normalization to W...")
        W = column_normalize(W)

    r = np.ones(N) / N
    residuals = []
    converged = False
    for iteration in range(max_iter):
        r_prev = r.copy()
        r_new = W @ r_prev
        r = r_new
        residual = np.linalg.norm(r - r_prev, ord=1)
        residuals.append(residual)
        if (iteration + 1) % 100 == 0 or iteration == 0:
            print(f"Forward Iteration {iteration + 1}/{max_iter}, Residual (L1 Change): {residual:.4e}")
        if residual < tol:
            print(f"Forward propagation converged after {iteration + 1} iterations.")
            converged = True
            break
    if not converged:
        print(f"Forward propagation reached max iterations ({max_iter}). Final L1 Change: {residual:.4e}")
    print("RepuLink Forward Propagation finished.")
    return r, residuals

# ============ Backward Propagation ============
def backward_propagation(F, signal, N, gamma, max_iter=20, tol=1e-6):
    """
    Propagates a signal backward through the endorsement network F.
    (Identical to the version in repulink_backward_prop)
    """
    if np.linalg.norm(signal, 1) < 1e-12:
        print("Backward Propagation: Input signal is zero. Returning zero vector.")
        return np.zeros(N)

    print(f"Starting Backward Propagation (gamma={gamma:.2f}, max_iter={max_iter})...")
    propagated_signal = np.zeros(N)
    current_term = signal.copy()
    residuals = [] # Track term norms
    converged = False
    for iteration in range(max_iter):
        next_term = gamma * (F @ current_term)
        propagated_signal += next_term
        term_norm = np.linalg.norm(next_term, 1)
        residuals.append(term_norm)
        current_term = next_term
        if (iteration + 1) % 5 == 0 or iteration == 0:
             print(f"  Backward Iter {iteration + 1}/{max_iter}, Term Norm (L1): {term_norm:.4e}")
        if term_norm < tol:
            print(f"  Backward propagation converged after {iteration + 1} iterations (term norm < {tol:.1e}).")
            converged = True
            break
    if not converged:
         print(f"  Backward propagation reached max iterations ({max_iter}). Final term norm: {term_norm:.4e}")
    print("Backward Propagation finished.")
    return propagated_signal

# ============ Apply Correction and Normalize (Modified for Ablation) ============
def apply_correction(r_forward, penalty=None, reward=None, N=0, epsilon=1e-12):
    """
    Applies specified corrections (penalty and/or reward), clips, and normalizes.
    If penalty or reward is None or zero vector, it's skipped.
    """
    if penalty is None:
        penalty = np.zeros(N)
    if reward is None:
        reward = np.zeros(N)

    print("Applying corrections (Penalty Norm: {:.2f}, Reward Norm: {:.2f})...".format(
        np.linalg.norm(penalty, 1), np.linalg.norm(reward, 1)
    ))

    r_corrected = r_forward - penalty + reward
    r_clipped = np.maximum(r_corrected, 0)
    sum_clipped = np.sum(r_clipped)

    if sum_clipped > epsilon:
        r_normalized = r_clipped / sum_clipped
    else:
        print("Warning: Corrected reputation sum is near zero. Returning uniform distribution.")
        r_normalized = np.ones(N) / N # Fallback to uniform
    print("Correction and normalization applied.")
    return r_normalized

# ============ Save Final Scores ============
def save_scores(users, scores, output_file, score_column_name="score"):
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
    label_font = FontProperties(weight='bold', size=20)
    title_font = FontProperties(weight='bold', size=20)
    tick_font = FontProperties(weight='normal', size=15)
    legend_font = FontProperties(weight='bold', size=15)
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
    interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_alpha.csv"
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
    # Define a specific output directory for the ablation study
    output_dir = "repulink_ablation_study_output"
    # Define output file names for each condition
    output_file_forward = os.path.join(output_dir, "repulink_forward_only.csv")
    output_file_penalty = os.path.join(output_dir, "repulink_penalty_only.csv")
    output_file_reward = os.path.join(output_dir, "repulink_reward_only.csv")
    output_file_full = os.path.join(output_dir, "repulink_penalty_reward.csv")
    residual_plot_file = os.path.join(output_dir, "repulink_forward_residuals.pdf")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Ablation study output will be saved in: {os.path.abspath(output_dir)}")

    # Algorithm Parameters (Shared)
    alpha = 0.8
    forward_max_iter = 1000
    forward_tol = 1e-20
    use_column_norm = True
    gamma = 0.8
    backward_max_iter = 50
    backward_tol = 1e-6
    epsilon = 1e-12

    # --- Start Timer ---
    overall_start_time = time.time()

    # --- Load Data (Once) ---
    print("--- Loading Data ---")
    load_start = time.time()
    C, df_inter, users, user2idx, N = load_interaction_data(interaction_file, epsilon=epsilon)
    if C is None: return
    F = load_endorsement_data(endorsement_file, user2idx, N, epsilon=epsilon)
    load_time = time.time() - load_start
    print(f"Data Loading Time: {load_time:.4f}s")

    # --- Forward Propagation (Once) ---
    print("\n--- Running Forward Propagation (Baseline) ---")
    forward_start_time = time.time()
    r_forward, forward_residuals = compute_repulink_forward(
        C, F, N, alpha=alpha, tol=forward_tol, max_iter=forward_max_iter, use_col_norm=use_column_norm
    )
    forward_runtime = time.time() - forward_start_time
    # Save Baseline (Forward Only) results
    save_scores(users, r_forward, output_file_forward, score_column_name="repulink_forward_score")
    # Plot forward residuals
    plot_residual_curve(forward_residuals, residual_plot_file, algorithm_name="RepuLink Forward", convergence_tol=forward_tol)

    # --- Calculate Backward Signals (Once) ---
    print("\n--- Calculating Backward Signals ---")
    signal_calc_start = time.time()
    # Penalty signal
    neg_signal = np.zeros(N)
    for _, row in df_inter.iterrows():
        if row["rating"] < 0 and row["dst"] in user2idx:
            neg_signal[user2idx[row["dst"]]] += abs(row["rating"])
    print(f"Calculated initial negative signal (L1 norm: {np.linalg.norm(neg_signal, 1):.2f})")
    # Reward signal
    pos_signal = np.zeros(N)
    for _, row in df_inter.iterrows():
        if row["rating"] > 0 and row["dst"] in user2idx:
            pos_signal[user2idx[row["dst"]]] += row["rating"]
    print(f"Calculated initial positive signal (L1 norm: {np.linalg.norm(pos_signal, 1):.2f})")
    signal_calc_time = time.time() - signal_calc_start
    print(f"Signal Calculation Time: {signal_calc_time:.4f}s")


    # --- Backward Propagation (Penalty) ---
    print("\n--- Running Backward Propagation (Penalty) ---")
    backward_pen_start_time = time.time()
    penalty = backward_propagation(F, neg_signal, N, gamma, backward_max_iter, backward_tol)
    backward_pen_runtime = time.time() - backward_pen_start_time

    # --- Apply Penalty Correction Only ---
    r_penalty_only = apply_correction(r_forward, penalty=penalty, reward=None, N=N, epsilon=epsilon)
    save_scores(users, r_penalty_only, output_file_penalty, score_column_name="repulink_penalty_only_score")


    # --- Backward Propagation (Reward) ---
    print("\n--- Running Backward Propagation (Reward) ---")
    backward_rew_start_time = time.time()
    reward = backward_propagation(F, pos_signal, N, gamma, backward_max_iter, backward_tol)
    backward_rew_runtime = time.time() - backward_rew_start_time

    # --- Apply Reward Correction Only ---
    r_reward_only = apply_correction(r_forward, penalty=None, reward=reward, N=N, epsilon=epsilon)
    save_scores(users, r_reward_only, output_file_reward, score_column_name="repulink_reward_only_score")


    # --- Apply Full Correction (Penalty + Reward) ---
    print("\n--- Applying Full Correction (Penalty + Reward) ---")
    r_full = apply_correction(r_forward, penalty=penalty, reward=reward, N=N, epsilon=epsilon)
    save_scores(users, r_full, output_file_full, score_column_name="repulink_full_bepp_score")

    # --- Stop Timer ---
    overall_runtime = time.time() - overall_start_time

    # --- Terminal Output ---
    final_residual_fwd = forward_residuals[-1] if forward_residuals else None
    print("\n" + "="*50)
    print("      RepuLink Ablation Study Summary")
    print("="*50)
    print(f"Data Loading Time (s):             {load_time:.4f}")
    print(f"Forward Propagation Runtime (s):   {forward_runtime:.4f}")
    print(f"Forward Iterations Run:            {len(forward_residuals)}")
    if final_residual_fwd is not None:
        print(f"Forward Final Residual (L1):       {final_residual_fwd:.6e}")
    else:
        print(f"Forward Final Residual (L1):       N/A")
    print("-" * 50)
    print(f"Signal Calculation Time (s):       {signal_calc_time:.4f}")
    print(f"Backward Penalty Runtime (s):    {backward_pen_runtime:.4f}")
    print(f"Backward Reward Runtime (s):     {backward_rew_runtime:.4f}")
    print(f"Backward Discount (gamma):         {gamma:.2f}")
    print("-" * 50)
    print(f"Total Runtime (s):                 {overall_runtime:.4f}")
    print(f"Alpha (Forward):                   {alpha:.2f}")
    print("="*50)
    print(f"Output Files:")
    print(f"  - Forward Only: {output_file_forward}")
    print(f"  - Penalty Only: {output_file_penalty}")
    print(f"  - Reward Only:  {output_file_reward}")
    print(f"  - Full Model:   {output_file_full}")
    print(f"  - Residual Plot:{residual_plot_file}")
    print("="*50 + "\n")

    print("RepuLink Ablation Study script finished successfully.")

# --- Execution Guard ---
if __name__ == "__main__":
    main()
