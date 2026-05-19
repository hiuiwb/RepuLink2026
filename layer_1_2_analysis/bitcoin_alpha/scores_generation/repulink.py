# # !/usr/bin/env python
# # -*- coding: utf-8 -*-

# import pandas as pd
# import numpy as np
# from collections import defaultdict

# # ============ Parameters ============
# interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_alpha.csv"
# endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
# output_file = "repulink_hybrid.csv"
# alpha = 0.5             # Weight between C (interaction) and F (endorsement)
# gamma = 0.85            # Discount factor for backward propagation
# epsilon = 1e-6          # Stability factor for normalization
# max_iter = 100
# tol = 1e-6

# # ============ Load Interaction Layer ============
# df_inter = pd.read_csv(interaction_file, header=None, names=["src", "dst", "rating", "timestamp"])
# users = sorted(set(df_inter["src"]).union(set(df_inter["dst"])))
# user2idx = {u: i for i, u in enumerate(users)}
# N = len(users)

# C = np.zeros((N, N))
# for _, row in df_inter.iterrows():
#     i, j = user2idx[row["src"]], user2idx[row["dst"]]
#     r = row["rating"]
#     if r >= 0:
#         C[i, j] += r
#     else:
#         C[i, j] += r  # allow both positive and negative

# # Row-normalize interaction matrix
# for i in range(N):
#     total = np.sum(np.maximum(C[i], 0))
#     if total > 0:
#         C[i] = np.maximum(C[i], 0) / total

# # ============ Load Endorsement Layer ============
# df_endo = pd.read_csv(endorsement_file, sep="\t", header=None, names=["src", "dst"])
# F = np.zeros((N, N))
# for _, row in df_endo.iterrows():
#     if row["src"] in user2idx and row["dst"] in user2idx:
#         i, j = user2idx[row["src"]], user2idx[row["dst"]]
#         F[i, j] = 1

# # Row-normalize endorsement matrix
# for i in range(N):
#     total = F[i].sum()
#     if total > 0:
#         F[i] /= total

# # ============ Forward Propagation ============
# W = alpha * C.T + (1 - alpha) * F.T
# r = np.ones(N) / N  # Initial reputation vector
# for _ in range(max_iter):
#     r_new = W @ r
#     if np.linalg.norm(r_new - r, 1) < tol:
#         break
#     r = r_new

# # ============ Backward Propagation ============
# def backward_propagation(F, signal, gamma, max_iter=20):
#     result = np.zeros(N)
#     current = signal.copy()
#     for _ in range(max_iter):
#         current = gamma * F @ current
#         result += current
#     return result

# # Penalty signal: total negative received
# neg_signal = np.zeros(N)
# for _, row in df_inter.iterrows():
#     if row["rating"] < 0:
#         j = user2idx[row["dst"]]
#         neg_signal[j] += abs(row["rating"])

# penalty = backward_propagation(F, neg_signal, gamma)

# # Reward signal: total positive received
# pos_signal = np.zeros(N)
# for _, row in df_inter.iterrows():
#     if row["rating"] > 0:
#         j = user2idx[row["dst"]]
#         pos_signal[j] += row["rating"]

# reward = backward_propagation(F, pos_signal, gamma)

# # ============ Apply Correction and Normalize ============
# # r_corrected = r - penalty + reward
# # r_clipped = np.maximum(r_corrected, 0)
# r_clipped = np.maximum(r, 0)
# r_normalized = r_clipped / (np.sum(r_clipped) + epsilon)

# # ============ Save Output ============
# df_out = pd.DataFrame({
#     "dst": users,
#     "repulink_score": r_normalized
# })
# df_out.to_csv(output_file, index=False)
# print(f"RepuLink scores saved to {output_file}")


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def load_interaction_data(input_file):
    df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
    users = pd.concat([df['src'], df['dst']]).unique()
    users.sort()
    user2idx = {u: i for i, u in enumerate(users)}
    N = len(users)
    C = np.zeros((N, N))
    for _, row in df.iterrows():
        i, j = user2idx[row["src"]], user2idx[row["dst"]]
        C[i, j] += row["rating"]
    for i in range(N):
        row_sum = np.sum(np.maximum(C[i], 0))
        if row_sum > 0:
            C[i] = np.maximum(C[i], 0) / row_sum
    return C, users, user2idx

def load_endorsement_data(endorsement_file, user2idx, N):
    df = pd.read_csv(endorsement_file, sep="\t", header=None, names=["src", "dst"])
    F = np.zeros((N, N))
    for _, row in df.iterrows():
        if row["src"] in user2idx and row["dst"] in user2idx:
            i, j = user2idx[row["src"]], user2idx[row["dst"]]
            F[i, j] = 1
    for i in range(N):
        row_sum = F[i].sum()
        if row_sum > 0:
            F[i] /= row_sum
    return F

def column_normalize(W, epsilon=1e-12):
    # Make each column sum to 1
    for j in range(W.shape[1]):
        col_sum = np.sum(W[:, j])
        if col_sum > epsilon:
            W[:, j] /= col_sum
    return W

def compute_repulink_hybrid(C, F, alpha=0.8, tol=1e-6, max_iter=1000):
    N = C.shape[0]
    W = alpha * C.T + (1 - alpha) * F.T

    # Column normalize W to prevent blow-up in power iteration
    W = column_normalize(W)

    r = np.ones(N) / N
    for iteration in range(max_iter):
        r_new = W @ r
        r_new = np.maximum(r_new, 0)  # Ensure non-negativity
        r_new = r_new / (np.sum(r_new) + 1e-12)
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            return r_new
        r = r_new
    print("Max iterations reached without full convergence.")
    return r

def save_scores(users, scores, output_file):
    df_out = pd.DataFrame({
        "dst": users,
        "repulink_score": scores
    })
    df_out.to_csv(output_file, index=False)
    print(f"RepuLink scores saved to {output_file}.")

def main():
    interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_alpha.csv"
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
    output_file = "repulink_hybrid_colnorm.csv"
    alpha = 0.8

    C, users, user2idx = load_interaction_data(interaction_file)
    F = load_endorsement_data(endorsement_file, user2idx, len(users))
    scores = compute_repulink_hybrid(C, F, alpha=alpha)
    save_scores(users, scores, output_file)

if __name__ == "__main__":
    main()
