#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script computes the EigenTrust score from interaction data (e.g., Bitcoin OTC)
and merges it with endorsement-based in-degree from a second dataset (e.g., Epinions).
The final output is a hybrid score that reflects both transactional and social trust.
"""

import pandas as pd
import numpy as np

# ======================== Load and Parse ============================
def load_interaction_data(input_file):
    df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
    return df

def build_trust_matrix(df, epsilon_row=1e-10):
    users = pd.concat([df['src'], df['dst']]).unique()
    users.sort()
    N = len(users)
    user2idx = { user: idx for idx, user in enumerate(users) }

    T = np.zeros((N, N))

    for _, row in df.iterrows():
        i, j = user2idx[row['src']], user2idx[row['dst']]
        rating = row['rating']
        if rating >= 0:
            T[i, j] += 1
        else:
            T[i, j] -= 1

    M = np.zeros_like(T)
    for i in range(N):
        row_sum = np.sum(np.maximum(T[i, :], 0))
        if row_sum > epsilon_row:
            M[i, :] = np.maximum(T[i, :], 0) / row_sum
        else:
            M[i, :] = 0

    return M, users

def compute_eigentrust(M, tol=1e-6, max_iter=1000):
    N = M.shape[0]
    r = np.ones(N) / N
    for i in range(max_iter):
        r_new = M.T @ r
        r_new = r_new / (np.sum(r_new) + 1e-12)
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"[EigenTrust] Converged in {i+1} iterations.")
            return r_new
        r = r_new
    print("[EigenTrust] Reached max iterations.")
    return r

# ==================== Endorsement Info ========================
def compute_endorsement_indegree(epinion_file):
    df = pd.read_csv(epinion_file, sep='\t', header=None, names=['src', 'dst'])
    in_deg = df['dst'].value_counts().reset_index()
    in_deg.columns = ['dst', 'endorse_count']
    return in_deg

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

# ========================= Main ===============================
def main():
    interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_alpha.csv"
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
    output_file = "eigentrust_hybrid.csv"
    lambda_weight = 0.5

    # Step 1: Load and compute EigenTrust from interaction layer
    df_inter = load_interaction_data(interaction_file)
    M, users = build_trust_matrix(df_inter)
    eigentrust_scores = compute_eigentrust(M)
    df_et = pd.DataFrame({"dst": users, "eigentrust_score": eigentrust_scores})

    # Step 2: Load endorsement in-degree
    df_endorse = compute_endorsement_indegree(endorsement_file)
    df_endorse['endorse_norm'] = normalize(df_endorse['endorse_count'])

    # Step 3: Merge and compute hybrid scores
    df_merged = pd.merge(df_et, df_endorse[['dst', 'endorse_norm']], on='dst', how='left')
    df_merged['endorse_norm'] = df_merged['endorse_norm'].fillna(0)
    df_merged['hybrid_score'] = (
        lambda_weight * df_merged['eigentrust_score'] +
        (1 - lambda_weight) * df_merged['endorse_norm']
    )

    # Step 4: Save result
    df_merged[['dst', 'hybrid_score']].to_csv(output_file, index=False)
    print(f"[Done] Hybrid EigenTrust scores saved to {output_file}")

if __name__ == "__main__":
    main()
