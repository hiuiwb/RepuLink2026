#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script computes AbsoluteTrust global scores based on the algorithm 
described in the IEEE TDSC paper. It reads the trust data from a CSV file
(Bitcoin OTC format), computes local trust T_ij, and then iteratively computes
the global trust vector t.

The final output is a CSV file: absolutetrust_scores.csv
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# =================== Step 1: Parameters ========================
alpha = 1 / 3     # AbsoluteTrust weight parameter a
max_iter = 1000    # Maximum number of iterations
tol = 1e-6        # Convergence tolerance
input_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_otc.csv"
output_file = "absolutetrust_scores.csv"

# =================== Step 2: Load Data =========================
df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])

# Re-map ratings into positive, neutral, negative
def classify_rating(r):
    if r > 0:
        return 'pos'
    elif r == 0:
        return 'neu'
    else:
        return 'neg'

df['fb_type'] = df['rating'].apply(classify_rating)

# Build local trust matrix T[i][j]
local_trust = defaultdict(lambda: defaultdict(list))

for _, row in df.iterrows():
    i, j, fb = row['src'], row['dst'], row['fb_type']
    local_trust[i][j].append(fb)

# Compute T_ij for all i,j using Equation (1)
Tij = defaultdict(dict)
nodes = set()

for i in local_trust:
    for j in local_trust[i]:
        fb_list = local_trust[i][j]
        ng = fb_list.count('pos')
        nn = fb_list.count('neu')
        nb = fb_list.count('neg')
        nt = ng + nn + nb
        if nt == 0:
            continue
        T_ij = (10 * ng + 5.5 * nn + 1 * nb) / nt
        Tij[i][j] = T_ij
        nodes.update([i, j])

nodes = sorted(list(nodes))
node2idx = {node: idx for idx, node in enumerate(nodes)}
N = len(nodes)

# Build T matrix (N x N)
T = np.zeros((N, N))
for i in Tij:
    for j in Tij[i]:
        src, dst = node2idx[i], node2idx[j]
        T[src][dst] = Tij[i][j]

# =================== Step 3: AbsoluteTrust Iteration =============
t = np.ones(N)  # initial global trust vector

for iteration in range(max_iter):
    t_prev = t.copy()
    
    diag_t = np.diag(t)
    Ct = T @ t             # Ttr * t
    Ct_diag = T @ (diag_t @ t)  # T * diag(t) * t
    Ct_sum = T @ np.ones(N)     # T * 1

    # Avoid divide-by-zero
    denom1 = np.maximum(Ct_diag, 1e-8)
    denom2 = np.maximum(Ct_sum, 1e-8)

    D = np.power(denom1, alpha) / np.power(denom2, 1 + alpha)
    D = np.diag(D)
    t = np.power(D @ Ct, 1 / (1 + alpha))

    # Check convergence
    if np.linalg.norm(t - t_prev, ord=1) < tol:
        print(f"Converged in {iteration+1} iterations.")
        break

# =============== Step 4: Output Scores ========================
df_out = pd.DataFrame({'dst': nodes, 'absolutetrust_score': t})
df_out.to_csv(output_file, index=False)
print(f"AbsoluteTrust scores written to {output_file}")
