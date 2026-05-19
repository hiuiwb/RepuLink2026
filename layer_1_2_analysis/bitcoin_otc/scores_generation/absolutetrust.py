#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import defaultdict

def compute_absolutetrust(interaction_file, endorsement_file, output_file, alpha=1/3, lambda_weight=0.5, max_iter=100, tol=1e-6):
    # Step 1: 读取 interaction 数据
    df = pd.read_csv(interaction_file, header=None, names=["src", "dst", "rating", "timestamp"])

    # classify feedback
    def classify_rating(r):
        if r > 0:
            return 'pos'
        elif r == 0:
            return 'neu'
        else:
            return 'neg'

    df['fb_type'] = df['rating'].apply(classify_rating)

    # Step 2: 计算 local trust T_ij
    local_trust = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        i, j, fb = row['src'], row['dst'], row['fb_type']
        local_trust[i][j].append(fb)

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
            score = (10 * ng + 5.5 * nn + 1 * nb) / nt
            Tij[i][j] = score
            nodes.update([i, j])

    nodes = sorted(list(nodes))
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    N = len(nodes)

    # Build T matrix
    T = np.zeros((N, N))
    for i in Tij:
        for j in Tij[i]:
            T[node2idx[i]][node2idx[j]] = Tij[i][j]

    # Step 3: AbsoluteTrust 迭代
    t = np.ones(N)
    for _ in range(max_iter):
        t_prev = t.copy()
        diag_t = np.diag(t)
        Ct = T @ t
        Ct_diag = T @ (diag_t @ t)
        Ct_sum = T @ np.ones(N)
        denom1 = np.maximum(Ct_diag, 1e-8)
        denom2 = np.maximum(Ct_sum, 1e-8)
        D = np.power(denom1, alpha) / np.power(denom2, 1 + alpha)
        D = np.diag(D)
        t = np.power(D @ Ct, 1 / (1 + alpha))
        if np.linalg.norm(t - t_prev, ord=1) < tol:
            break

    df_trust = pd.DataFrame({
        "dst": nodes,
        "absolutetrust_score": t
    })

    # Step 4: 加入 endorsement 层（Epinions）
    df_endorse = pd.read_csv(endorsement_file, sep="\t", header=None, names=["src", "dst"])
    endorse_count = df_endorse["dst"].value_counts().reset_index()
    endorse_count.columns = ["dst", "endorse_count"]

    df_merged = pd.merge(df_trust, endorse_count, on="dst", how="left").fillna(0)
    df_merged["endorse_norm"] = (df_merged["endorse_count"] - df_merged["endorse_count"].min()) / \
                                (df_merged["endorse_count"].max() - df_merged["endorse_count"].min() + 1e-6)

    # Step 5: 计算 hybrid 分数
    df_merged["hybrid_score"] = lambda_weight * df_merged["absolutetrust_score"] + \
                                 (1 - lambda_weight) * df_merged["endorse_norm"]

    df_merged[["dst", "hybrid_score"]].to_csv(output_file, index=False)
    print(f"Saved hybrid AbsoluteTrust scores to {output_file}")

if __name__ == "__main__":
    compute_absolutetrust(
        interaction_file="/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_otc.csv",
        endorsement_file="/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt",
        output_file="../scores/absolutetrust_hybrid.csv",
        alpha=1/3,
        lambda_weight=0.8
    )
