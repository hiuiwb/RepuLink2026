#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from scipy.stats import kendalltau

# Ground truth
df_gt = pd.concat([
    pd.read_csv("scores/high_reputation.csv").assign(binary_label=1),
    pd.read_csv("scores/low_reputation.csv").assign(binary_label=0)
], ignore_index=True)

methods = {
    "PageRank": ("scores/pagerank_scores.csv", "pagerank_score"),
    "EigenTrust": ("scores/eigentrust_scores.csv", "eigentrust_score"),
    "PowerTrust": ("scores/powertrust_scores.csv", "powertrust_score"),
    "AbsoluteTrust": ("scores/absolutetrust_scores.csv", "absolutetrust_score"),
    "ShapleyTrust": ("scores/shapleytrust_scores.csv", "shapleytrust_score"),
    "RepuLink": ("scores/repulink_scores.csv", "repulink_score")
}

print("Kendall’s Tau Correlation:\n")
for label, (file_path, score_col) in methods.items():
    try:
        df_score = pd.read_csv(file_path)
        df = pd.merge(df_gt, df_score, on="dst", how="inner")
        tau, _ = kendalltau(df["binary_label"], df[score_col])
        print(f"{label:<15}: τ = {tau:.4f}")
    except Exception as e:
        print(f"{label:<15}: Error - {e}")
