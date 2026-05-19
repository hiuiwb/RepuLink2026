#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid PowerTrust implementation combining:
- PowerTrust scores (from selected high-reputation nodes in Bitcoin OTC)
- Normalized endorsement in-degree from Epinions

Only users that appear in the Bitcoin OTC dataset are retained in the final output.
"""

import pandas as pd
import numpy as np

def normalize(series):
    """Min-max normalization to scale values to [0, 1]."""
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def compute_powertrust_with_endorsement(interaction_file, endorsement_file, output_file,
                                        power_ratio=0.05, min_feedback=10, lambda_weight=0.8):
    # ======== Load Bitcoin OTC (Interaction Layer) ==========
    df = pd.read_csv(interaction_file, header=None, names=["src", "dst", "rating", "timestamp"])
    print("Interaction data loaded:", df.shape)

    all_users = pd.unique(df[["src", "dst"]].values.ravel())
    all_users = sorted(all_users)

    # ======== Step 1: Identify Power Nodes ==========
    feedback_stats = df.groupby("src")["rating"].agg(["mean", "count"]).reset_index()
    feedback_stats = feedback_stats[feedback_stats["count"] >= min_feedback]
    feedback_stats = feedback_stats.sort_values(by="mean", ascending=False)
    num_power = int(len(feedback_stats) * power_ratio)
    power_nodes = set(feedback_stats["src"].iloc[:num_power])
    print(f"Selected {len(power_nodes)} power nodes.")

    # ======== Step 2: Ratings from Power Nodes ==========
    power_df = df[df["src"].isin(power_nodes)]
    scores = power_df.groupby("dst")["rating"].mean().reset_index()
    scores.rename(columns={"rating": "powertrust_score"}, inplace=True)
    scores["powertrust_score"] = normalize(scores["powertrust_score"])

    # ======== Step 3: Endorsement from Epinions ==========
    df_endorse = pd.read_csv(endorsement_file, sep='\t', header=None, names=['src', 'dst'])
    endorse_count = df_endorse['dst'].value_counts().reset_index()
    endorse_count.columns = ['dst', 'endorse_count']
    endorse_count['endorse_norm'] = normalize(endorse_count['endorse_count'])

    # ======== Step 4: Merge & Align with Bitcoin OTC Nodes Only ==========
    df_all = pd.DataFrame({'dst': all_users})

    df_all = df_all.merge(scores[['dst', 'powertrust_score']], on='dst', how='left')
    df_all = df_all.merge(endorse_count[['dst', 'endorse_norm']], on='dst', how='left')

    df_all['powertrust_score'] = df_all['powertrust_score'].fillna(0)
    df_all['endorse_norm'] = df_all['endorse_norm'].fillna(0)

    df_all['hybrid_score'] = (
        lambda_weight * df_all['powertrust_score'] +
        (1 - lambda_weight) * df_all['endorse_norm']
    )

    # ======== Step 5: Save ==========
    df_all[['dst', 'hybrid_score']].to_csv(output_file, index=False)
    print(f"PowerTrust hybrid scores saved to: {output_file}")

if __name__ == "__main__":
    interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_otc.csv"
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
    output_file = "powertrust_hybrid.csv"

    compute_powertrust_with_endorsement(interaction_file, endorsement_file, output_file)
