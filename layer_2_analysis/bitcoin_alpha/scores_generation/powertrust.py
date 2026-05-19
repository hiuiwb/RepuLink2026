#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements a simplified version of the PowerTrust algorithm.
It selects a small set of high-reputation nodes as power nodes, and computes each user's
PowerTrust score based on ratings they receive from these power nodes only.
"""

import pandas as pd

def compute_powertrust(input_file, output_file, power_ratio=0.05, min_feedback=10):
    # ========== Load Dataset ==========
    df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])

    # ========== Step 1: Identify Power Nodes ==========
    feedback_stats = df.groupby("src")["rating"].agg(["mean", "count"]).reset_index()
    feedback_stats = feedback_stats[feedback_stats["count"] >= min_feedback]
    feedback_stats = feedback_stats.sort_values(by="mean", ascending=False)
    num_power = int(len(feedback_stats) * power_ratio)
    power_nodes = set(feedback_stats["src"].iloc[:num_power])
    print(f"Selected {len(power_nodes)} power nodes.")

    # ========== Step 2: Filter Ratings from Power Nodes Only ==========
    power_df = df[df["src"].isin(power_nodes)]

    # ========== Step 3: Compute PowerTrust Scores ==========
    # Compute the average rating each user receives from power nodes
    scores = power_df.groupby("dst")["rating"].mean().reset_index()
    scores.rename(columns={"dst": "user", "rating": "powertrust_score"}, inplace=True)

    # ========== Step 4: Normalize Scores to [0,1] ==========
    min_score = scores["powertrust_score"].min()
    max_score = scores["powertrust_score"].max()
    scores["powertrust_score"] = (scores["powertrust_score"] - min_score) / (max_score - min_score + 1e-6)

    # ========== Step 5: Save ==========
    scores.rename(columns={"user": "dst"}, inplace=True)  # Align with ground truth format
    scores.to_csv(output_file, index=False)
    print(f"PowerTrust scores saved to: {output_file}")

if __name__ == "__main__":
    input_file = "/ECShome/ww3y23/Github/Repulink/datasets/bitcoin_alpha.csv"
    output_file = "powertrust_scores.csv"
    compute_powertrust(input_file, output_file)
