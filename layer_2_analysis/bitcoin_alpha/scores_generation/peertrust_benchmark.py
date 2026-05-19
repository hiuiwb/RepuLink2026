#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements a simplified version of the PeerTrust algorithm:
It computes the PeerTrust score of each user as their weighted average rating,
optionally normalized by the number of feedbacks.
"""

import pandas as pd

def compute_peertrust(input_file, output_file, min_feedback=5):
    # Load dataset
    df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])

    # Group by target user (evaluatee) and compute average rating and feedback count
    grouped = df.groupby("dst")["rating"].agg(["mean", "count"]).reset_index()
    grouped.rename(columns={"dst": "user", "mean": "peertrust_score", "count": "feedback_count"}, inplace=True)

    # Optional: filter out users with too few ratings to avoid unreliable scores
    grouped = grouped[grouped["feedback_count"] >= min_feedback]

    # Normalize score to [0, 1] range for fair comparison (optional but useful for ROC curve)
    min_score = grouped["peertrust_score"].min()
    max_score = grouped["peertrust_score"].max()
    grouped["peertrust_score"] = (grouped["peertrust_score"] - min_score) / (max_score - min_score + 1e-6)

    # Keep only necessary columns
    grouped = grouped[["user", "peertrust_score"]]
    grouped.rename(columns={"user": "dst"}, inplace=True)  # Align with ground truth format

    # Save to CSV
    grouped.to_csv(output_file, index=False)
    print(f"PeerTrust scores saved to: {output_file}")

if __name__ == "__main__":
    # Change this path to your dataset
    input_file = "/ECShome/ww3y23/Github/Repulink/datasets/bitcoin_alpha.csv"
    output_file = "peertrust_scores.csv"
    compute_peertrust(input_file, output_file)
