#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates hybrid trust scores for ShapleyTrust using both:
- Interaction scores (from shapleytrust_scores.csv)
- Endorsement information (from Epinions dataset)

The final hybrid score is a weighted sum:
    HybridScore = λ · interaction_score + (1 - λ) · normalized_endorsement
"""

import pandas as pd
import numpy as np

def main():
    # ============== Parameters ==============
    interaction_file = "../scores/shapleytrust_scores.csv"
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
    output_file = "../scores/shapleytrust_hybrid.csv"
    lambda_weight = 0.5  # Weight for interaction score

    # ============== Load Interaction Scores ==============
    df_inter = pd.read_csv(interaction_file)
    df_inter = df_inter.rename(columns={"shapleytrust_score": "interaction_score"})

    # ============== Load Endorsement Network ==============
    try:
        df_endorse = pd.read_csv(endorsement_file, sep="\t", comment="#", header=None, names=["src", "dst"])
    except Exception as e:
        print("Failed to read endorsement file:", e)
        return

    # ============== Compute Endorsement Score ==============
    endorsement_counts = df_endorse['dst'].value_counts().rename("endorsement_count").reset_index()
    endorsement_counts.columns = ["dst", "endorsement_count"]

    # Normalize endorsement counts
    max_count = endorsement_counts["endorsement_count"].max()
    endorsement_counts["endorsement_score"] = endorsement_counts["endorsement_count"] / (max_count + 1e-8)

    # ============== Merge with Interaction Scores ==============
    df_hybrid = pd.merge(df_inter, endorsement_counts, on="dst", how="left")
    df_hybrid["endorsement_score"] = df_hybrid["endorsement_score"].fillna(0)

    # Compute Hybrid Score
    df_hybrid["shapleytrust_hybrid_score"] = (
        lambda_weight * df_hybrid["interaction_score"] +
        (1 - lambda_weight) * df_hybrid["endorsement_score"]
    )

    # ============== Save ===================
    df_hybrid[["dst", "shapleytrust_hybrid_score"]].to_csv(output_file, index=False)
    print(f"Hybrid ShapleyTrust scores saved to: {output_file}")

if __name__ == "__main__":
    main()
