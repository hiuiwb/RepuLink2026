#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.font_manager import FontProperties

def load_ground_truth():
    df_high = pd.read_csv("scores/high_reputation.csv")
    df_low = pd.read_csv("scores/low_reputation.csv")
    df_high["binary_label"] = 1
    df_low["binary_label"] = 0
    df_gt = pd.concat([df_high, df_low], ignore_index=True)
    return df_gt[["dst", "binary_label"]]

def load_scores(file_path, score_column):
    df = pd.read_csv(file_path)
    return df[["dst", score_column]]

def evaluate(df_gt, df_score, score_column):
    df_merged = pd.merge(df_gt, df_score, on="dst", how="inner")
    y_true = df_merged["binary_label"].values
    y_scores = df_merged[score_column].values
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    return fpr, tpr, auc

def main():
    # Ground truth
    df_gt = load_ground_truth()

    # Load and evaluate all methods
    methods = {
        "PageRank": ("scores/pagerank_scores.csv", "pagerank_score"),
        "EigenTrust": ("scores/eigentrust_scores.csv", "eigentrust_score"),
        "PowerTrust": ("scores/powertrust_scores.csv", "powertrust_score"),
        "AbsoluteTrust": ("scores/absolutetrust_scores.csv", "absolutetrust_score"),
        "ShapleyTrust": ("scores/shapleytrust_scores.csv", "shapleytrust_score"),
        "RepuLink": ("scores/repulink_scores.csv", "repulink_score")
    }

    plt.figure(figsize=(8, 6))

    for label, (file_path, col_name) in methods.items():
        try:
            df_score = load_scores(file_path, col_name)
            fpr, tpr, auc = evaluate(df_gt, df_score, col_name)
            plt.plot(fpr, tpr, linewidth=3, label=f"{label} (AUC = {auc:.3f})")
        except Exception as e:
            print(f"Error processing {label}: {e}")

    # Plot reference line
    bold_font = FontProperties()
    bold_font.set_weight('bold')
    bold_font.set_size(20)
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate", fontsize=20, fontproperties=bold_font)
    plt.ylabel("True Positive Rate", fontsize=20, fontproperties=bold_font)
    plt.xticks(fontproperties=bold_font)
    plt.yticks(fontproperties=bold_font)
    plt.title("Bitcoin Alpha", fontproperties=bold_font, fontsize=20)
    legend_font = FontProperties()
    legend_font.set_weight('bold')
    legend_font.set_size(15)
    plt.legend(loc="lower right", prop=legend_font)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("all_methods_roc_curve.pdf", dpi=300)
    plt.savefig("all_methods_roc_curve.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
