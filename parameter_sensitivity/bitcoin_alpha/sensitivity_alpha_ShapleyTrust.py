#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parameter sensitivity for ShapleyTrust's hybrid weighting on Bitcoin-Alpha.

Raw scores come from
    layer_1_2_analysis/bitcoin_alpha/scores/shapleytrust_hybrid.csv
(no *_corrected* variant exists for ShapleyTrust). The file stores
raw Shapley values in the column ``shapleytrust_score``.
"""

import os

from _common import (
    OUT_DIR, ALPHAS, PK_K,
    ensure_ground_truth, load_ground_truth, load_endorse_norm,
    load_corrected_baseline_score,
    sweep_hybrid_lambda, save_metrics, plot_metric_vs_alpha_single,
)


def main():
    print("Loading raw ShapleyTrust scores...")
    raw = load_corrected_baseline_score("shapleytrust_hybrid.csv",
                                        "shapleytrust_score")
    endo = load_endorse_norm()

    ensure_ground_truth()
    gt = load_ground_truth()
    print(f"Labelled {len(gt)} users "
          f"({int(gt['label'].sum())} high / {int((gt['label']==0).sum())} low)")

    df = sweep_hybrid_lambda(raw, "shapleytrust_score", endo, gt, ALPHAS)
    save_metrics(df, "ShapleyTrust")

    plot_metric_vs_alpha_single(df["alpha"], df["AUC"], "AUC",
                                os.path.join(OUT_DIR, "sensitivity_auc_ShapleyTrust.pdf"),
                                label="ShapleyTrust")
    plot_metric_vs_alpha_single(df["alpha"], df[f"Precision@{PK_K}"], f"Precision@{PK_K}",
                                os.path.join(OUT_DIR, "sensitivity_precision_at_k_ShapleyTrust.pdf"),
                                label="ShapleyTrust")
    plot_metric_vs_alpha_single(df["alpha"], df["KendallTau"], r"Kendall's $\tau$",
                                os.path.join(OUT_DIR, "sensitivity_kendall_ShapleyTrust.pdf"),
                                label="ShapleyTrust")
    plot_metric_vs_alpha_single(df["alpha"], df["Spearman"], r"Spearman's $\rho$",
                                os.path.join(OUT_DIR, "sensitivity_spearman_ShapleyTrust.pdf"),
                                label="ShapleyTrust")


if __name__ == "__main__":
    main()
