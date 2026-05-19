#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parameter sensitivity for EigenTrust's hybrid weighting on Bitcoin-OTC.
Raw scores come from the corrected file
    layer_1_2_analysis/bitcoin_otc/scores/eigentrust_hybrid_corrected.csv
"""

import os

from _common import (
    OUT_DIR, ALPHAS, PK_K,
    ensure_ground_truth, load_ground_truth, load_endorse_norm,
    load_corrected_baseline_score,
    sweep_hybrid_lambda, save_metrics, plot_metric_vs_alpha_single,
)


def main():
    print("Loading corrected raw EigenTrust scores...")
    raw = load_corrected_baseline_score("eigentrust_hybrid_corrected.csv",
                                        "eigentrust_score")
    endo = load_endorse_norm()

    ensure_ground_truth()
    gt = load_ground_truth()
    print(f"Labelled {len(gt)} users "
          f"({int(gt['label'].sum())} high / {int((gt['label']==0).sum())} low)")

    df = sweep_hybrid_lambda(raw, "eigentrust_score", endo, gt, ALPHAS)
    save_metrics(df, "EigenTrust")

    plot_metric_vs_alpha_single(df["alpha"], df["AUC"], "AUC",
                                os.path.join(OUT_DIR, "sensitivity_auc_EigenTrust.pdf"),
                                label="EigenTrust")
    plot_metric_vs_alpha_single(df["alpha"], df[f"Precision@{PK_K}"], f"Precision@{PK_K}",
                                os.path.join(OUT_DIR, "sensitivity_precision_at_k_EigenTrust.pdf"),
                                label="EigenTrust")
    plot_metric_vs_alpha_single(df["alpha"], df["KendallTau"], r"Kendall's $\tau$",
                                os.path.join(OUT_DIR, "sensitivity_kendall_EigenTrust.pdf"),
                                label="EigenTrust")
    plot_metric_vs_alpha_single(df["alpha"], df["Spearman"], r"Spearman's $\rho$",
                                os.path.join(OUT_DIR, "sensitivity_spearman_EigenTrust.pdf"),
                                label="EigenTrust")


if __name__ == "__main__":
    main()
