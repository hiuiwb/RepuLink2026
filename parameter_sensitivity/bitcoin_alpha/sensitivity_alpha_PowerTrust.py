#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parameter sensitivity for PowerTrust's hybrid weighting on Bitcoin-Alpha.
Raw scores come from the pre-computed raw scores file
    layer_1_2_analysis/bitcoin_alpha/scores/powertrust_hybrid.csv
"""

import os

from _common import (
    OUT_DIR, ALPHAS, PK_K,
    ensure_ground_truth, load_ground_truth, load_endorse_norm,
    load_corrected_baseline_score,
    sweep_hybrid_lambda, save_metrics, plot_metric_vs_alpha_single,
)


def main():
    print("Loading raw PowerTrust scores...")
    raw = load_corrected_baseline_score("powertrust_hybrid.csv",
                                        "powertrust_score")
    endo = load_endorse_norm()

    ensure_ground_truth()
    gt = load_ground_truth()
    print(f"Labelled {len(gt)} users "
          f"({int(gt['label'].sum())} high / {int((gt['label']==0).sum())} low)")

    df = sweep_hybrid_lambda(raw, "powertrust_score", endo, gt, ALPHAS)
    save_metrics(df, "PowerTrust")

    plot_metric_vs_alpha_single(df["alpha"], df["AUC"], "AUC",
                                os.path.join(OUT_DIR, "sensitivity_auc_PowerTrust.pdf"),
                                label="PowerTrust")
    plot_metric_vs_alpha_single(df["alpha"], df[f"Precision@{PK_K}"], f"Precision@{PK_K}",
                                os.path.join(OUT_DIR, "sensitivity_precision_at_k_PowerTrust.pdf"),
                                label="PowerTrust")
    plot_metric_vs_alpha_single(df["alpha"], df["KendallTau"], r"Kendall's $\tau$",
                                os.path.join(OUT_DIR, "sensitivity_kendall_PowerTrust.pdf"),
                                label="PowerTrust")
    plot_metric_vs_alpha_single(df["alpha"], df["Spearman"], r"Spearman's $\rho$",
                                os.path.join(OUT_DIR, "sensitivity_spearman_PowerTrust.pdf"),
                                label="PowerTrust")


if __name__ == "__main__":
    main()
