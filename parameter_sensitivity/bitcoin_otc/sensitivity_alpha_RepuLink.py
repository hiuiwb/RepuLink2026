#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parameter sensitivity for RepuLink's alpha on Bitcoin-OTC.

Alpha controls the internal fusion W = alpha * C^T + (1-alpha) * F^T.
We sweep alpha in {0.1, ..., 0.9} and evaluate against the fixed two-layer
ground truth used throughout layer_1_2_analysis.
"""

import os
import sys
import pandas as pd

from _common import (
    SG_DIR, INTERACTION_FILE, ENDORSEMENT_FILE, OUT_DIR, ALPHAS,
    ensure_ground_truth, load_ground_truth,
    sweep_repulink_alpha, save_metrics, plot_metric_vs_alpha_single, PK_K,
)

sys.path.insert(0, SG_DIR)
from repulink import (  # type: ignore  # noqa: E402
    load_interaction_data,
    load_endorsement_data,
    compute_repulink_hybrid,
)


def main():
    print("Loading Bitcoin-OTC...")
    C, users, user2idx = load_interaction_data(INTERACTION_FILE)
    F = load_endorsement_data(ENDORSEMENT_FILE, user2idx, len(users))
    print(f"Users={len(users)}")

    ensure_ground_truth()
    gt = load_ground_truth()
    print(f"Labelled {len(gt)} users "
          f"({int(gt['label'].sum())} high / {int((gt['label']==0).sum())} low)")

    def compute_for_alpha(alpha: float) -> pd.DataFrame:
        scores = compute_repulink_hybrid(C, F, alpha=alpha)
        return pd.DataFrame({"dst": users, "repulink_score": scores})

    df = sweep_repulink_alpha(compute_for_alpha, gt, "repulink_score", ALPHAS)
    save_metrics(df, "RepuLink")

    plot_metric_vs_alpha_single(df["alpha"], df["AUC"], "AUC",
                                os.path.join(OUT_DIR, "sensitivity_auc_RepuLink.pdf"))
    plot_metric_vs_alpha_single(df["alpha"], df[f"Precision@{PK_K}"], f"Precision@{PK_K}",
                                os.path.join(OUT_DIR, "sensitivity_precision_at_k_RepuLink.pdf"))
    plot_metric_vs_alpha_single(df["alpha"], df["KendallTau"], r"Kendall's $\tau$",
                                os.path.join(OUT_DIR, "sensitivity_kendall_RepuLink.pdf"))
    plot_metric_vs_alpha_single(df["alpha"], df["Spearman"], r"Spearman's $\rho$",
                                os.path.join(OUT_DIR, "sensitivity_spearman_RepuLink.pdf"))


if __name__ == "__main__":
    main()
