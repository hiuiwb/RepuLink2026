#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate per-algorithm metrics_vs_alpha_<Algo>.csv files into a single
table and produce one multi-line plot per metric (AUC, Precision@K,
Kendall's tau, Spearman's rho) in this folder.
"""

import os
import pandas as pd

from _common import (
    OUT_DIR, PK_K, plot_metric_vs_alpha_multi,
)


ALGORITHMS = [
    "PageRank",
    "EigenTrust",
    "PowerTrust",
    "AbsoluteTrust",
    "ShapleyTrust",
    "RepuLink",
]


def load_per_algo():
    frames = {}
    for algo in ALGORITHMS:
        path = os.path.join(OUT_DIR, f"metrics_vs_alpha_{algo}.csv")
        if not os.path.exists(path):
            print(f"[skip] missing {path} - run sensitivity_alpha_{algo}.py first")
            continue
        df = pd.read_csv(path)
        df["algorithm"] = algo
        frames[algo] = df
    return frames


def main():
    frames = load_per_algo()
    if not frames:
        print("No per-algorithm results found.")
        return

    combined = pd.concat(frames.values(), ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "all_metrics_vs_alpha.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Saved {combined_path}")

    alphas = sorted(combined["alpha"].unique())
    metrics = [
        ("AUC", "AUC", "sensitivity_auc.pdf"),
        (f"Precision@{PK_K}", f"Precision@{PK_K}", "sensitivity_precision_at_k.pdf"),
        ("KendallTau", r"Kendall's $\tau$", "sensitivity_kendall.pdf"),
        ("Spearman", r"Spearman's $\rho$", "sensitivity_spearman.pdf"),
    ]

    for col, label, fname in metrics:
        per_algo = {a: frames[a].sort_values("alpha")[col].tolist() for a in frames}
        plot_metric_vs_alpha_multi(alphas, per_algo, label,
                                   os.path.join(OUT_DIR, fname))


if __name__ == "__main__":
    main()
