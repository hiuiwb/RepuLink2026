#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared utilities for the parameter-sensitivity study.

Every per-algorithm script in this folder follows the same protocol:
  1. Ensure the ground truth is available from layer_1_2_analysis.
  2. Load the algorithm's raw interaction-only score vector on Bitcoin-Alpha.
  3. For alpha in {0.1, ..., 0.9}, combine with normalized endorsement
     in-degree and evaluate AUC / Precision@100 / Kendall's tau / Spearman.
  4. Save metrics_vs_alpha_<Algo>.csv.

This module centralises the shared pieces.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterable, Callable, Dict

import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_KEY = "bitcoin_alpha"
DATASET_NAME = "Bitcoin-Alpha"
LAYER_DIR = os.path.join(REPO, "layer_1_2_analysis", DATASET_KEY)
SG_DIR = os.path.join(LAYER_DIR, "scores_generation")

INTERACTION_FILE = os.path.join(REPO, "datasets", "bitcoin_alpha.csv")
ENDORSEMENT_FILE = os.path.join(REPO, "datasets", "epinions.txt")

GT_HIGH = os.path.join(OUT_DIR, "groundtruth_high.csv")
GT_LOW = os.path.join(OUT_DIR, "groundtruth_low.csv")

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
PK_K = 100


# ---------- Ground truth ----------
def ensure_ground_truth() -> None:
    if os.path.exists(GT_HIGH) and os.path.exists(GT_LOW):
        return
    print("Copying ground truth from layer_1_2_analysis ...")
    for fn, dst in (("groundtruth_high.csv", GT_HIGH),
                    ("groundtruth_low.csv", GT_LOW)):
        src = os.path.join(BASELINE_SCORES_DIR, fn)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing layer_1_2_analysis ground truth: {src}")
        shutil.copyfile(src, dst)


def load_ground_truth() -> pd.DataFrame:
    high = pd.read_csv(GT_HIGH).assign(label=1)
    low = pd.read_csv(GT_LOW).assign(label=0)
    gt = pd.concat([high, low], ignore_index=True)
    if "user" in gt.columns and "dst" not in gt.columns:
        gt = gt.rename(columns={"user": "dst"})
    return gt[["dst", "hybrid_score", "label"]]


BASELINE_SCORES_DIR = os.path.join(LAYER_DIR, "scores")


def load_corrected_baseline_score(filename: str, score_col: str) -> pd.DataFrame:
    """Load a pre-computed Bitcoin-Alpha baseline score file."""
    path = os.path.join(BASELINE_SCORES_DIR, filename)
    raw = pd.read_csv(path)
    missing = {"dst", score_col} - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return raw[["dst", score_col]].sort_values("dst").reset_index(drop=True)


# ---------- Endorsement normalisation ----------
def load_endorse_norm() -> pd.DataFrame:
    """Return DataFrame with columns [dst, endorse_norm], covering every dst
    that appears in the endorsement file. Uses the same min-max normalisation
    as the layer_1_2_analysis scripts."""
    df = pd.read_csv(ENDORSEMENT_FILE, sep="\t", header=None, names=["src", "dst"])
    cnt = df["dst"].value_counts().reset_index()
    cnt.columns = ["dst", "endorse_count"]
    rng = cnt["endorse_count"].max() - cnt["endorse_count"].min() + 1e-6
    cnt["endorse_norm"] = (cnt["endorse_count"] - cnt["endorse_count"].min()) / rng
    return cnt[["dst", "endorse_norm"]]


# ---------- Metrics ----------
def compute_metrics(scores_df: pd.DataFrame, gt_df: pd.DataFrame,
                    score_col: str, k: int = PK_K):
    df = pd.merge(gt_df, scores_df, on="dst", how="inner")
    y_true = df["label"].values
    y_score = df[score_col].values
    auc = roc_auc_score(y_true, y_score)
    top_k = df.sort_values(score_col, ascending=False).head(k)
    pk = top_k["label"].sum() / k
    tau, _ = kendalltau(df["hybrid_score"], df[score_col])
    rho, _ = spearmanr(df["hybrid_score"], df[score_col])
    return auc, pk, tau, rho


# ---------- Sweep helpers ----------
def sweep_hybrid_lambda(raw_scores_df: pd.DataFrame, raw_col: str,
                        endorse_df: pd.DataFrame, gt_df: pd.DataFrame,
                        alphas: Iterable[float] = ALPHAS) -> pd.DataFrame:
    """For each alpha in alphas, form hybrid = alpha * raw + (1-alpha) * endorse_norm
    over every dst present in raw_scores_df, then evaluate metrics vs gt_df."""
    merged = raw_scores_df.merge(endorse_df, on="dst", how="left")
    merged["endorse_norm"] = merged["endorse_norm"].fillna(0)

    rows = []
    for alpha in alphas:
        merged["hybrid_score_eval"] = (
            alpha * merged[raw_col] + (1 - alpha) * merged["endorse_norm"]
        )
        scores_df = merged[["dst", "hybrid_score_eval"]].copy()
        auc, pk, tau, rho = compute_metrics(scores_df, gt_df, "hybrid_score_eval")
        rows.append({
            "alpha": alpha, "AUC": auc, f"Precision@{PK_K}": pk,
            "KendallTau": tau, "Spearman": rho,
        })
        print(f"  alpha={alpha:.1f}  AUC={auc:.4f}  P@{PK_K}={pk:.4f}  "
              f"KT={tau:.4f}  SRC={rho:.4f}")
    return pd.DataFrame(rows)


def sweep_repulink_alpha(compute_fn: Callable[[float], pd.DataFrame],
                         gt_df: pd.DataFrame,
                         score_col: str,
                         alphas: Iterable[float] = ALPHAS) -> pd.DataFrame:
    """Generic sweep for methods whose alpha changes the raw score vector
    itself (i.e. RepuLink)."""
    rows = []
    for alpha in alphas:
        scores_df = compute_fn(alpha)
        auc, pk, tau, rho = compute_metrics(scores_df, gt_df, score_col)
        rows.append({
            "alpha": alpha, "AUC": auc, f"Precision@{PK_K}": pk,
            "KendallTau": tau, "Spearman": rho,
        })
        print(f"  alpha={alpha:.1f}  AUC={auc:.4f}  P@{PK_K}={pk:.4f}  "
              f"KT={tau:.4f}  SRC={rho:.4f}")
    return pd.DataFrame(rows)


def save_metrics(df: pd.DataFrame, algo_name: str) -> str:
    path = os.path.join(OUT_DIR, f"metrics_vs_alpha_{algo_name}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {path}")
    return path


# ---------- Plotting ----------
def _bold():
    f = FontProperties(); f.set_weight("bold"); return f


def plot_metric_vs_alpha_single(alphas, values, metric_name: str,
                                output_file: str, label: str = "RepuLink"):
    """Single-line sensitivity plot (used when only one method is present)."""
    legend = FontProperties(); legend.set_weight("bold"); legend.set_size(16)
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, values, linewidth=3, marker="o", markersize=10, label=label)
    plt.xlabel(r"$\alpha$", fontsize=30, fontproperties=_bold())
    plt.ylabel(metric_name, fontsize=30, fontproperties=_bold())
    plt.title(DATASET_NAME, fontproperties=_bold(), fontsize=30)
    plt.xticks(alphas, fontproperties=_bold(), fontsize=25)
    plt.yticks(fontproperties=_bold(), fontsize=25)
    plt.grid(True)
    plt.legend(loc="lower right", prop=legend)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved {output_file}")


def plot_metric_vs_alpha_multi(alphas, per_algo_values: Dict[str, Iterable[float]],
                               metric_name: str, output_file: str,
                               title: str = DATASET_NAME,
                               highlight: str = "RepuLink",
                               highlight_alpha: float = 1.0,
                               other_alpha: float = 0.7):
    """Multi-line sensitivity plot - one curve per algorithm.

    The curve whose label matches ``highlight`` is drawn at full opacity
    and with a slightly thicker line/marker; the remaining curves are
    drawn at ``other_alpha`` opacity so RepuLink visually stands out.
    """
    legend = FontProperties(); legend.set_weight("bold"); legend.set_size(16)
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    plt.figure(figsize=(8, 5))
    for i, (algo, values) in enumerate(per_algo_values.items()):
        is_highlight = (algo == highlight)
        plt.plot(
            alphas, values,
            linewidth=4 if is_highlight else 3,
            marker=markers[i % len(markers)],
            markersize=12 if is_highlight else 10,
            alpha=highlight_alpha if is_highlight else other_alpha,
            zorder=3 if is_highlight else 2,
            label=algo,
        )
    plt.xlabel(r"$\alpha$", fontsize=30, fontproperties=_bold())
    plt.ylabel(metric_name, fontsize=30, fontproperties=_bold())
    plt.title(title, fontproperties=_bold(), fontsize=30)
    plt.xticks(alphas, fontproperties=_bold(), fontsize=25)
    plt.yticks(fontproperties=_bold(), fontsize=25)
    plt.grid(True)
    plt.legend(loc="lower right", prop=legend)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved {output_file}")
