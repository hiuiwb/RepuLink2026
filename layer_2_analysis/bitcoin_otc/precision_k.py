#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def load_ground_truth(high_file, low_file):
    df_high = pd.read_csv(high_file)
    df_high['binary_label'] = 1
    df_low = pd.read_csv(low_file)
    df_low['binary_label'] = 0
    df_gt = pd.concat([df_high, df_low], ignore_index=True)
    return df_gt[['dst', 'binary_label']], len(df_high)

def evaluate_metrics_at_k(df_gt, df_score, score_column, ks, total_positives):
    df_merged = pd.merge(df_gt, df_score, on='dst', how='inner')
    df_sorted = df_merged.sort_values(by=score_column, ascending=False)

    precision_list, recall_list, f1_list = [], [], []

    for k in ks:
        top_k = df_sorted.head(k)
        tp = top_k['binary_label'].sum()

        precision = tp / k
        recall = tp / total_positives if total_positives > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return precision_list, recall_list, f1_list



def plot_metric_at_k(metric_dict, ks, metric_name="Precision@K", title="Bitcoin OTC", output_file="precision_at_k.pdf"):
    """
    Plot evaluation curves (Precision@K, Recall@K, F1@K) across different top-K thresholds,
    using the same visual style as the previous ROC curve plot.

    Parameters:
    - metric_dict: dict, e.g., {"EigenTrust": [0.8, 0.75, ...], "PageRank": [...]}
    - ks: list of K values used for evaluation (e.g., [5, 10, 20, 50, 75])
    - metric_name: str, label for y-axis ("Precision@K", "Recall@K", or "F1@K")
    - title: str, plot title
    - output_file: str, output file name (should end with .pdf)
    """

    # Define font style and size
    bold_font = FontProperties()
    bold_font.set_weight('bold')
    bold_font.set_size(20)

    legend_font = FontProperties()
    legend_font.set_weight('bold')
    legend_font.set_size(15)

    # Start plotting
    plt.figure(figsize=(8, 6))

    # Plot each algorithm's metric line
    for label, values in metric_dict.items():
        plt.plot(ks, values, linewidth=3, marker='o', label=label)

    # Configure axis labels and ticks
    plt.xlabel("Top-K", fontsize=20, fontproperties=bold_font)
    plt.ylabel(metric_name, fontsize=20, fontproperties=bold_font)
    plt.title(title, fontproperties=bold_font, fontsize=20)
    plt.xticks(ks, fontproperties=bold_font)
    plt.yticks(fontproperties=bold_font)
    plt.grid(True)

    # Add legend
    plt.legend(loc="lower right", prop=legend_font)

    # Save and show the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.savefig(output_file.replace(".pdf", ".png"), dpi=300)
    plt.show()



def main():
    ks = [10, 20, 30, 40, 50, 75, 100]

    high_file = "scores/high_reputation.csv"
    low_file = "scores/low_reputation.csv"

    score_files = {
        "PageRank": ("scores/pagerank_scores.csv", "pagerank_score"),
        "EigenTrust": ("scores/eigentrust_scores.csv", "eigentrust_score"),
        "PowerTrust": ("scores/powertrust_scores.csv", "powertrust_score"),
        "AbsoluteTrust": ("scores/absolutetrust_scores.csv", "absolutetrust_score"),
        "ShapleyTrust": ("scores/shapleytrust_scores.csv", "shapleytrust_score"),
        "RepuLink": ("scores/repulink_scores.csv", "repulink_score")
    }

    df_gt, total_positives = load_ground_truth(high_file, low_file)

    precision_all, recall_all, f1_all = {}, {}, {}

    # for algo, (file, score_col) in score_files.items():
    #     df_score = pd.read_csv(file)
    #     if score_col not in df_score.columns:
    #         print(f"[Warning] Column {score_col} not found in {file}, skipped.")
    #         continue

    #     precision, recall, f1 = evaluate_metrics_at_k(df_gt, df_score, score_col, ks, total_positives)
    #     precision_all[algo] = precision
    #     recall_all[algo] = recall
    #     f1_all[algo] = f1

    print("\nPrecision@100 Summary:")
    for algo, (file, score_col) in score_files.items():
        df_score = pd.read_csv(file)
        if score_col not in df_score.columns:
            print(f"[Warning] Column {score_col} not found in {file}, skipped.")
            continue

        precision, recall, f1 = evaluate_metrics_at_k(df_gt, df_score, score_col, ks, total_positives)
        precision_all[algo] = precision
        recall_all[algo] = recall
        f1_all[algo] = f1

        # Print Top-100 precision value
        precision_100 = precision[ks.index(100)]
        print(f"{algo:<15}: Precision@100 = {precision_100:.4f}")

    plot_metric_at_k(precision_all, ks, metric_name="Precision@K", output_file="precision_at_k.pdf")
    # plot_metric_at_k(recall_all, ks, metric_name="Recall@K", output_file="recall_at_k.pdf")
    # plot_metric_at_k(f1_all, ks, metric_name="F1@K", output_file="f1_at_k.pdf")



if __name__ == "__main__":
    main()
