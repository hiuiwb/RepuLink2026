#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for the RepuLink Ablation Study.

Loads the results from the four scenarios (Forward Only, Penalty Only,
Reward Only, Full Model) and generates plots to compare scores and rankings,
visualizing the effects of BEPP and BERP.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Ensure seaborn is imported
import os
from matplotlib.font_manager import FontProperties

# ============ Configuration ============

# --- Input ---
# Directory where the ablation study CSV files are saved
INPUT_DIR = "repulink_ablation_study_output"
FILE_FORWARD = os.path.join(INPUT_DIR, "repulink_forward_only.csv")
FILE_PENALTY = os.path.join(INPUT_DIR, "repulink_penalty_only.csv")
FILE_REWARD = os.path.join(INPUT_DIR, "repulink_reward_only.csv")
FILE_FULL = os.path.join(INPUT_DIR, "repulink_penalty_reward.csv")

# Column names used in the CSV files
COL_FORWARD = "repulink_forward_score"
COL_PENALTY = "repulink_penalty_only_score"
COL_REWARD = "repulink_reward_only_score"
COL_FULL = "repulink_full_bepp_score"
COL_USER = "dst" # User ID column

# --- Output ---
# Directory to save the generated figures
OUTPUT_DIR_FIGS = os.path.join(INPUT_DIR, "figures")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR_FIGS, exist_ok=True)
print(f"Figures will be saved in: {os.path.abspath(OUTPUT_DIR_FIGS)}")

# --- Plotting Parameters ---
# SNS_STYLE = "ticks" # Seaborn style name
FIG_DPI = 300
FONT_TITLE = FontProperties(weight='bold', size=20)
FONT_LABEL = FontProperties(weight='bold', size=20)
FONT_TICK = FontProperties(weight='bold', size=20)
legend_font = FontProperties(weight='bold', size=15)

# Apply the Seaborn style globally
# sns.set_style(SNS_STYLE)

# ============ Helper Functions ============

def load_and_merge_scores(file_fwd, file_pen, file_rew, file_full):
    """Loads the four score files and merges them into a single DataFrame."""
    print("Loading score files...")
    try:
        df_fwd = pd.read_csv(file_fwd)
        df_pen = pd.read_csv(file_pen)
        df_rew = pd.read_csv(file_rew)
        df_full = pd.read_csv(file_full)
        print("Files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Ensure the ablation study script has been run and files exist in:", INPUT_DIR)
        return None
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return None

    # Merge the dataframes
    print("Merging dataframes...")
    df_merged = pd.merge(df_fwd, df_pen, on=COL_USER, how='inner', suffixes=('', '_pen'))
    df_merged = pd.merge(df_merged, df_rew, on=COL_USER, how='inner', suffixes=('', '_rew'))
    df_merged = pd.merge(df_merged, df_full, on=COL_USER, how='inner', suffixes=('', '_full'))
    print("Dataframes merged.")

    # Add ranks for each score column (higher score = lower rank number)
    print("Calculating ranks...")
    df_merged['rank_forward'] = df_merged[COL_FORWARD].rank(ascending=False, method='first')
    df_merged['rank_penalty'] = df_merged[COL_PENALTY].rank(ascending=False, method='first')
    df_merged['rank_reward'] = df_merged[COL_REWARD].rank(ascending=False, method='first')
    df_merged['rank_full'] = df_merged[COL_FULL].rank(ascending=False, method='first')
    print("Ranks calculated.")

    return df_merged

def plot_rank_comparison(df, rank_col1, rank_col2, title, filename):
    """Generates a scatter plot comparing ranks from two scenarios."""
    print(f"Generating rank comparison plot: {title}")
    # plt.style.use(SNS_STYLE) # Removed: Style set globally via sns.set_style()
    plt.figure(figsize=(8, 6))

    # Scatter plot
    sns.scatterplot(data=df, x=rank_col1, y=rank_col2, alpha=0.8, s=15, edgecolor = 'blue')

    # Add y=x line for reference
    max_rank = max(df[rank_col1].max(), df[rank_col2].max())
    # Ensure max_rank is at least 1 for plotting
    max_rank = max(1, max_rank)
    plt.plot([1, max_rank], [1, max_rank], color='red', linestyle='--', linewidth=3)

    plt.title(title, fontproperties=FONT_TITLE)
    plt.xlabel(f"Rank ({rank_col1.split('_')[1].capitalize()})", fontproperties=FONT_LABEL)
    plt.ylabel(f"Rank ({rank_col2.split('_')[1].capitalize()})", fontproperties=FONT_LABEL)
    # plt.legend(prop=legend_font)
    # plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.grid(True)
    # Set axis limits to start from 1 and potentially use log scale if ranks vary widely
    plt.xlim(left=0.8, right=max_rank * 1.05)
    plt.ylim(bottom=0.8, top=max_rank * 1.05)
    # Optional: Log scale if needed for visualization
    # plt.xscale('log')
    # plt.yscale('log')

    # Apply tick font properties
    ax = plt.gca()
    for label in ax.get_xticklabels(): label.set_fontproperties(FONT_TICK)
    for label in ax.get_yticklabels(): label.set_fontproperties(FONT_TICK)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR_FIGS, filename)
    plt.savefig(filepath, dpi=FIG_DPI)
    print(f"Saved: {filepath}")
    plt.close()

def plot_score_distribution(df, cols, labels, title, filename):
    """Generates density plots comparing score distributions."""
    print(f"Generating score distribution plot: {title}")
    # plt.style.use(SNS_STYLE) # Removed: Style set globally via sns.set_style()
    plt.figure(figsize=(10, 6))

    for col, label in zip(cols, labels):
        # Use kdeplot from seaborn
        sns.histplot(df[col], label=label, fill=True, alpha=0.3, lw=3, kde=True)

    plt.title(title, fontproperties=FONT_TITLE)
    plt.xlabel("Reputation Score", fontproperties=FONT_LABEL)
    plt.ylabel("Density", fontproperties=FONT_LABEL)
    # plt.legend(prop=legend_font)
    # plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.grid(True)
    # Optional: Adjust x-axis limits if scores are clustered
    # plt.xlim(left=0, right=df[cols].max().max() * 1.1)

    # Apply tick font properties
    ax = plt.gca()
    for label in ax.get_xticklabels(): label.set_fontproperties(FONT_TICK)
    for label in ax.get_yticklabels(): label.set_fontproperties(FONT_TICK)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR_FIGS, filename)
    plt.savefig(filepath, dpi=FIG_DPI)
    print(f"Saved: {filepath}")
    plt.close()

def plot_rank_change_histogram(df, rank_col_base, rank_col_compare, title, filename):
    """Generates a histogram of rank changes."""
    print(f"Generating rank change histogram: {title}")
    # Calculate rank change (positive means rank improved, i.e., rank number decreased)
    rank_change = df[rank_col_base] - df[rank_col_compare]

    # plt.style.use(SNS_STYLE) # Removed: Style set globally via sns.set_style()
    plt.figure(figsize=(10, 6))

    # Use histplot from seaborn
    sns.histplot(rank_change, bins=50, kde=True, edgecolor="black", line_kws={'color': 'crimson', 'lw': 3})

    avg_change = rank_change.mean()
    plt.axvline(avg_change, color='red', linestyle='--', linewidth=3, label=f'Avg Change: {avg_change:.2f}')

    plt.title(title, fontproperties=FONT_TITLE)
    plt.xlabel("Rank Change (Base Rank - New Rank)", fontproperties=FONT_LABEL)
    plt.ylabel("Frequency", fontproperties=FONT_LABEL)
    # plt.legend(prop=legend_font)
    # plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.grid(True)

    # Apply tick font properties
    ax = plt.gca()
    for label in ax.get_xticklabels(): label.set_fontproperties(FONT_TICK)
    for label in ax.get_yticklabels(): label.set_fontproperties(FONT_TICK)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR_FIGS, filename)
    plt.savefig(filepath, dpi=FIG_DPI)
    print(f"Saved: {filepath}")
    plt.close()


# ============ Main Execution ============

if __name__ == "__main__":
    print("--- RepuLink Ablation Study Visualization ---")

    # Load and prepare data
    df_scores = load_and_merge_scores(FILE_FORWARD, FILE_PENALTY, FILE_REWARD, FILE_FULL)

    if df_scores is not None:
        print(f"\nLoaded and merged data for {len(df_scores)} users.")

        # --- Generate Plots ---

        # 1. Rank Comparison Scatter Plots
        plot_rank_comparison(df_scores, 'rank_forward', 'rank_penalty',
                             'Forward Only vs. BEPP Only',
                             'rank_scatter_fwd_vs_pen.png')
        plot_rank_comparison(df_scores, 'rank_forward', 'rank_reward',
                             'Forward Only vs. BERP Only',
                             'rank_scatter_fwd_vs_rew.png')
        plot_rank_comparison(df_scores, 'rank_forward', 'rank_full',
                             'Forward Only vs. Full Model',
                             'rank_scatter_fwd_vs_full.png')
        plot_rank_comparison(df_scores, 'rank_penalty', 'rank_full',
                             'BEPP Only vs. Full Model',
                             'rank_scatter_pen_vs_full.png')
        plot_rank_comparison(df_scores, 'rank_reward', 'rank_full',
                             'BERP Only vs. Full Model',
                             'rank_scatter_rew_vs_full.png')

        # 2. Score Distribution Plots
        score_cols = [COL_FORWARD, COL_PENALTY, COL_REWARD, COL_FULL]
        score_labels = ['Forward Only', 'Penalty Only', 'Reward Only', 'Full Model']
        plot_score_distribution(df_scores, score_cols, score_labels,
                                'Score Distributions Comparison',
                                'score_distribution_comparison.png')

        # 3. Rank Change Histograms
        plot_rank_change_histogram(df_scores, 'rank_forward', 'rank_penalty',
                                   'Rank Change Distribution (BEPP Only vs Forward Only)',
                                   'rank_change_hist_penalty.png')
        plot_rank_change_histogram(df_scores, 'rank_forward', 'rank_reward',
                                   'Rank Change Distribution (BERP Only vs Forward Only)',
                                   'rank_change_hist_reward.png')
        plot_rank_change_histogram(df_scores, 'rank_forward', 'rank_full',
                                   'Rank Change Distribution (Full Model vs Forward Only)',
                                   'rank_change_hist_full.png')

        print("\nVisualization script finished successfully.")
    else:
        print("\nVisualization script failed due to data loading errors.")

