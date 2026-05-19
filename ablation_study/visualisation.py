#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script comparing RepuLink results across two datasets
(e.g., Bitcoin-OTC and Bitcoin-Alpha).

Loads results from specified scenarios (e.g., Forward Only, Full Model)
for each dataset, calculates ranks within each dataset, combines the data,
and generates plots comparing scores and rankings using hues/labels
to distinguish the datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import EngFormatter

# ============ Configuration ============

# --- Input Directories for the two datasets ---
# !!! CHANGE THESE PATHS to point to your actual result directories !!!
INPUT_DIR_DS1 = "./bitcoin_otc/repulink_ablation_study_output"
INPUT_DIR_DS2 = "./bitcoin_alpha/repulink_ablation_study_output"
DATASET_LABEL_DS1 = "Bitcoin-OTC"
DATASET_LABEL_DS2 = "Bitcoin-Alpha"

# --- File Names (Relative to the input directories above) ---
# Define the files needed for the comparisons
FILES_TO_LOAD = {
    'forward': "repulink_forward_only.csv",
    'penalty': "repulink_penalty_only.csv",
    'reward': "repulink_reward_only.csv",
    'full': "repulink_penalty_reward.csv"
}

# --- Column Names (Ensure these match the columns in your CSVs) ---
SCORE_COLS = {
    'forward': "repulink_forward_score",
    'penalty': "repulink_penalty_only_score",
    'reward': "repulink_reward_only_score",
    'full': "repulink_full_bepp_score"
}
COL_USER = "dst" # User ID column used for merging

# --- Output ---
OUTPUT_DIR_FIGS = "repulink_comparison_figures"
os.makedirs(OUTPUT_DIR_FIGS, exist_ok=True)
print(f"Figures will be saved in: {os.path.abspath(OUTPUT_DIR_FIGS)}")

# --- Plotting Parameters ---
FIG_DPI = 300
FONT_TITLE = FontProperties(weight='bold', size=35)
FONT_LABEL = FontProperties(weight='bold', size=35)
FONT_TICK = FontProperties(weight='bold', size=25)
legend_font = FontProperties(weight='bold', size=25)

# ============ Helper Functions ============

def load_and_process_dataset(input_dir, dataset_label, files_to_load, score_cols, user_col):
    """
    Loads specified score files for a single dataset, merges them,
    calculates ranks, and adds a dataset label.
    """
    print(f"\n--- Processing Dataset: {dataset_label} ---")
    print(f"Loading files from: {input_dir}")

    loaded_dfs = {}
    try:
        for key, filename in files_to_load.items():
            filepath = os.path.join(input_dir, filename)
            print(f"  Loading {key} data from: {filepath}")
            df = pd.read_csv(filepath)

            # Validate required columns
            required_score_col = score_cols.get(key)
            if not required_score_col:
                print(f"Error: Score column mapping missing for key '{key}'. Skipping file.")
                continue
            if user_col not in df.columns:
                print(f"Error: User column '{user_col}' not found in {filename}. Skipping file.")
                return None # Cannot proceed without user column
            if required_score_col not in df.columns:
                print(f"Error: Score column '{required_score_col}' not found in {filename}. Skipping file.")
                continue # Skip this specific file/score

            # Keep only user and the specific score column for merging
            loaded_dfs[key] = df[[user_col, required_score_col]].copy()

        if not loaded_dfs:
            print(f"Error: No data successfully loaded for {dataset_label}.")
            return None
        print("Files loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error: File not found during loading for {dataset_label}.")
        print(f"Missing file: {e.filename}")
        return None
    except Exception as e:
        print(f"An error occurred during file loading for {dataset_label}: {e}")
        return None

    # Merge the loaded dataframes for this dataset
    print("Merging dataframes for this dataset...")
    keys = list(loaded_dfs.keys())
    df_merged = loaded_dfs[keys[0]]
    # Rename score columns based on the key to avoid clashes during merge
    df_merged = df_merged.rename(columns={score_cols[keys[0]]: f"score_{keys[0]}"})

    for key in keys[1:]:
        if key in loaded_dfs: # Check if DF was loaded successfully
             df_to_merge = loaded_dfs[key]
             score_col_name = score_cols[key]
             df_to_merge = df_to_merge.rename(columns={score_col_name: f"score_{key}"})
             df_merged = pd.merge(df_merged, df_to_merge, on=user_col, how='inner') # Inner merge keeps common users

    print(f"Dataframes merged. Shape: {df_merged.shape}")

    # Add ranks for each score column (using the renamed columns like 'score_forward')
    print("Calculating ranks for this dataset...")
    rank_cols_map = {}
    for key in loaded_dfs.keys(): # Iterate through keys that were successfully loaded
        renamed_score_col = f"score_{key}"
        rank_col = f"rank_{key}"
        if renamed_score_col in df_merged.columns:
            df_merged[rank_col] = df_merged[renamed_score_col].rank(ascending=False, method='first')
            rank_cols_map[key] = rank_col # Store the name of the created rank column
        else:
             print(f"Warning: Renamed score column '{renamed_score_col}' not found after merge. Cannot calculate rank for '{key}'.")

    print("Ranks calculated.")

    # Add dataset label
    df_merged['dataset'] = dataset_label
    print(f"Finished processing {dataset_label}.")
    return df_merged


# --- Modified Plotting Functions ---

def plot_rank_comparison(df_combined, rank_col1, rank_col2, title, filename, label1, label2):
    """
    Generates a scatter plot comparing ranks from two scenarios,
    using hue to distinguish datasets.
    """
    print(f"Generating rank comparison plot: {title}")

    # Check if necessary columns exist
    required_cols = [rank_col1, rank_col2, 'dataset']
    if not all(col in df_combined.columns for col in required_cols):
        print(f"Error: Missing required columns ({required_cols}) for plot '{title}'. Skipping.")
        return

    plt.figure(figsize=(9, 6))

    # Scatter plot with hue
    scatter_plot = sns.scatterplot(data=df_combined, x=rank_col1, y=rank_col2, hue='dataset',
                                   alpha=0.7, s=15, edgecolor=None) # Use hue

    # Add y=x line for reference
    # Calculate max rank across both datasets for the relevant columns
    max_rank_x = df_combined[rank_col1].max()
    max_rank_y = df_combined[rank_col2].max()
    # Check if max ranks are NaN (can happen if columns are missing/empty)
    if pd.isna(max_rank_x) or pd.isna(max_rank_y):
        print(f"Warning: Cannot determine max rank for plot '{title}' due to NaN values. Skipping y=x line.")
        max_rank = 1 # Default fallback
    else:
        max_rank = max(max_rank_x, max_rank_y)
        max_rank = max(1, max_rank) # Ensure at least 1
        plt.plot([1, max_rank], [1, max_rank], color='red', linestyle='--', linewidth=3)


    plt.title(title, fontproperties=FONT_TITLE)
    plt.xlabel(f"Rank ({label1})", fontproperties=FONT_LABEL)
    plt.ylabel(f"Rank ({label2})", fontproperties=FONT_LABEL)
    # Get legend handles and labels from seaborn plot
    handles, labels = scatter_plot.get_legend_handles_labels()
    # Ensure there are handles/labels before creating legend
    if handles and labels:
         # Use the handles and labels provided by Seaborn
         plt.legend(handles=handles, labels=labels, prop=legend_font)
    else:
         print(f"Warning: No legend handles/labels generated by Seaborn for plot '{title}'.")


    plt.grid(True)
    plt.xlim(left=0.8, right=max_rank * 1.05)
    plt.ylim(bottom=0.8, top=max_rank * 1.05)

    # Apply tick font properties
    ax = plt.gca()

    # --- Apply EngFormatter for 'k'/'M' etc. notation ---
    # Use sep="" for no space, e.g., "1k" instead of "1 k"
    formatter_k = EngFormatter(sep="")
    ax.xaxis.set_major_formatter(formatter_k)
    ax.yaxis.set_major_formatter(formatter_k)

    for label in ax.get_xticklabels(): label.set_fontproperties(FONT_TICK)
    for label in ax.get_yticklabels(): label.set_fontproperties(FONT_TICK)
    # ax.tick_params(axis='x', labelrotation=45)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR_FIGS, filename)
    plt.savefig(filepath, dpi=FIG_DPI)
    print(f"Saved: {filepath}")
    plt.close()

def plot_score_distribution(df_combined, score_cols_map, title, filename):
    """
    Generates density plots comparing score distributions for multiple scenarios,
    using hue to distinguish datasets.
    score_cols_map: dict mapping scenario key (e.g., 'full') to its score column name (e.g., 'score_full')
    """
    print(f"Generating score distribution plot: {title}")

    # Check if necessary columns exist
    required_cols = list(score_cols_map.values()) + ['dataset']
    if not all(col in df_combined.columns for col in required_cols):
        print(f"Error: Missing required columns for plot '{title}'. Required: {required_cols}. Found: {df_combined.columns}. Skipping.")
        return

    num_scenarios = len(score_cols_map)
    plt.figure(figsize=(12, 7))

    plot_has_data = False
    # Define a color palette for scenarios if needed, otherwise seaborn handles dataset hue
    # scenario_palette = sns.color_palette("viridis", num_scenarios)

    # Instead of plotting all scores on one axis, let's melt the data
    # Or plot each scenario separately but overlay datasets
    for i, (key, score_col) in enumerate(score_cols_map.items()):
        label = key.capitalize() # e.g., 'Full'
        # Plotting each score type, with datasets distinguished by hue
        hist_plot = sns.kdeplot(data=df_combined, x=score_col, hue='dataset',
                                fill=True, alpha=0.3, lw=2, label=f"{label} ({DATASET_LABEL_DS1} vs {DATASET_LABEL_DS2})") # label might not be used directly if hue is set
        plot_has_data = True # Mark that we have plotted something

    if not plot_has_data:
        print(f"Warning: No data plotted for '{title}'.")
        plt.close()
        return

    plt.title(title, fontproperties=FONT_TITLE)
    plt.xlabel("Reputation Score", fontproperties=FONT_LABEL)
    plt.ylabel("Density", fontproperties=FONT_LABEL)

    # Create legend manually if needed, or rely on seaborn's hue legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles and labels:
        # Filter labels/handles if seaborn creates duplicates due to kdeplot loop
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        plt.legend(handles=unique_labels.values(), labels=unique_labels.keys(), prop=legend_font)
    else:
        print(f"Warning: No legend handles/labels generated for plot '{title}'.")


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


# def plot_rank_change_histogram(df_combined, rank_col_base, rank_col_compare, title, filename):
#     """
#     Generates histograms of rank changes, using hue to distinguish datasets.
#     """
#     print(f"Generating rank change histogram: {title}")

#     # Check if necessary columns exist
#     required_cols = [rank_col_base, rank_col_compare, 'dataset']
#     if not all(col in df_combined.columns for col in required_cols):
#         print(f"Error: Missing required columns ({required_cols}) for plot '{title}'. Skipping.")
#         return

#     # Calculate rank change (positive means rank improved, i.e., rank number decreased)
#     df_combined['rank_change'] = df_combined[rank_col_base] - df_combined[rank_col_compare]

#     plt.figure(figsize=(10, 6))

#     # Use histplot with hue
#     hist_plot = sns.histplot(data=df_combined, x='rank_change', hue='dataset',
#                              bins=50, kde=True, edgecolor=None, alpha=0.6,
#                              line_kws={'lw': 3})

#     # Add average lines per dataset
#     avg_changes = df_combined.groupby('dataset')['rank_change'].mean()
#     colors = sns.color_palette(n_colors=df_combined['dataset'].nunique())
#     dataset_labels = df_combined['dataset'].unique()
#     color_map = dict(zip(dataset_labels, colors))

#     # for label in dataset_labels:
#     #     if label in avg_changes:
#     #         avg_val = avg_changes[label]
#     #         plt.axvline(avg_val, color=color_map[label], linestyle='--', linewidth=2,
#     #                     label=label)
#     avg_change = 0
#     plt.axvline(avg_change, color='red', linestyle='--', linewidth=3)


#     plt.title(title, fontproperties=FONT_TITLE)
#     plt.xlabel("Rank Change (Base Rank - New Rank)", fontproperties=FONT_LABEL)
#     plt.ylabel("Frequency", fontproperties=FONT_LABEL)

#     # Get legend handles and labels
#     handles, labels = plt.gca().get_legend_handles_labels()
#     if handles and labels:
#          # Combine histplot hue legend with axvline labels
#          # Keep only unique labels
#          unique_labels_map = {}
#          for handle, label in zip(handles, labels):
#              # Avoid duplicate labels if histplot and axvline create similar ones
#              if label not in unique_labels_map:
#                  unique_labels_map[label] = handle
#          plt.legend(handles=unique_labels_map.values(), labels=unique_labels_map.keys(), prop=legend_font)
#     else:
#          print(f"Warning: No legend handles/labels generated for plot '{title}'.")


#     plt.grid(True)

#     # Apply tick font properties
#     ax = plt.gca()
#     for label in ax.get_xticklabels(): label.set_fontproperties(FONT_TICK)
#     for label in ax.get_yticklabels(): label.set_fontproperties(FONT_TICK)

#     plt.tight_layout()
#     filepath = os.path.join(OUTPUT_DIR_FIGS, filename)
#     plt.savefig(filepath, dpi=FIG_DPI)
#     print(f"Saved: {filepath}")
#     # Clean up added column
#     df_combined.drop(columns=['rank_change'], inplace=True)
#     plt.close()



def plot_rank_change_histogram(df_combined, rank_col_base, rank_col_compare, title, filename):
    """
    Generates histograms of rank changes, using hue to distinguish datasets.
    Draws a single reference line at x=0.
    """
    print(f"Generating rank change histogram: {title}")

    # Check if necessary columns exist
    required_cols = [rank_col_base, rank_col_compare, 'dataset']
    if not all(col in df_combined.columns for col in required_cols):
        print(f"Error: Missing required columns ({required_cols}) for plot '{title}'. Skipping.")
        return

    # Calculate rank change (positive means rank improved, i.e., rank number decreased)
    # Use .copy() to avoid SettingWithCopyWarning if df_combined is used later
    df_plot = df_combined.copy()
    df_plot['rank_change'] = df_plot[rank_col_base] - df_plot[rank_col_compare]

    plt.figure(figsize=(12, 7))

    # Use histplot with hue - Seaborn automatically handles colors and legend entries
    hist_plot = sns.histplot(data=df_plot, x='rank_change', hue='dataset',
                             bins=50, kde=True, edgecolor=None, alpha=0.6,
                             line_kws={'lw': 3})

    # --- Keep single vertical line at x=0 ---
    plt.axvline(0, color='red', linestyle='--', linewidth=3) # Draw line at zero rank change


    plt.title(title, fontproperties=FONT_TITLE)
    plt.xlabel("Rank Change (Base Rank - New Rank)", fontproperties=FONT_LABEL)
    plt.ylabel("Frequency", fontproperties=FONT_LABEL)

    # --- Corrected Legend Handling ---
    # Get handles and labels generated by seaborn's hue
    # Seaborn's histplot with hue often manages the legend well directly.
    # We just need to ensure the font is applied.
    current_legend = plt.gca().get_legend()
    if current_legend:
        plt.setp(current_legend.get_texts(), fontproperties=legend_font)
        plt.setp(current_legend.get_title(), fontproperties=legend_font) # Set title font too if needed
        current_legend.set_title(None) # Set legend title explicitly
    else:
         # Fallback if seaborn didn't create one automatically (less common with hue)
         handles, labels = plt.gca().get_legend_handles_labels()
         if handles and labels:
             unique_labels_map = {}
             for handle, label in zip(handles, labels):
                 if label and not label.startswith('_') and label not in unique_labels_map:
                     unique_labels_map[label] = handle
             if unique_labels_map:
                  plt.legend(handles=unique_labels_map.values(), labels=unique_labels_map.keys(), prop=legend_font)
             else:
                  print(f"Warning: No valid legend entries found after filtering for plot '{title}'.")
         else:
              print(f"Warning: No legend handles/labels generated for plot '{title}'.")


    plt.grid(True)

    # Apply tick font properties
    ax = plt.gca()
    for label in ax.get_xticklabels(): label.set_fontproperties(FONT_TICK)
    for label in ax.get_yticklabels(): label.set_fontproperties(FONT_TICK)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR_FIGS, filename)
    plt.savefig(filepath, dpi=FIG_DPI)
    print(f"Saved: {filepath}")
    # We used a copy df_plot, so no need to drop column from df_combined
    plt.close()

# ============ Main Execution ============

if __name__ == "__main__":
    print("--- RepuLink Cross-Dataset Visualization ---")

    # Load and process each dataset
    df_ds1 = load_and_process_dataset(INPUT_DIR_DS1, DATASET_LABEL_DS1, FILES_TO_LOAD, SCORE_COLS, COL_USER)
    df_ds2 = load_and_process_dataset(INPUT_DIR_DS2, DATASET_LABEL_DS2, FILES_TO_LOAD, SCORE_COLS, COL_USER)

    # Combine datasets if both loaded successfully
    if df_ds1 is not None and df_ds2 is not None:
        df_combined = pd.concat([df_ds1, df_ds2], ignore_index=True)
        print(f"\n--- Combined data from {DATASET_LABEL_DS1} and {DATASET_LABEL_DS2} ---")
        print(f"Total entries: {len(df_combined)}")
        print(f"Columns available: {df_combined.columns.tolist()}")

        # --- Generate Plots ---
        print("\n--- Generating Plots ---")

        # Define maps for columns used in plots (using renamed columns)
        score_cols_renamed = {key: f"score_{key}" for key in FILES_TO_LOAD.keys()}
        rank_cols = {key: f"rank_{key}" for key in FILES_TO_LOAD.keys()}

        # --- 1. Rank Comparison Scatter Plots (5 plots total) ---
        print("\nGenerating Rank Comparison plots...")
        # Plot 1: Fwd vs Pen
        if all(col in df_combined.columns for col in [rank_cols['forward'], rank_cols['penalty']]):
            plot_rank_comparison(df_combined, rank_cols['forward'], rank_cols['penalty'],
                                 'Forward Only vs. With BEPP',
                                 'rank_scatter_fwd_vs_pen_datasets.pdf',
                                 'Forward Only', 'With BEPP')
        else: print("Skipping Rank Scatter (Fwd vs Pen): Missing columns.")

        # Plot 2: Fwd vs Rew
        if all(col in df_combined.columns for col in [rank_cols['forward'], rank_cols['reward']]):
             plot_rank_comparison(df_combined, rank_cols['forward'], rank_cols['reward'],
                                  'Forward Only vs. With BERP',
                                  'rank_scatter_fwd_vs_rew_datasets.pdf',
                                  'Forward Only', 'With BERP')
        else: print("Skipping Rank Scatter (Fwd vs Rew): Missing columns.")

        # Plot 3: Fwd vs Full
        if all(col in df_combined.columns for col in [rank_cols['forward'], rank_cols['full']]):
             plot_rank_comparison(df_combined, rank_cols['forward'], rank_cols['full'],
                                  'Forward Only vs. Full Model',
                                  'rank_scatter_fwd_vs_full_datasets.pdf',
                                  'Forward Only', 'Full Model')
        else: print("Skipping Rank Scatter (Fwd vs Full): Missing columns.")

        # Plot 4: Pen vs Full <--- ADDED THIS CALL
        if all(col in df_combined.columns for col in [rank_cols['penalty'], rank_cols['full']]):
             plot_rank_comparison(df_combined, rank_cols['penalty'], rank_cols['full'],
                                  'With BEPP vs. Full Model',
                                  'rank_scatter_pen_vs_full_datasets.pdf', # Added filename
                                  'With BEPP', 'Full Model')
        else: print("Skipping Rank Scatter (Pen vs Full): Missing columns.") # Added check message

        # Plot 5: Rew vs Full <--- ADDED THIS CALL
        if all(col in df_combined.columns for col in [rank_cols['reward'], rank_cols['full']]):
             plot_rank_comparison(df_combined, rank_cols['reward'], rank_cols['full'],
                                  'With BERP vs. Full Model',
                                  'rank_scatter_rew_vs_full_datasets.pdf', # Added filename
                                  'With BERP', 'Full Model')
        else: print("Skipping Rank Scatter (Rew vs Full): Missing columns.") # Added check message


        # --- 2. Score Distribution Plots (1 plot total) ---
        print("\nGenerating Score Distribution plot...")
        valid_score_cols_map = {k: v for k, v in score_cols_renamed.items() if v in df_combined.columns}
        if valid_score_cols_map:
             plot_score_distribution(df_combined, valid_score_cols_map,
                                     'Score Distributions Comparison',
                                     'score_distribution_comparison_datasets.pdf')
        else:
            print("Skipping score distribution plot: No valid score columns found.")


        # --- 3. Rank Change Histograms (3 plots total) ---
        print("\nGenerating Rank Change Histograms...")
        # Plot 1: Pen vs Fwd
        if all(col in df_combined.columns for col in [rank_cols['forward'], rank_cols['penalty']]):
            plot_rank_change_histogram(df_combined, rank_cols['forward'], rank_cols['penalty'],
                                       'With BEPP vs Forward Only',
                                       'rank_change_hist_penalty_datasets.pdf')
        else: print("Skipping Rank Change Hist (Pen vs Fwd): Missing columns.")

        # Plot 2: Rew vs Fwd
        if all(col in df_combined.columns for col in [rank_cols['forward'], rank_cols['reward']]):
             plot_rank_change_histogram(df_combined, rank_cols['forward'], rank_cols['reward'],
                                        'With BERP vs Forward Only',
                                        'rank_change_hist_reward_datasets.pdf')
        else: print("Skipping Rank Change Hist (Rew vs Fwd): Missing columns.")

        # Plot 3: Full vs Fwd
        if all(col in df_combined.columns for col in [rank_cols['forward'], rank_cols['full']]):
             plot_rank_change_histogram(df_combined, rank_cols['forward'], rank_cols['full'],
                                        'Full Model vs Forward Only',
                                        'rank_change_hist_full_datasets.pdf')
        else: print("Skipping Rank Change Hist (Full vs Fwd): Missing columns.")


        print("\nVisualization script finished successfully.")
    else:
        print("\nVisualization script failed: Could not load data for one or both datasets.")