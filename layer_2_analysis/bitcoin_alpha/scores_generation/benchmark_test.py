#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script processes a trust dataset (e.g., Bitcoin OTC or Epinions data)
and generates ground truth labels for high-reputation and low-reputation users.
The input CSV file is expected to have the columns: src, dst, rating, timestamp.
High-reputation users are defined as those with a high average rating (with sufficient feedback),
and low-reputation users are those with a low average rating.
"""

import pandas as pd
import numpy as np

def main():
    # ============ Parameter Settings ================
    input_file = "/ECShome/ww3y23/Github/Repulink/datasets/bitcoin_alpha.csv"  # Input CSV file path
    output_high = "high_reputation.csv"   # Output file for high-reputation users
    output_low = "low_reputation.csv"     # Output file for low-reputation users
    min_feedback = 10                     # Minimum number of feedback for a user to be considered
    top_percent = 0.2                     # Top 10% users are considered high-reputation
    bottom_percent = 0.2                  # Bottom 10% users are considered low-reputation

    # ============ Data Loading ========================
    try:
        # 读取数据，设定无标题，且数据列依次为 src, dst, rating, timestamp
        df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
    except Exception as e:
        print("Error reading file {}: {}".format(input_file, e))
        return

    print("Input data shape:", df.shape)

    # ============ Compute User Reputation Statistics ============
    # Group feedback by target user; 此处目标用户为 "dst"
    grouped = df.groupby('dst')
    
    # 统计每个用户：反馈次数、平均评分、总评分
    stats = grouped['rating'].agg(['count', 'mean', 'sum']).reset_index()
    stats.rename(columns={'count': 'feedback_count', 'mean': 'avg_rating', 'sum': 'total_rating'}, inplace=True)
    
    # 过滤掉反馈次数不足的用户
    stats_filtered = stats[stats['feedback_count'] >= min_feedback].copy()
    print("Number of users with sufficient feedback:", len(stats_filtered))
    
    # ============ Labeling High/Low Reputation Users ============
    # 按平均评分降序排序
    stats_sorted = stats_filtered.sort_values(by='avg_rating', ascending=False).reset_index(drop=True)
    n = len(stats_sorted)
    
    # 计算高信誉和低信誉用户数量
    top_n = int(np.ceil(n * top_percent))
    bottom_n = int(np.ceil(n * bottom_percent))
    
    # 将前 top_n 用户标记为 high，将后 bottom_n 用户标记为 low，其它为 neutral
    stats_sorted['label'] = 'neutral'
    stats_sorted.loc[:top_n-1, 'label'] = 'high'
    stats_sorted.loc[n - bottom_n:, 'label'] = 'low'
    
    high_users = stats_sorted[stats_sorted['label'] == 'high']
    low_users = stats_sorted[stats_sorted['label'] == 'low']
    
    print("High reputation users:", len(high_users))
    print("Low reputation users:", len(low_users))
    
    # ============ Output the Results ================
    high_users.to_csv(output_high, index=False)
    low_users.to_csv(output_low, index=False)
    
    print("Ground truth labels generated and saved to {} and {}.".format(output_high, output_low))

if __name__ == "__main__":
    main()
