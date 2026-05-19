#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def main():
    # ============ 参数设置 ==========================
    interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_alpha.csv"     # interaction layer 数据（含 rating）
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"        # endorsement layer 数据（仅 src, dst）
    output_high = "groundtruth_high.csv"
    output_low = "groundtruth_low.csv"
    lambda_weight = 0.5        # 加权系数
    min_feedback = 10
    top_percent = 0.2
    bottom_percent = 0.2

    # ============ Step 1: 读取 interaction layer 数据 ================
    df_inter = pd.read_csv(interaction_file, header=None, names=["src", "dst", "rating", "timestamp"])
    avg_ratings = df_inter.groupby("dst")["rating"].agg(["mean", "count"]).reset_index()
    avg_ratings = avg_ratings.rename(columns={"dst": "user", "mean": "avg_rating", "count": "feedback_count"})
    avg_ratings = avg_ratings[avg_ratings["feedback_count"] >= min_feedback]

    # ============ Step 2: 读取 endorsement layer 数据 ==================
    df_endorse = pd.read_csv(endorsement_file, sep="\t", header=None, names=["src", "dst"])
    endorse_count = df_endorse["dst"].value_counts().reset_index()
    endorse_count.columns = ["user", "endorse_count"]

    # ============ Step 3: 合并并归一化 ====================
    merged = pd.merge(avg_ratings, endorse_count, on="user", how="left").fillna(0)
    merged["endorse_norm"] = (merged["endorse_count"] - merged["endorse_count"].min()) / \
                              (merged["endorse_count"].max() - merged["endorse_count"].min() + 1e-6)

    # ============ Step 4: 计算混合分数 =====================
    merged["hybrid_score"] = lambda_weight * merged["avg_rating"] + (1 - lambda_weight) * merged["endorse_norm"]

    # ============ Step 5: 排序与生成标签 ===================
    merged = merged.sort_values(by="hybrid_score", ascending=False).reset_index(drop=True)
    n = len(merged)
    top_n = int(np.ceil(n * top_percent))
    bottom_n = int(np.ceil(n * bottom_percent))

    merged["label"] = "neutral"
    merged.loc[:top_n-1, "label"] = "high"
    merged.loc[n-bottom_n:, "label"] = "low"

    merged[merged["label"] == "high"].to_csv(output_high, index=False)
    merged[merged["label"] == "low"].to_csv(output_low, index=False)
    print(f"Saved high-reputation ground truth to: {output_high}")
    print(f"Saved low-reputation ground truth to: {output_low}")

if __name__ == "__main__":
    main()
