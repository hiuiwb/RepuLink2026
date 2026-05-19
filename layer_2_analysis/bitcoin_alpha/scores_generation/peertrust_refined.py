#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements the PeerTrust algorithm based on the original paper:
"PeerTrust: supporting reputation-based trust for peer-to-peer electronic communities".

It computes trust scores for each peer using five key factors:
1. Feedback score
2. Feedback credibility
3. Transaction context factor
4. Community context factor (optional)
5. Transaction volume (implicitly embedded via counts)

The output is saved to peertrust_scores.csv with columns: dst, peertrust_score
"""

import pandas as pd
import numpy as np

# ========== Parameter Configuration ==========
input_file = "/ECShome/ww3y23/Github/Repulink/datasets/bitcoin_alpha.csv"
output_file = "peertrust_scores.csv"
# min_feedback = 5

# Context factor (e.g., time decay for transactions)
def context_factor(timestamp, current_time):
    # Simple exponential decay based on timestamp difference
    delta = current_time - timestamp
    return np.exp(-delta / 1e7)

def main():
    df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
    print("Original data:", df.shape)

    # Normalize ratings to [-1, 1]
    df['norm_rating'] = df['rating'] / 10.0

    # Compute credibility: average agreement with others' ratings to the same dst
    credibility = {}
    for peer in df['dst'].unique():
        peer_feedback = df[df['dst'] == peer]
        if len(peer_feedback) < 2:
            continue
        avg = peer_feedback['norm_rating'].mean()
        for _, row in peer_feedback.iterrows():
            diff = abs(row['norm_rating'] - avg)
            credibility[row['src']] = credibility.get(row['src'], 0) + (1 - diff)

    # Normalize credibility scores to [0, 1]
    if credibility:
        max_cred = max(credibility.values())
        for k in credibility:
            credibility[k] /= max_cred

    # Use latest timestamp to compute context factors
    current_time = df['timestamp'].max()

    # Compute PeerTrust score
    trust_scores = {}
    grouped = df.groupby('dst')
    for dst, group in grouped:
        if len(group) < min_feedback:
            continue

        trust_sum = 0
        weight_sum = 0

        for _, row in group.iterrows():
            f = row['norm_rating']
            c = credibility.get(row['src'], 1.0)
            t = context_factor(row['timestamp'], current_time)
            trust_sum += f * c * t
            weight_sum += c * t

        score = trust_sum / weight_sum if weight_sum > 0 else 0
        trust_scores[dst] = score

    df_out = pd.DataFrame(list(trust_scores.items()), columns=['dst', 'peertrust_score'])
    df_out.to_csv(output_file, index=False)
    print(f"Saved PeerTrust scores to {output_file}")

if __name__ == "__main__":
    main()
