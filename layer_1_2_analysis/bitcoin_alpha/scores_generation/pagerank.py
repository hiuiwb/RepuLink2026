#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
import numpy as np

def normalize(series):
    """Min-max normalization to scale endorsement scores between 0 and 1."""
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def main():
    # ========== File paths ==========
    interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_alpha.csv"
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
    output_file = "pagerank_hybrid.csv"
    lambda_weight = 0.5  # weighting factor between interaction and endorsement

    # ========== Load interaction data ==========
    try:
        df = pd.read_csv(interaction_file, header=None, names=["src", "dst", "rating", "timestamp"])
    except Exception as e:
        print("Failed to read interaction file: {}".format(e))
        return
    print("Interaction data loaded, shape:", df.shape)

    # ========== Build weighted directed graph ==========
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, rating = row["src"], row["dst"], row["rating"]
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += rating
        else:
            G.add_edge(src, dst, weight=rating)
    print("Graph built: nodes = {}, edges = {}".format(G.number_of_nodes(), G.number_of_edges()))

    # ========== Compute PageRank ==========
    pagerank_scores = nx.pagerank(G, alpha=0.85, weight='weight', max_iter=10000, tol=1e-3)
    df_pr = pd.DataFrame(pagerank_scores.items(), columns=["dst", "pagerank_score"])

    # ========== Load endorsement in-degree (Epinions) ==========
    try:
        df_endorse = pd.read_csv(endorsement_file, sep='\t', header=None, names=['src', 'dst'])
    except Exception as e:
        print("Failed to read endorsement file: {}".format(e))
        return
    in_deg = df_endorse['dst'].value_counts().reset_index()
    in_deg.columns = ['dst', 'endorse_count']
    in_deg['endorse_norm'] = normalize(in_deg['endorse_count'])

    # ========== Merge and compute Hybrid Score ==========
    df_merged = pd.merge(df_pr, in_deg[['dst', 'endorse_norm']], on='dst', how='left')
    df_merged['endorse_norm'] = df_merged['endorse_norm'].fillna(0)
    df_merged['hybrid_score'] = (
        lambda_weight * df_merged['pagerank_score'] +
        (1 - lambda_weight) * df_merged['endorse_norm']
    )

    # ========== Save result ==========
    df_merged[['dst', 'hybrid_score']].to_csv(output_file, index=False)
    print("Hybrid PageRank scores saved to", output_file)

if __name__ == "__main__":
    main()
