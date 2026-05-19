#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx

def main():
    # ================== 参数设置 =====================
    input_file = "/ECShome/ww3y23/Github/Repulink/datasets/bitcoin_alpha.csv"
    output_file = "pagerank_scores.csv"

    # ============ 读取数据 ===========================
    try:
        df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
    except Exception as e:
        print("读取文件失败: {}".format(e))
        return

    print("数据读取完成，数据维度:", df.shape)

    # ============ 构建图 =============================
    # 创建有向图（带权边）
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src = row["src"]
        dst = row["dst"]
        rating = row["rating"]
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += rating
        else:
            G.add_edge(src, dst, weight=rating)

    print("图构建完成，节点数: {}, 边数: {}".format(G.number_of_nodes(), G.number_of_edges()))

    # ============ 计算 PageRank =======================
    # 可调 damping 参数，默认是 0.85
    pagerank_scores = nx.pagerank(G, alpha=0.85, weight='weight', max_iter=10000, tol=1e-3)

    # 转换为 DataFrame 并保存
    result_df = pd.DataFrame(pagerank_scores.items(), columns=["dst", "pagerank_score"])
    result_df.sort_values(by="pagerank_score", ascending=False, inplace=True)
    result_df.to_csv(output_file, index=False)

    print("PageRank 分数已保存至", output_file)

if __name__ == "__main__":
    main()
