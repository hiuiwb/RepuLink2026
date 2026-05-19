#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates the EigenTrust scores and saves them to "eigentrust_scores.csv".
The input CSV file (data.csv) is assumed to have no header and columns:
src, dst, rating, timestamp.
The algorithm computes a local trust matrix from ratings (only positive scores),
normalizes it row-wise, and then applies power iteration to compute the global
eigenvector (trust/reputation scores). The output CSV contains two columns:
dst and eigentrust_score.
"""

import pandas as pd
import numpy as np

def load_data(input_file):
    # 读取数据，文件无表头，因此指定 header=None 和列名
    df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
    return df

def build_trust_matrix(df, epsilon_row=1e-10):
    """
    构造局部信任矩阵 T, T[i,j] = max(rating, 0)；
    对于同一对 (i,j)，评分累加。
    然后对每一行进行归一化，得到行随机矩阵 M.
    """
    # 取所有用户（合并 src 和 dst）
    users = pd.concat([df['src'], df['dst']]).unique()
    users.sort()  # 排序，便于索引一致性
    N = len(users)
    
    # 建立用户id与索引的映射字典
    user2idx = { user: idx for idx, user in enumerate(users) }
    
    # 初始化局部信任矩阵（累加所有正评分）
    T = np.zeros((N, N))
    
    # 遍历所有边，累加正反馈（取 max(rating,0)）
    for _, row in df.iterrows():
        i = user2idx[row['src']]
        j = user2idx[row['dst']]
        r = row['rating']
        T[i, j] += r

    # 对每一行进行归一化
    M = np.zeros_like(T)
    for i in range(N):
        row_sum = np.sum(np.maximum(T[i, :], 0))
        if row_sum > epsilon_row:
            M[i, :] = np.maximum(T[i, :], 0) / row_sum
        else:
            # 若这一行全为0，则赋予均匀分布
            M[i, :] = 0

    return M, users

def compute_repulink(M, tol=1e-3, max_iter=1000):
    """
    使用幂迭代法计算 EigenTrust 得分。
    注意：这里我们使用 M 的转置 (即 M^\top) 进行右乘，
    使得全局信任向量 r 满足 r = M^\top r, r 为概率分布向量。
    """
    N = M.shape[0]
    # 初始化为均匀分布
    r = np.ones(N) / N
    for iter in range(max_iter):
        r_new = M.T @ r
        # 归一化 r_new，确保总和为1
        r_new = r_new / (np.sum(r_new) + 1e-12)
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"Convergence reached after {iter+1} iterations.")
            return r_new
        r = r_new
    print("Max iterations reached without full convergence.")
    return r

def save_scores(users, scores, output_file):
    """
    将用户与其 EigenTrust 得分保存到 CSV 文件中。文件包含两列：dst, eigentrust_score
    """
    df_out = pd.DataFrame({
        "dst": users,
        "repulink_score": scores
    })
    df_out.to_csv(output_file, index=False)
    print(f"RepuLink scores saved to {output_file}.")

def main():
    # 参数设定：文件路径和参数
    input_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_otc.csv"  # 输入数据文件
    output_file = "repulink_scores.csv"   # 输出文件

    # 读取数据
    df = load_data(input_file)
    print("Input data shape:", df.shape)

    # 构造局部信任矩阵，并获取所有用户列表
    M, users = build_trust_matrix(df)
    print("Number of users:", len(users))

    # 计算 EigenTrust 得分（全局信任向量）
    scores = compute_repulink(M)
    
    # 保存到 CSV 文件中
    save_scores(users, scores, output_file)

if __name__ == "__main__":
    main()
