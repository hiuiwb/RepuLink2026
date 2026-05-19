#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

# ============ Load Interaction Layer ============
def load_interaction_data(input_file):
    df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
    users = pd.concat([df['src'], df['dst']]).unique()
    users.sort()
    user2idx = {u: i for i, u in enumerate(users)}
    N = len(users)
    C = np.zeros((N, N))
    for _, row in df.iterrows():
        i, j = user2idx[row["src"]], user2idx[row["dst"]]
        C[i, j] += row["rating"]
    for i in range(N):
        row_sum = np.sum(np.maximum(C[i], 0))
        if row_sum > 0:
            C[i] = np.maximum(C[i, :], 0) / row_sum
    return C, users, user2idx

# ============ Load Endorsement Layer ============
def load_endorsement_data(endorsement_file, user2idx, N):
    df = pd.read_csv(endorsement_file, sep="\t", header=None, names=["src", "dst"])
    F = np.zeros((N, N))
    for _, row in df.iterrows():
        if row["src"] in user2idx and row["dst"] in user2idx:
            i, j = user2idx[row["src"]], user2idx[row["dst"]]
            F[i, j] = 1
    for i in range(N):
        row_sum = F[i].sum()
        if row_sum > 0:
            F[i] /= row_sum
    return F

# ============ Normalize Columns ============
def column_normalize(W, epsilon=1e-12):
    for j in range(W.shape[1]):
        col_sum = np.sum(W[:, j])
        if col_sum > epsilon:
            W[:, j] /= col_sum
    return W

# ============ Compute RepuLink Hybrid ============
def compute_repulink_hybrid(C, F, alpha=0.8, tol=1e-20, max_iter=1000):
    N = C.shape[0]
    W = alpha * C.T + (1 - alpha) * F.T
    W = column_normalize(W)

    r = np.ones(N) / N
    residuals = []

    for iteration in range(max_iter):
        r_new = W @ r
        r_new = np.maximum(r_new, 0)
        r_new = r_new / (np.sum(r_new) + 1e-12)

        residual = np.linalg.norm(r_new - r, 1)
        residuals.append(residual)

        if residual < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        r = r_new

    # 保存 residual 曲线
    pd.DataFrame({"iteration": list(range(1, len(residuals)+1)), "residual": residuals}) \
      .to_csv("repulink_hybrid_residuals.csv", index=False)

    return r, residuals

# ============ Save Final Scores ============
def save_scores(users, scores, output_file):
    df_out = pd.DataFrame({
        "dst": users,
        "repulink_score": scores
    })
    df_out.to_csv(output_file, index=False)
    print(f"RepuLink scores saved to {output_file}.")

# # ============ Plot Residual Curve ============
# def plot_residual_curve(residuals, output_fig):
#     plt.figure(figsize=(8,6))
#     plt.plot(range(1, len(residuals)+1), residuals, linewidth=1)
#     plt.xlabel("Iteration", fontsize=20)
#     plt.ylabel("Residual (L1 norm)", fontsize=20)
#     plt.yscale('log')
#     plt.title("RepuLink Convergence Curve", fontsize=20)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(output_fig, dpi=300)
#     print(f"Residual curve saved to {output_fig}")
#     plt.close()

# ============ Plot Residual Curve (Optimized) ============
def plot_residual_curve(residuals, output_fig, convergence_tol=1e-6):
    """
    绘制 RepuLink 收敛曲线图 (风格与 ROC 示例匹配)。

    参数:
        residuals (list): 每次迭代的残差列表 (L1 范数)。
        output_fig (str): 输出图像文件的路径。
        convergence_tol (float): 收敛阈值，用于在图上绘制参考线。
    """
    plt.figure(figsize=(8, 6)) # 保持与 ROC 示例一致的画布大小

    # 定义字体属性 (参考 ROC 示例)
    label_font = FontProperties()
    label_font.set_weight('bold')
    label_font.set_size(30)

    title_font = FontProperties()
    title_font.set_weight('bold')
    title_font.set_size(30) # 与 ROC 示例标题字号一致

    tick_font = FontProperties()
    tick_font.set_weight('bold')
    # 注意：刻度标签通常不建议设置过大字号，这里保持默认或稍作调整
    tick_font.set_size(25) # 如果需要完全一致，取消注释，但可能不美观

    legend_font = FontProperties()
    legend_font.set_weight('bold')
    legend_font.set_size(25) 
    # 绘制残差曲线
    plt.plot(
        range(1, len(residuals) + 1),
        residuals,
        linestyle='-',
        linewidth=3,         # 匹配 ROC 示例的线宽
        label='L1 Residual'
    )

    # 在收敛阈值处添加一条水平虚线作为参考
    if convergence_tol:
        plt.axhline(
            y=convergence_tol,
            color='red', # 使用黑色虚线，与 ROC 示例的 Random 线一致
            linestyle='--',
            linewidth=3, # 可以调整线宽
            label=f'Tolerance ({convergence_tol:.1e})'
        )

    # 设置 Y 轴为对数刻度
    plt.yscale('log')

    # 应用字体属性设置标题和标签
    plt.title("Bitcoin Alpha", fontproperties=title_font)
    plt.xlabel("Iteration", fontproperties=label_font)
    plt.ylabel("Residual (L1 norm)", fontproperties=label_font) # Y 轴标签也应用字体

    # 应用字体属性设置刻度 (可选，可能影响美观)
    plt.xticks(fontproperties=tick_font)
    plt.yticks(fontproperties=tick_font)
    # 或者只调整大小
    # plt.tick_params(axis='both', which='major', labelsize=12) # 调整为更合适的刻度字号

    # Y 轴刻度格式化 (保留科学计数法)
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    # 显示网格
    plt.grid(True)

    # 应用字体属性设置图例
    plt.legend(loc="upper right", prop=legend_font) # 调整位置到右上角可能更合适

    # 自动调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"Residual curve saved to {output_fig}")
    plt.close() # 关闭图像



# ============ Main Function ============
def main():
    interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_alpha.csv"
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
    output_file = "repulink_hybrid_colnorm.csv"
    residual_plot = "repulink_hybrid_residuals.pdf"
    alpha = 0.8

    C, users, user2idx = load_interaction_data(interaction_file)
    F = load_endorsement_data(endorsement_file, user2idx, len(users))
    scores, residuals = compute_repulink_hybrid(C, F, alpha=alpha)
    save_scores(users, scores, output_file)
    plot_residual_curve(residuals, residual_plot)

if __name__ == "__main__":
    main()
