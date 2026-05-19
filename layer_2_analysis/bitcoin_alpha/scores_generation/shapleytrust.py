#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def load_trust_matrix(filename):
    df = pd.read_csv(filename, header=None, names=["src", "dst", "rating", "timestamp"])
    users = sorted(set(df["src"]).union(set(df["dst"])))
    user_to_idx = {u: i for i, u in enumerate(users)}
    N = len(users)

    trust_matrix = np.zeros((N, N))
    feedback_count = {}

    for _, row in df.iterrows():
        i, j, rating = row["src"], row["dst"], row["rating"]
        i_idx, j_idx = user_to_idx[i], user_to_idx[j]

        key = (i_idx, j_idx)
        if key not in feedback_count:
            feedback_count[key] = {"ng": 0, "nn": 0, "nb": 0}

        if rating > 0:
            feedback_count[key]["ng"] += 1
        elif rating == 0:
            feedback_count[key]["nn"] += 1
        else:
            feedback_count[key]["nb"] += 1

    for (i, j), count in feedback_count.items():
        total = count["ng"] + count["nn"] + count["nb"]
        if total > 0:
            score = (count["ng"] + 0.5 * count["nn"]) / total
            trust_matrix[i][j] = score

    return trust_matrix, users

def compute_marginal(args):
    cij, user_idx, N, num_samples = args
    contribs = []

    for _ in range(num_samples):
        perm = np.random.permutation(N)
        idx = np.where(perm == user_idx)[0][0]
        coalition = list(perm[:idx])
        coalition_with_user = coalition + [user_idx]

        value_with = cij[np.ix_(coalition_with_user, coalition_with_user)].sum()
        value_without = cij[np.ix_(coalition, coalition)].sum()
        contribs.append(value_with - value_without)

    return np.mean(contribs)

def main():
    input_file = "../datasets/bitcoin_alpha.csv"
    output_file = "shapleytrust_scores.csv"
    num_samples = 500
    num_processes = 40

    print("Loading trust matrix...")
    cij, users = load_trust_matrix(input_file)
    N = len(users)

    print("Estimating Shapley values...")
    args = [(cij, i, N, num_samples) for i in range(N)]

    with mp.Pool(num_processes) as pool:
        shapley_scores = list(tqdm(pool.imap(compute_marginal, args), total=N))

    df_out = pd.DataFrame({
        "dst": users,
        "shapleytrust_score": shapley_scores
    })
    df_out.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
