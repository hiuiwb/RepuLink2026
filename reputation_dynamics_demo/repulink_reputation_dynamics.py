#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time-evolving reputation dynamics demo for RepuLink.

This script highlights RepuLink's signature mechanism: backward accountability.
Five nodes are configured with intentional roles so that the impact of BERP
(Backward Endorsement Reward Propagation) and BEPP (Backward Endorsement Penalty
Propagation) is directly visible:

  - A_good           : 80% positive feedback (well-performing actor)
  - B_endorser_of_A  : 60% positive (decent own behavior); only endorses A
                       Expected: rises ABOVE E_baseline due to BERP
  - C_bad            : 30% positive feedback (misbehaving actor)
  - D_endorser_of_C  : 60% positive (decent own behavior); only endorses C
                       Expected: falls BELOW E_baseline due to BEPP
  - E_baseline       : 60% positive; endorses nobody
                       Control baseline — B and D differ from E only due to
                       backward propagation from their endorsement choices

Display interpretation (display values are sqrt-compressed for clarity):
  - > 0.8        : HIGH reputation
  - 0.5 to 0.8   : MEDIUM reputation
  - < 0.5        : LOW reputation

Initial reputation is set to 0.8 (display-scaled) and tracked over 30 timeslots.
"""

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

from core.config import (EPSILON, ALPHA, BETA, GAMMA, MAX_ITER,
                         CONVERGENCE_THRESHOLD, LAMBDA)
from core.reputation_forward_propagation import ReputationForwardPropagation
from core.reputation_backward_propagation import ReputationBackwardPropagation
from core.reputation_normalisation import ReputationNormalizer
from core.trustworthiness import TrustworthinessCalculator
from core.endorsement_penalty import EndorsementPenalty
from core.endorsement_reward import EndorsementReward
from core.endorsement_manager import EndorsementManager
from core.node_manager import NodeManager
from core.interaction_manager import InteractionManager


# ============ Configuration ============
NUM_TIMESLOTS = 30
MEAN_EVENTS_PER_TIMESLOT = 50  # mean events per timeslot (count is randomized)
EVENTS_STD = 14                 # std dev of per-timeslot event count
WARMUP_EVENTS = 300             # pre-loaded events so cumulative counts don't start from zero
INITIAL_REPUTATION = 0.8        # Display-scaled initial value

# Per-timeslot noise on each node's positive feedback rate. Each timeslot
# samples a fresh rate around the baseline, simulating "good days" and
# "bad days" — this is what produces visible wobbles in the reputation
# trajectory instead of perfectly monotonic curves.
POS_RATE_STD = 0.10

# When True, the endorsement matrix F is held fixed across timeslots so that
# the BERP/BEPP effect on B and D is clearly visible. With F updating
# (default RepuLink behavior), endorsers gradually withdraw their support
# from misbehaving nodes, causing the BEPP penalty to self-extinguish.
FIX_ENDORSEMENT_MATRIX = True

# Momentum smoothing on the reputation update across timeslots:
#   R_t = DEMO_MOMENTUM * R_{t-1} + (1 - DEMO_MOMENTUM) * R_new
# This produces a smooth approach to the asymptotic reputation while still
# letting per-timeslot fluctuations show through as visible wobble. The
# fixed point of the update is unchanged (R* = R* at equilibrium), so the
# BERP/BEPP demonstration is preserved — only the trajectory is smoothed.
DEMO_MOMENTUM = 0.78

# β and λ control penalty/reward signal saturation. Calibrated so that the
# signals stay sensitive to the DIFFERENCE between well-performing and
# misbehaving nodes throughout the 30-timeslot run.
# γ is increased to amplify BERP/BEPP magnitudes for clear visualization.
# α is increased to give more weight to the interaction trust matrix so that
# misbehaving nodes (whose interaction trust collapses to 0) cannot ride on
# their endorsers' reputation alone.
DEMO_ALPHA = 0.8
DEMO_BETA = 0.0065
DEMO_LAMBDA = 0.003
DEMO_GAMMA = 0.28

OUTPUT_DIR = "reputation_dynamics_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Node roles: each tuple is (label, positive_feedback_rate)
NODES = ["A_good", "B_endorser_of_A", "C_bad", "D_endorser_of_C", "E_baseline",
         "F_falling", "G_rising"]
N = len(NODES)

# Static baseline positive rate, used by warmup (no t) and by nodes whose
# behavior is constant. Nodes with time-varying behavior override this via
# TIME_VARYING_RATE during the simulated timeslots.
POSITIVE_RATE = {
    "A_good":           0.80,
    "B_endorser_of_A":  0.60,
    "C_bad":            0.30,
    "D_endorser_of_C":  0.60,
    "E_baseline":       0.60,
    "F_falling":        0.55,  # warmup baseline; per-slot rate decays during the run
    "G_rising":         0.55,  # warmup baseline; per-slot rate rises during the run
}

# Linearly-evolving per-timeslot positive rate (start_rate, end_rate).
# F starts moderate then drifts toward poor behavior; G starts moderate then
# drifts toward strong behavior. Combined with extra per-node noise this
# produces an "unstable but trending" curve.
TIME_VARYING_RATE = {
    "F_falling": (0.65, 0.52),
    "G_rising":  (0.55, 0.72),
}

# Per-node override of POS_RATE_STD. F and G are noisier than the static
# nodes so their unstable behavior is visually obvious.
NODE_POS_RATE_STD = {
    "F_falling": 0.15,
    "G_rising":  0.15,
}

# Per-node override of the initial display value at t=0. Nodes not listed
# default to INITIAL_REPUTATION (0.8). The initial L1 reputation vector is
# computed by squaring these displays and L1-normalizing.
INITIAL_DISPLAY = {
    "F_falling": 0.90,
    "G_rising":  0.45,
}


def get_positive_rate(node, t):
    """Returns the positive feedback rate for `node` at timeslot `t`.

    If t is None (warmup), uses the static POSITIVE_RATE. Otherwise, if the
    node has a time-varying entry, linearly interpolates between start_rate
    and end_rate over [1, NUM_TIMESLOTS].
    """
    if t is not None and node in TIME_VARYING_RATE:
        start, end = TIME_VARYING_RATE[node]
        denom = max(1, NUM_TIMESLOTS - 1)
        progress = (t - 1) / denom
        return start + (end - start) * progress
    return POSITIVE_RATE[node]


def generate_interactions_for_timeslot(nodes, t=None, mean_events=MEAN_EVENTS_PER_TIMESLOT,
                                        std=EVENTS_STD, pos_rate_std=POS_RATE_STD):
    """Generates a randomized number of interactions per timeslot.

    Two sources of per-timeslot randomness:
      1. Event count is sampled from a normal distribution around the mean,
         so the volume of activity varies from slot to slot.
      2. Each node's positive feedback rate is resampled around its baseline,
         creating "good days" and "bad days" that produce visible wobble in
         the reputation trajectories instead of perfectly monotonic curves.

    If `t` is provided and the node appears in TIME_VARYING_RATE, the
    baseline rate evolves linearly across the run.
    """
    interactions = []
    node_indices = list(range(len(nodes)))
    if len(nodes) < 2:
        return pd.DataFrame(interactions)
    num_events = max(15, int(np.random.normal(mean_events, std)))
    # Resample each node's per-timeslot positive rate around its (possibly
    # time-varying) baseline. Nodes in NODE_POS_RATE_STD use a per-node std.
    timeslot_pos_rate = {}
    for n in nodes:
        base = get_positive_rate(n, t)
        # When pos_rate_std == 0 we want a fully deterministic run (e.g. for
        # warmup), so ignore the per-node overrides as well.
        if pos_rate_std == 0:
            node_std = 0.0
        else:
            node_std = NODE_POS_RATE_STD.get(n, pos_rate_std)
        timeslot_pos_rate[n] = float(np.clip(np.random.normal(base, node_std), 0.05, 0.95))
    for _ in range(num_events):
        src_idx, dst_idx = np.random.choice(node_indices, 2, replace=False)
        src, dst = nodes[src_idx], nodes[dst_idx]
        p_pos = timeslot_pos_rate[dst]
        rating = int(np.random.choice([1, -1], p=[p_pos, 1.0 - p_pos]))
        interactions.append({
            "src": src,
            "dst": dst,
            "rating": rating,
            "timestamp": int(time.time()),
            "month": time.localtime().tm_mon,
        })
    return pd.DataFrame(interactions)


def build_initial_endorsement_matrix(node_to_index):
    """
    Builds an endorsement network designed to isolate BERP/BEPP effects.

    Edges (each row = endorser, each col = endorsee):
      - B_endorser_of_A --> A_good   (B's only outgoing endorsement)
      - D_endorser_of_C --> C_bad    (D's only outgoing endorsement)

    A_good, C_bad, E_baseline have NO outgoing endorsements. This isolation
    ensures B's reputation change comes purely from BERP (rewarding the endorser
    of a well-performing node) and D's change comes purely from BEPP.
    """
    E = np.zeros((N, N), dtype=np.float32)
    E[node_to_index["B_endorser_of_A"], node_to_index["A_good"]] = 1.0
    E[node_to_index["D_endorser_of_C"], node_to_index["C_bad"]] = 1.0
    np.fill_diagonal(E, 0)
    return E


def main():
    np.random.seed(42)

    nodes = NODES
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    index_to_node = {idx: node for node, idx in node_to_index.items()}

    node_manager = NodeManager()
    node_manager.load_nodes(nodes)

    E_initial = build_initial_endorsement_matrix(node_to_index)
    print("=== Initial Endorsement Matrix ===")
    print(pd.DataFrame(E_initial, index=nodes, columns=nodes))

    endorsement_manager = EndorsementManager(E_initial, nodes, node_to_index)

    initial_interactions_df = pd.DataFrame(columns=["src", "dst", "rating", "timestamp", "month"])
    interaction_manager = InteractionManager(initial_interactions_df, nodes, node_to_index)

    # Pre-warm interaction history so cumulative counts don't start from zero.
    # Without this warmup, the very first timeslot's small cumulative counts
    # cause an outsized 1/t-shaped reputation jump that drowns the rest of
    # the dynamics. With warmup, each subsequent timeslot is a small relative
    # perturbation, producing smooth and uniformly-distributed changes.
    print(f"Pre-warming interaction history with ~{WARMUP_EVENTS} events...")
    warmup_df = generate_interactions_for_timeslot(nodes, t=None,
                                                   mean_events=WARMUP_EVENTS,
                                                   std=0, pos_rate_std=0)
    for _, row in warmup_df.iterrows():
        interaction_manager.add_interaction(
            row["src"], row["dst"], row["rating"], row["timestamp"], row["month"]
        )

    # Build initial R from per-node desired displays. Since
    #     display = INITIAL_REPUTATION * sqrt(R * N)
    # solving for R gives R proportional to (display / INITIAL_REPUTATION)²,
    # which we then L1-normalize so the algorithm's invariants hold.
    raw_init = np.array(
        [(INITIAL_DISPLAY.get(n, INITIAL_REPUTATION) / INITIAL_REPUTATION) ** 2
         for n in nodes],
        dtype=np.float32,
    )
    R_current = raw_init / raw_init.sum()
    normalizer = ReputationNormalizer(epsilon=EPSILON)
    R_current = normalizer.normalize(R_current)

    # Display rescaling: sqrt-compress raw L1-normalized values so that
    #   - uniform 1/N maps to INITIAL_REPUTATION (= 0.8)
    #   - the dynamic range is compressed (high < 1.2, low > 0.3 typically)
    # Formula: display = INITIAL_REPUTATION * sqrt(raw * N)
    def to_display(raw_vec):
        return INITIAL_REPUTATION * np.sqrt(np.maximum(raw_vec, 0) * N)

    history = []
    # t=0 is the initial state: every node starts at uniform 1/N (display 0.8)
    history.append({"timeslot": 0, "R": R_current.copy()})

    print(f"\n--- Simulating {NUM_TIMESLOTS} timeslots ---\n")
    for t in range(1, NUM_TIMESLOTS + 1):
        F_used = endorsement_manager.endorsement_matrix.copy()

        interactions_t = generate_interactions_for_timeslot(nodes, t=t)
        for _, row in interactions_t.iterrows():
            interaction_manager.add_interaction(
                row["src"], row["dst"], row["rating"], row["timestamp"], row["month"]
            )

        P_cum = interaction_manager.pos_counts
        N_cum = interaction_manager.neg_counts

        trust_calc = TrustworthinessCalculator(P_cum, N_cum, epsilon=EPSILON)
        C_current = trust_calc.compute_local_trust()

        rep_forward = ReputationForwardPropagation(C_current, F_used)
        rep_forward.alpha = DEMO_ALPHA
        R_forward = rep_forward.compute_reputation(initial_reputation=R_current)

        total_neg = N_cum.sum(axis=0)
        total_pos = P_cum.sum(axis=0)

        ep = EndorsementPenalty(F_used, total_neg)
        er = EndorsementReward(F_used, total_pos)
        ep.beta = DEMO_BETA
        er.lambda_param = DEMO_LAMBDA
        ep.gamma = DEMO_GAMMA
        er.gamma = DEMO_GAMMA

        gN_signal = ep.compute_penalty_signal_gN() if hasattr(ep, "compute_penalty_signal_gN") else ep.compute_penalty_signal()
        rP_signal = er.compute_reward_signal_rP() if hasattr(er, "compute_reward_signal_rP") else er.compute_reward_signal()

        penalty_vec = ep.compute_penalty()
        reward_vec = er.compute_reward()

        rbp = ReputationBackwardPropagation(epsilon=EPSILON)
        R_corrected = rbp.apply_backward_propagation(R_forward, penalty_vec, reward_vec)
        R_target = normalizer.normalize(R_corrected)

        # Momentum smoothing: exponentially-weighted update toward the target.
        # Produces smooth trajectories instead of a large initial jump.
        R_new = DEMO_MOMENTUM * R_current + (1.0 - DEMO_MOMENTUM) * R_target
        R_new = normalizer.normalize(R_new)

        if not FIX_ENDORSEMENT_MATRIX:
            endorsement_manager.update_endorsements_with_penalty(gN_signal)
            endorsement_manager.update_endorsements_with_reward(rP_signal)
            np.fill_diagonal(endorsement_manager.endorsement_matrix, 0)
            endorsement_manager.normalize_all_rows()

        R_current = R_new
        history.append({"timeslot": t, "R": R_current.copy()})

        display_vec = to_display(R_current)
        display_vals = {n: display_vec[node_to_index[n]] for n in nodes}
        print(f"t={t:2d}  " + "  ".join(f"{n}={display_vals[n]:.3f}" for n in nodes))

    # ============ Save CSV ============
    rep_df = pd.DataFrame([h["R"] for h in history], columns=nodes)
    rep_df["timeslot"] = [h["timeslot"] for h in history]
    display_matrix = np.vstack([to_display(h["R"]) for h in history])
    for i, n in enumerate(nodes):
        rep_df[f"{n}_display"] = display_matrix[:, i]

    csv_path = os.path.join(OUTPUT_DIR, "reputation_over_time.csv")
    rep_df.to_csv(csv_path, index=False)
    print(f"\nReputation history saved to {csv_path}")

    # ============ Plot ============
    fig, ax = plt.subplots(figsize=(12, 7))

    style = {
        "A_good":          {"color": "#2E7D32", "linestyle": "-",  "marker": "o", "label": "A_good (80% positive feedback)"},
        "B_endorser_of_A": {"color": "#66BB6A", "linestyle": "--", "marker": "s", "label": "B_endorser_of_A (60% pos, rises via BERP)"},
        "C_bad":           {"color": "#C62828", "linestyle": "-",  "marker": "x", "label": "C_bad (30% positive feedback)"},
        "D_endorser_of_C": {"color": "#EF5350", "linestyle": "--", "marker": "v", "label": "D_endorser_of_C (60% pos, falls via BEPP)"},
        "E_baseline":      {"color": "#616161", "linestyle": ":",  "marker": "^", "label": "E_baseline (60% pos, no endorsements)"},
        "F_falling":       {"color": "#1976D2", "linestyle": "-.", "marker": "D", "label": "F (high start, unstable decline)"},
        "G_rising":        {"color": "#9C27B0", "linestyle": "-.", "marker": "P", "label": "G (low start, unstable rise)"},
    }

    for n in nodes:
        s = style[n]
        ax.plot(
            rep_df["timeslot"],
            rep_df[f"{n}_display"],
            color=s["color"],
            linestyle=s["linestyle"],
            marker=s["marker"],
            label=s["label"],
            linewidth=2.4,
            markersize=6,
        )

    # Reputation tier thresholds
    ax.axhspan(0.8, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.8 else 1.3,
               facecolor="#E8F5E9", alpha=0.4, zorder=0)
    ax.axhspan(0.5, 0.8, facecolor="#FFF8E1", alpha=0.4, zorder=0)
    ax.axhspan(0.0, 0.5, facecolor="#FFEBEE", alpha=0.4, zorder=0)
    ax.axhline(y=0.8, color="#388E3C", linestyle="-", linewidth=1, alpha=0.6)
    ax.axhline(y=0.5, color="#D32F2F", linestyle="-", linewidth=1, alpha=0.6)
    ax.text(NUM_TIMESLOTS, 0.82, "HIGH (>0.8)", fontsize=15, fontweight="bold",
            alpha=0.7, ha="right", color="#388E3C")
    ax.text(NUM_TIMESLOTS, 0.65, "MEDIUM (0.5-0.8)", fontsize=15, fontweight="bold",
            alpha=0.7, ha="right", color="#F57C00")
    ax.text(NUM_TIMESLOTS, 0.42, "LOW (<0.5)", fontsize=15, fontweight="bold",
            alpha=0.7, ha="right", color="#D32F2F")

    ax.set_title("Time-Evolving Reputation Dynamics under RepuLink",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Timeslot", fontsize=15, fontweight="bold")
    ax.set_ylabel("Reputation (display-scaled, initial = 0.8)",
                  fontsize=15, fontweight="bold")
    ax.set_xticks(range(0, NUM_TIMESLOTS + 1, 2))
    ax.set_xlim(0, NUM_TIMESLOTS)
    ax.tick_params(axis="both", labelsize=15)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    legend = ax.legend(loc="best", fontsize=15, framealpha=0.3)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    plot_path_png = os.path.join(OUTPUT_DIR, "reputation_evolution.png")
    plot_path_pdf = os.path.join(OUTPUT_DIR, "reputation_evolution.pdf")
    plt.savefig(plot_path_png, dpi=200)
    plt.savefig(plot_path_pdf)
    print(f"Plot saved to {plot_path_png}")
    print(f"Plot saved to {plot_path_pdf}")


if __name__ == "__main__":
    main()
