"""Figures for the progression check, from results/*.csv and *_analysis.json.

  python make_plots.py      # writes figures/*.png
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "results", "figures"
GREY, BLUE, LIGHT = "#9a9a9a", "#2a6fdb", "#c9d8f5"


def curve(tag):
    with open(f"{R}/{tag}_curve.csv") as f:
        return list(csv.DictReader(f))


def analysis(tag):
    with open(f"{R}/{tag}_analysis.json") as f:
        return json.load(f)


def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for tag, name, col in [("cfr24", "CFR", GREY), ("cfrplus24", "CFR+", BLUE)]:
        rows = curve(tag)
        t = [int(r["iteration"]) for r in rows]
        g = [float(r["vs_greedy"]) for r in rows]
        se = [float(r["vs_greedy_se"]) for r in rows]
        axes[0].errorbar(t, g, yerr=[1.96 * s for s in se], marker="o", ms=3, capsize=2, color=col, label=name)
        ch = [float(r["strategy_change"]) for r in rows[1:]]
        axes[1].plot(t[1:], ch, marker="o", ms=3, color=col, label=name)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Chips per hand vs Greedy (95% CI)")
    axes[0].set_title("(a) Winnings against the greedy baseline")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Mean strategy change per 10k iterations")
    axes[1].set_title("(b) Change in the average strategy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Figure 1: Training on the 24-card deck")
    fig.tight_layout()
    fig.savefig(f"{F}/fig1_training.png", dpi=150)


def fig_position():
    tags = [("cfr20", "20-card"), ("cfr24", "24-card"), ("cfr28", "28-card")]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (tag, name) in enumerate(tags):
        sp = analysis(tag)["selfplay_p0"]
        ax.errorbar(i, sp["mean"], yerr=1.96 * sp["se"], fmt="o", color=BLUE, capsize=5)
        ax.annotate(f"{sp['mean']:+.3f}", (i, sp["mean"]), xytext=(10, 0), textcoords="offset points", va="center")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(tags)), [n for _, n in tags])
    ax.set_xlim(-0.5, len(tags) - 0.4)
    ax.set_ylabel("First player's chips per hand (95% CI)")
    ax.set_title("Figure 2: First-player value in self-play, by deck size")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{F}/fig2_position.png", dpi=150)


def fig_draws():
    a = analysis("cfr24")
    order = [c for c in ["High card", "One pair", "Two pair", "Three of a kind", "Straight",
                         "Full house", "Flush"] if c in a["draw_by_category"]]
    shades = ["#1f4fa0", "#4d7fd0", "#8fb0e6", "#d3e0f6"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    left = [0.0] * len(order)
    for k in range(4):
        vals = [a["draw_by_category"][c].get(str(k), 0.0) for c in order]
        ax.barh(order, vals, left=left, color=shades[k], label=f"draw {k}")
        for i, v in enumerate(vals):
            if v > 0.07:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if k < 2 else "black")
        left = [l + v for l, v in zip(left, vals)]
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of decisions (reach-weighted, self-play)")
    ax.set_title("Figure 3: How many cards the 24-card CFR strategy draws, by starting hand")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.38), frameon=False)
    fig.tight_layout()
    fig.savefig(f"{F}/fig3_draws.png", dpi=150)


if __name__ == "__main__":
    os.makedirs(F, exist_ok=True)
    fig_convergence()
    fig_position()
    fig_draws()
    print("figures written to", F)
