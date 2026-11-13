"""Method diagram for the poster: how the exploitability lower bound is computed."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
steps = [
    ("Sample deals", "$10^6$ shuffles for fitting,\n$3\\times10^5$ fresh ones for testing"),
    ("Walk the public tree once", "511 decisions; every node holds\na vector over all deals"),
    ("Opponent nodes", "look up the strategy for\nevery deal at once"),
    ("Best-responder nodes", "group deals by information set;\npick the best action per group"),
    ("Evaluate on fresh deals", "fixed policy, unbiased value:\na lower bound on exploitability"),
]
fig, ax = plt.subplots(figsize=(12, 2.6))
ax.set_xlim(0, len(steps) * 2.4)
ax.set_ylim(0, 2.2)
ax.axis("off")
for i, (title, body) in enumerate(steps):
    x = i * 2.4 + 0.1
    ax.add_patch(FancyBboxPatch((x, 0.2), 2.0, 1.8, boxstyle="round,pad=0.05",
                                facecolor="#f2f2f2" if i % 2 == 0 else "#d9d9d9",
                                edgecolor="black", linewidth=1))
    ax.text(x + 1.0, 1.6, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + 1.0, 0.85, body, ha="center", va="center", fontsize=9)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 2.35, 1.1), xytext=(x + 2.05, 1.1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "poster_method.png"), dpi=200)
print("wrote figures/poster_method.png")
