"""
Figures and summary tables for the final report and poster (greyscale).

  python make_final_plots.py

Reads results/br/*.json, results/*_curve.csv, results/*_analysis.json and
writes figures/final_*.png plus results/summary.json (every number the report
quotes, so it can be checked against the raw results).
"""

import csv
import glob
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
import math  # noqa: E402

R, FIG = "results", "figures"
SEEDS = (0, 1, 2)
# name: (colour, marker, linestyle)
STYLE = {
    "CFR": ("#000000", "o", "-"),
    "CFR+": ("#7f7f7f", "s", "--"),
    "RM+ only": ("#4d4d4d", "^", ":"),
    "Linear only": ("#a6a6a6", "D", "-."),
    "External sampling": ("#262626", "v", (0, (5, 1))),
}
VARIANTS = {"CFR": "cfr24", "CFR+": "cfrplus24", "RM+ only": "rmplus24",
            "Linear only": "linear24", "External sampling": "es24"}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def load_br(tag):
    p = f"{R}/br/{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def load_curve_csv(tag):
    p = f"{R}/{tag}_curve.csv"
    if not os.path.exists(p):
        return {}
    return {int(r["iteration"]): r for r in csv.DictReader(open(p))}


def seeds_of(prefix):
    return [s for s in SEEDS if load_br(f"{prefix}_s{s}")]


def curve_stats(prefix, x="iteration"):
    """Mean, min and max exploitability across seeds at each checkpoint."""
    by_x = {}
    for s in seeds_of(prefix):
        br = load_br(f"{prefix}_s{s}")
        rows = load_curve_csv(f"{prefix}_s{s}")
        for pt in br.get("curve", []):
            it = pt["iteration"]
            key = it if x == "iteration" else int(rows[it]["touched"])
            by_x.setdefault(it, []).append((key, pt["expl"]))
    out = []
    for it in sorted(by_x):
        xs = [k for k, _ in by_x[it]]
        ys = [v for _, v in by_x[it]]
        out.append((st.mean(xs), st.mean(ys), min(ys), max(ys), len(ys)))
    return out


def plot_line(ax, name, pts):
    if not pts:
        return
    c, m, ls = STYLE[name]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo = [p[1] - p[2] for p in pts]
    hi = [p[3] - p[1] for p in pts]
    ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=m, ls=ls, ms=4, capsize=2, label=name)


def plain_log_axes(ax):
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if str(f"{v:g}")[-1] in "25" else ""))


def slope(pts):
    """Least-squares slope of log(exploitability) against log(x)."""
    xs = [math.log(p[0]) for p in pts]
    ys = [math.log(p[1]) for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def fig_variants(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in ("CFR", "CFR+", "RM+ only", "Linear only"):
        plot_line(axes[0], name, curve_stats(VARIANTS[name]))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Exploitability lower bound (chips/hand)")
    axes[0].set_title("(a) CFR+ and its two ingredients")
    axes[0].legend()
    for name in ("CFR", "CFR+", "External sampling"):
        plot_line(axes[1], name, curve_stats(VARIANTS[name], x="touched"))
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Game-tree nodes visited (machine-independent cost)")
    axes[1].set_title("(b) Equal-cost comparison with external sampling")
    axes[1].legend()
    for ax in axes:
        plain_log_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/final_variants.png", dpi=170)
    plt.close(fig)
    for name, prefix in VARIANTS.items():
        pts = curve_stats(prefix)
        summary.setdefault("curves", {})[name] = pts
        summary.setdefault("curves_touched", {})[name] = curve_stats(prefix, x="touched")
        if len(pts) >= 2:
            summary.setdefault("slopes", {})[name] = slope(pts)


def final_table(summary):
    """Final exploitability and self-play value per configuration and seed."""
    tab = {}
    for path in sorted(glob.glob(f"{R}/br/*_s[0-9].json")):
        b = json.load(open(path))
        f = b["final"]
        tab[b["tag"]] = {"config": b["config"], "abstraction": b["abstraction"], "iters": b["iters"],
                         "expl": f["bucket"]["expl"], "expl_se": f["bucket"]["se"],
                         "in_sample": f["bucket"]["in_sample"],
                         "br_p0": f["bucket"]["br_value_p0"], "br_p1": f["bucket"]["br_value_p1"],
                         "selfplay_p0": f["selfplay_p0"]["mean"],
                         "selfplay_se": f["selfplay_p0"]["se"]}
        pkl = f"{R}/{b['tag']}_curve.csv"
        if os.path.exists(pkl):
            last = list(csv.DictReader(open(pkl)))[-1]
            tab[b["tag"]]["train_seconds"] = float(last["train_seconds"])
            tab[b["tag"]]["touched"] = int(last["touched"])
            tab[b["tag"]]["infosets"] = int(last["infosets"])
    summary["final"] = tab
    groups = {}
    for tag, row in tab.items():
        groups.setdefault(tag.rsplit("_s", 1)[0], []).append(row)
    summary["groups"] = {g: {"n": len(rows),
                             "expl_mean": st.mean(r["expl"] for r in rows),
                             "expl_min": min(r["expl"] for r in rows),
                             "expl_max": max(r["expl"] for r in rows),
                             "selfplay_mean": st.mean(r["selfplay_p0"] for r in rows),
                             "selfplay_min": min(r["selfplay_p0"] for r in rows),
                             "selfplay_max": max(r["selfplay_p0"] for r in rows)}
                         for g, rows in groups.items()}
    return tab


def fig_decks(summary):
    tab = summary["final"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = ["20-card", "24-card", "28-card"]
    prefixes = ["cfr20", "cfr24", "cfr28"]
    for i, pre in enumerate(prefixes):
        rows = [tab[f"{pre}_s{s}"] for s in SEEDS if f"{pre}_s{s}" in tab]
        for j, r in enumerate(rows):
            off = (j - 1) * 0.12
            axes[0].errorbar(i + off, r["expl"], yerr=1.96 * r["expl_se"], fmt="o", color="black",
                             capsize=3, ms=4)
            axes[1].errorbar(i + off, r["selfplay_p0"], yerr=1.96 * r["selfplay_se"], fmt="s",
                             color="black", capsize=3, ms=4)
    for ax in axes:
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, 2.5)
    axes[0].set_ylabel("Exploitability lower bound (chips/hand)")
    axes[0].set_title("(a) Exploitability by deck")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("P0 value in self-play (chips/hand)")
    axes[1].set_title("(b) First-player value by deck")
    fig.tight_layout()
    fig.savefig(f"{FIG}/final_decks.png", dpi=170)
    plt.close(fig)


def behaviour(summary):
    """Behaviour statistics across seeds (from `analyse`)."""
    out = {}
    for path in sorted(glob.glob(f"{R}/*_s[0-9]_analysis.json")):
        a = json.load(open(path))
        tag = os.path.basename(path).replace("_analysis.json", "")
        hc = a["draw_by_category"].get("High card", {})
        trips = a["draw_by_category"].get("Three of a kind", {})
        resp = a["p1_response_to_bet_by_p0_draw"]
        bets = a["first_bet_by_final_category"]
        out[tag] = {
            "vs_greedy": a["vs_greedy"]["mean"], "vs_greedy_se": a["vs_greedy"]["se"],
            "vs_random": a["vs_random"]["mean"], "vs_random_se": a["vs_random"]["se"],
            "pat_on_high_card": hc.get("0"), "one_to_trips": trips.get("1"),
            "fold_vs_pat": resp.get("0", {}).get("f"),
            "fold_vs_draw": {k: v.get("f") for k, v in resp.items() if k != "0"},
            "raise_vs_pat": resp.get("0", {}).get("r"),
            "raise_vs_draw": {k: v.get("r") for k, v in resp.items() if k != "0"},
            "bet_by_category": bets,
        }
    summary["behaviour"] = out


def fig_draws(tag="cfr24_s0"):
    p = f"{R}/{tag}_analysis.json"
    if not os.path.exists(p):
        # identical strategy (same seed and length)
        p = f"{R}/cfr24_analysis.json"
    a = json.load(open(p))
    order = ["High card", "One pair", "Two pair", "Three of a kind", "Straight", "Full house", "Flush"]
    order = [c for c in order if c in a["draw_by_category"]]
    shades = ["#262626", "#666666", "#a6a6a6", "#e0e0e0"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, cat in enumerate(order[::-1]):
        left = 0.0
        d = a["draw_by_category"][cat]
        for k in range(4):
            v = d.get(str(k), 0.0)
            ax.barh(i, v, left=left, color=shades[k], edgecolor="black", linewidth=0.4,
                    label=f"draw {k}" if i == 0 else None)
            if v >= 0.07:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if k < 2 else "black")
            left += v
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of decisions (reach-weighted, self-play)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/final_draws.png", dpi=170)
    plt.close(fig)


def fig_kuhn():
    p = f"{R}/kuhn_convergence.csv"
    if not os.path.exists(p):
        return
    rows = list(csv.DictReader(open(p)))
    t = [int(r["iteration"]) for r in rows]
    e = [float(r["exploitability"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.loglog(t, e, "o-", color="black", ms=3, label="CFR average strategy")
    c = e[0] * t[0] ** 0.5
    ax.loglog(t, [c / x ** 0.5 for x in t], "--", color="grey", label="O(1/sqrt(T)) reference")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Exact exploitability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG}/final_kuhn.png", dpi=170)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    summary = {}
    fig_variants(summary)
    final_table(summary)
    fig_decks(summary)
    behaviour(summary)
    fig_draws()
    fig_kuhn()
    with open(f"{R}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("figures written; summary in results/summary.json")


if __name__ == "__main__":
    main()
