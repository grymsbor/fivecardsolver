"""
Experiments for the progression check.

  python run_experiments.py train   --ranks 6 --iters 150000 --tag cfr24
  python run_experiments.py train   --ranks 6 --iters 150000 --plus --tag cfrplus24
  python run_experiments.py analyse --tag cfr24
  python run_experiments.py plots

`train` writes results/<tag>.pkl (average strategy) and results/<tag>_curve.csv
(checkpoint evaluations). `analyse` writes results/<tag>_analysis.json.
"""

import argparse
import csv
import json
import os
import pickle
import time

from fcd.cfr import Trainer
from fcd.evaluate import (GreedyAgent, RandomAgent, StrategyAgent, behaviour,
                          head_to_head, sample_deals, self_play)
from fcd.game import Config

OUT = "results"


def strategy_change(old, new):
    """Mean total-variation distance between two average strategies,
    over information sets present in both."""
    common = [k for k in new if k in old]
    if not common:
        return float("nan")
    return sum(0.5 * sum(abs(a - b) for a, b in zip(old[k], new[k])) for k in common) / len(common)


def train(args):
    cfg = Config(n_ranks=args.ranks)
    trainer = Trainer(cfg, seed=args.seed, plus=args.plus)
    deals = sample_deals(cfg, args.eval_deals, seed=10_000 + args.seed)
    rows, prev, start = [], {}, time.time()

    def checkpoint(tr):
        strat = tr.average_strategy()
        agent = StrategyAgent(strat)
        sp = self_play(cfg, agent, deals)
        vr = head_to_head(cfg, agent, RandomAgent(), deals)
        vg = head_to_head(cfg, agent, GreedyAgent(), deals)
        row = {"iteration": tr.t, "seconds": round(time.time() - start, 1),
               "infosets": len(tr.nodes), "selfplay_p0": sp["mean"], "selfplay_se": sp["se"],
               "vs_random": vr["mean"], "vs_random_se": vr["se"],
               "vs_greedy": vg["mean"], "vs_greedy_se": vg["se"],
               "strategy_change": strategy_change(prev, strat),
               "unseen_lookups": agent.unseen}
        rows.append(row)
        prev.clear()
        prev.update(strat)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    trainer.train(args.iters, checkpoint, args.every)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{args.tag}.pkl", "wb") as f:
        pickle.dump({"config": cfg, "plus": args.plus, "iters": args.iters,
                     "strategy": trainer.average_strategy()}, f)
    with open(f"{OUT}/{args.tag}_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyse(args):
    with open(f"{OUT}/{args.tag}.pkl", "rb") as f:
        saved = pickle.load(f)
    cfg = saved["config"]
    agent = StrategyAgent(saved["strategy"])
    deals = sample_deals(cfg, args.deals, seed=99)
    t0 = time.time()
    result = {"config": cfg.label(), "iters": saved["iters"], "plus": saved["plus"],
              "infosets": len(saved["strategy"]), "eval_deals": args.deals,
              "ranking_weakest_first": cfg.deck.order,
              "selfplay_p0": self_play(cfg, agent, deals),
              "vs_random": head_to_head(cfg, agent, RandomAgent(), deals),
              "vs_greedy": head_to_head(cfg, agent, GreedyAgent(), deals),
              "greedy_selfplay_p0": self_play(cfg, GreedyAgent(), deals)}
    draws, bets, resp = behaviour(cfg, agent, deals[: args.behaviour_deals])
    norm = lambda d: {str(k): v / sum(d.values()) for k, v in sorted(d.items())}
    result["draw_by_category"] = {c: norm(v) for c, v in draws.items()}
    result["first_bet_by_final_category"] = {c: v[1] / v[0] for c, v in bets.items() if v[0] > 0}
    result["p1_response_to_bet_by_p0_draw"] = {str(k): norm(v) for k, v in sorted(resp.items())}
    result["unseen_lookups"] = agent.unseen
    result["analysis_seconds"] = round(time.time() - t0, 1)
    with open(f"{OUT}/{args.tag}_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--ranks", type=int, default=6)
    t.add_argument("--iters", type=int, default=150_000)
    t.add_argument("--every", type=int, default=10_000)
    t.add_argument("--eval-deals", type=int, default=2_000)
    t.add_argument("--plus", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--tag", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--tag", required=True)
    a.add_argument("--deals", type=int, default=20_000)
    a.add_argument("--behaviour-deals", type=int, default=5_000)
    args = ap.parse_args()
    {"train": train, "analyse": analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
