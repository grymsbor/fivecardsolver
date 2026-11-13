"""
Experiments.

  python run_experiments.py train   --ranks 6 --iters 150000 --tag cfr24
  python run_experiments.py train   --ranks 6 --iters 150000 --plus --tag cfrplus24
  python run_experiments.py analyse --tag cfr24
  python run_experiments.py exploit --tag cfr24 [--snapshots]

`train` writes results/<tag>.pkl (average strategy) and results/<tag>_curve.csv
(checkpoint evaluations); with --snapshots it also saves the average strategy
at every checkpoint in results/snap/<tag>/. `analyse` writes
results/<tag>_analysis.json. `exploit` writes results/br/<tag>.json with the
best-response lower bound on exploitability (see fcd/br.py).
"""

import argparse
import csv
import json
import os
import pickle
import time

import numpy as np

from fcd.br import best_response, evaluate_policy, strategy_value
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
    cfg = Config(n_ranks=args.ranks, abstraction="fine" if args.fine else "base")
    rm_plus = args.plus or args.rm_plus
    linear = args.plus or args.linear
    trainer = Trainer(cfg, seed=args.seed, rm_plus=rm_plus, linear=linear,
                      sampling="external" if args.external else "chance")
    deals = sample_deals(cfg, args.eval_deals, seed=10_000 + args.seed)
    rows, prev, start = [], {}, time.time()
    snapdir = f"{OUT}/snap/{args.tag}"
    if args.snapshots:
        os.makedirs(snapdir, exist_ok=True)

    def checkpoint(tr):
        strat = tr.average_strategy()
        if args.snapshots:
            with open(f"{snapdir}/{tr.t}.pkl", "wb") as f:
                pickle.dump(strat, f)
        row = {"iteration": tr.t, "seconds": round(time.time() - start, 1),
               "train_seconds": round(tr.train_seconds, 1), "touched": tr.touched,
               "infosets": len(tr.nodes)}
        agent = StrategyAgent(strat)
        if not args.no_eval:
            sp = self_play(cfg, agent, deals)
            vr = head_to_head(cfg, agent, RandomAgent(), deals)
            vg = head_to_head(cfg, agent, GreedyAgent(), deals)
            row.update({"selfplay_p0": sp["mean"], "selfplay_se": sp["se"],
                        "vs_random": vr["mean"], "vs_random_se": vr["se"],
                        "vs_greedy": vg["mean"], "vs_greedy_se": vg["se"]})
        row.update({"strategy_change": strategy_change(prev, strat),
                    "unseen_lookups": agent.unseen})
        rows.append(row)
        prev.clear()
        prev.update(strat)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    trainer.train(args.iters, checkpoint, args.every)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{args.tag}.pkl", "wb") as f:
        pickle.dump({"config": cfg, "plus": rm_plus and linear, "rm_plus": rm_plus,
                     "linear": linear, "sampling": trainer.sampling, "seed": args.seed,
                     "iters": args.iters, "train_seconds": trainer.train_seconds,
                     "touched": trainer.touched,
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


# ---------------- Exploitability (best-response lower bound) ----------------

FIT_SEED, EVAL_SEED = 1, 2


def features(cfg, n, seed):
    """Deal features, cached on disk per (deck, abstraction, n, seed).
    Only the arrays the bucket-key best response needs are cached."""
    from fcd.vgame import HandIndex, deal_features
    index = HandIndex(cfg)
    path = f"{OUT}/features/{cfg.label()}_{'fine' if cfg.fine else 'base'}_{n}_{seed}.npz"
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}, index
    feats = deal_features(cfg, n, seed, index)
    feats = {k: feats[k] for k in ("pre", "post", "bpre", "bpost", "res")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **feats)
    return feats, index


def exploitability(cfg, strat, fit, ev, index, modes=("bucket",)):
    """Lower bound on exploitability: BR policies fitted on `fit` deals,
    evaluated on independent `ev` deals. Returns a dict per BR mode."""
    from fcd.vgame import DrawVGame
    out = {}
    for mode in modes:
        per = []
        for b in (0, 1):
            vin, pol = best_response(DrawVGame(cfg, fit, index, b), strat, b, mode)
            vout, unseen = evaluate_policy(DrawVGame(cfg, ev, index, b), strat, b, pol, mode)
            per.append({"in_sample": float(vin.mean()), "value": float(vout.mean()),
                        "se": float(vout.std(ddof=1) / np.sqrt(len(vout))),
                        "unseen_reach": unseen, "vals": vout})
        # same deals: paired
        diff = (per[0]["vals"] + per[1]["vals"]) / 2
        out[mode] = {"expl": float(diff.mean()),
                     "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                     "in_sample": (per[0]["in_sample"] + per[1]["in_sample"]) / 2,
                     "br_value_p0": per[0]["value"], "br_value_p1": per[1]["value"],
                     "unseen_reach": [per[0]["unseen_reach"], per[1]["unseen_reach"]]}
    sv = strategy_value(DrawVGame(cfg, ev, index), strat)
    out["selfplay_p0"] = {"mean": float(sv.mean()), "se": float(sv.std(ddof=1) / np.sqrt(len(sv)))}
    return out


def exploit(args):
    with open(f"{OUT}/{args.tag}.pkl", "rb") as f:
        saved = pickle.load(f)
    cfg = saved["config"]
    fit, index = features(cfg, args.fit_deals, FIT_SEED)
    ev, _ = features(cfg, args.eval_deals, EVAL_SEED)
    os.makedirs(f"{OUT}/br", exist_ok=True)
    result = {"tag": args.tag, "config": cfg.label(), "abstraction": "fine" if cfg.fine else "base",
              "iters": saved["iters"], "fit_deals": args.fit_deals, "eval_deals": args.eval_deals,
              "final": exploitability(cfg, saved["strategy"], fit, ev, index)}
    print(args.tag, "final", round(result["final"]["bucket"]["expl"], 4),
          round(result["final"]["bucket"]["se"], 4), flush=True)
    snapdir = f"{OUT}/snap/{args.tag}"
    if args.snapshots and os.path.isdir(snapdir):
        curve = []
        have = sorted(int(x[:-4]) for x in os.listdir(snapdir))
        wanted = [int(x) for x in args.curve_iters.split(",")] if args.curve_iters else have
        for it in [i for i in have if i in wanted]:
            with open(f"{snapdir}/{it}.pkl", "rb") as f:
                strat = pickle.load(f)
            e = exploitability(cfg, strat, fit, ev, index, modes=(args.curve_mode,))
            curve.append({"iteration": it, "expl": e[args.curve_mode]["expl"],
                          "se": e[args.curve_mode]["se"]})
            print(args.tag, it, round(curve[-1]["expl"], 4), flush=True)
        result["curve_mode"] = args.curve_mode
        result["curve"] = curve
    with open(f"{OUT}/br/{args.tag}.json", "w") as f:
        json.dump(result, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--ranks", type=int, default=6)
    t.add_argument("--iters", type=int, default=150_000)
    t.add_argument("--every", type=int, default=10_000)
    t.add_argument("--eval-deals", type=int, default=2_000)
    t.add_argument("--plus", action="store_true", help="CFR+: both ingredients")
    t.add_argument("--rm-plus", action="store_true", help="regret floor only")
    t.add_argument("--linear", action="store_true", help="linear averaging only")
    t.add_argument("--external", action="store_true", help="external-sampling MCCFR")
    t.add_argument("--fine", action="store_true", help="finer card abstraction")
    t.add_argument("--snapshots", action="store_true")
    t.add_argument("--no-eval", action="store_true", help="skip baseline evals at checkpoints")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--tag", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--tag", required=True)
    a.add_argument("--deals", type=int, default=20_000)
    a.add_argument("--behaviour-deals", type=int, default=5_000)
    e = sub.add_parser("exploit")
    e.add_argument("--tag", required=True)
    e.add_argument("--fit-deals", type=int, default=1_000_000)
    e.add_argument("--eval-deals", type=int, default=300_000)
    e.add_argument("--snapshots", action="store_true")
    e.add_argument("--curve-mode", default="bucket")
    e.add_argument("--curve-iters", default="", help="comma-separated snapshot iterations")
    args = ap.parse_args()
    {"train": train, "analyse": analyse, "exploit": exploit}[args.cmd](args)


if __name__ == "__main__":
    main()
