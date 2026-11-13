"""
Vanilla Counterfactual Regret Minimisation (CFR) for Kuhn poker,
with exact exploitability via brute-force best response.

Kuhn poker (Kuhn, 1950):
  - Deck of three cards: J < Q < K. Each player antes 1 and gets one card.
  - Player 0 acts first. Actions: 'p' (pass = check/fold), 'b' (bet/call), bet size 1.
  - Terminal histories: pp, bp, bb, pbp, pbb.

Known equilibrium family (player 0 = first to act):
  - bet J with probability alpha in [0, 1/3]
  - bet K with probability 3*alpha
  - with Q after "pass, bet", call with probability alpha + 1/3
  - game value for player 0 = -1/18

Reference for the algorithm: Neller & Lanctot (2013), Zinkevich et al. (2007).

Usage:
  python cfr_kuhn.py                 # 100,000 iterations, writes convergence.csv
  python cfr_kuhn.py --iters 20000 --csv out.csv
"""

import argparse
import csv
import itertools
import time

# J, Q, K
CARDS = [0, 1, 2]
CARD_NAMES = "JQK"
ACTIONS = ["p", "b"]
TERMINALS = {"pp", "bp", "bb", "pbp", "pbb"}
GAME_VALUE = -1 / 18
# 6 equally likely deals
DEALS = list(itertools.permutations(CARDS, 2))


def is_terminal(h):
    return h in TERMINALS


def utility(h, cards):
    """Payoff to player 0 at terminal history h."""
    # player 1 folds
    if h == "bp":
        return 1
    # player 0 folds
    if h == "pbp":
        return -1
    stake = 2 if h in ("bb", "pbb") else 1
    return stake if cards[0] > cards[1] else -stake


def player(h):
    return len(h) % 2


def infoset(h, cards):
    return CARD_NAMES[cards[player(h)]] + h


# All 12 information sets (6 per player).
INFOSETS = [CARD_NAMES[c] + h for c in CARDS for h in ["", "pb"]] + \
           [CARD_NAMES[c] + h for c in CARDS for h in ["p", "b"]]


class Node:
    def __init__(self):
        self.regret_sum = [0.0, 0.0]
        self.strategy_sum = [0.0, 0.0]

    def strategy(self):
        """Regret matching: play in proportion to positive regret."""
        pos = [max(r, 0.0) for r in self.regret_sum]
        total = sum(pos)
        return [x / total for x in pos] if total > 0 else [0.5, 0.5]

    def average_strategy(self):
        total = sum(self.strategy_sum)
        return [x / total for x in self.strategy_sum] if total > 0 else [0.5, 0.5]


class KuhnCFR:
    def __init__(self):
        self.nodes = {}

    def node(self, key):
        if key not in self.nodes:
            self.nodes[key] = Node()
        return self.nodes[key]

    def cfr(self, cards, h, reach0, reach1):
        """Returns expected utility for player 0 from history h.
        reach0/reach1 are each player's own contribution to reaching h."""
        if is_terminal(h):
            return utility(h, cards)

        p = player(h)
        nd = self.node(infoset(h, cards))
        strat = nd.strategy()
        my_reach, opp_reach = (reach0, reach1) if p == 0 else (reach1, reach0)

        action_values = [0.0, 0.0]
        for i, a in enumerate(ACTIONS):
            if p == 0:
                action_values[i] = self.cfr(cards, h + a, reach0 * strat[i], reach1)
            else:
                action_values[i] = self.cfr(cards, h + a, reach0, reach1 * strat[i])
        node_value = sum(s * v for s, v in zip(strat, action_values))

        # Utilities are from player 0's view; flip sign for player 1.
        sign = 1 if p == 0 else -1
        for i in range(2):
            regret = sign * (action_values[i] - node_value)
            # counterfactual regret
            nd.regret_sum[i] += opp_reach * regret
            # reach-weighted average
            nd.strategy_sum[i] += my_reach * strat[i]
        return node_value

    def iterate(self):
        """One full-tree iteration, enumerating all chance outcomes."""
        total = 0.0
        for cards in DEALS:
            total += self.cfr(cards, "", 1.0, 1.0)
        return total / len(DEALS)

    def average_profile(self):
        return {k: n.average_strategy() for k, n in self.nodes.items()}


# ---------- Evaluation: expected value and exploitability ----------

def expected_value(profile, h="", cards=None):
    """EV for player 0 when both players follow `profile` (infoset -> [p, b])."""
    if cards is None:
        return sum(expected_value(profile, "", c) for c in DEALS) / len(DEALS)
    if is_terminal(h):
        return utility(h, cards)
    strat = profile[infoset(h, cards)]
    return sum(strat[i] * expected_value(profile, h + a, cards)
               for i, a in enumerate(ACTIONS))


def best_response_value(profile, br_player):
    """Exact best-response value for br_player against the profile.
    Each player has only 6 information sets, so we enumerate all
    2^6 = 64 pure strategies and take the best."""
    keys = [k for k in INFOSETS if player(k[1:]) == br_player]
    best = float("-inf")
    for choice in itertools.product([0, 1], repeat=len(keys)):
        mixed = dict(profile)
        for k, c in zip(keys, choice):
            mixed[k] = [1.0, 0.0] if c == 0 else [0.0, 1.0]
        v = expected_value(mixed)
        best = max(best, v if br_player == 0 else -v)
    return best


def exploitability(profile):
    """Average gain of a best responder over the game value.
    Zero exactly at a Nash equilibrium."""
    return (best_response_value(profile, 0) + best_response_value(profile, 1)) / 2


# ---------- Main ----------

def print_strategy(profile):
    print("\nAverage strategy (probability of 'b' = bet/call):")
    labels = {"": "P0 opens", "p": "P1 after check", "b": "P1 facing bet",
              "pb": "P0 facing bet after check"}
    for h in ["", "p", "b", "pb"]:
        row = "  ".join(f"{c}: {profile[c + h][1]:.3f}" for c in CARD_NAMES)
        print(f"  {labels[h]:<27} {row}")


def check_equilibrium(profile, ev):
    alpha = profile["J"][1]
    checks = [
        ("alpha = P(bet J) in [0, 1/3]", alpha, (0, 1 / 3)),
        ("P(bet K) = 3*alpha", profile["K"][1], 3 * alpha),
        ("P(call Q | pb) = alpha + 1/3", profile["Qpb"][1], alpha + 1 / 3),
        ("P(bet Q opening) = 0", profile["Q"][1], 0.0),
        ("P1: bet K after check = 1", profile["Kp"][1], 1.0),
        ("P1: call K facing bet = 1", profile["Kb"][1], 1.0),
        ("P1: fold J facing bet = 1", profile["Jb"][0], 1.0),
        ("P1: bluff J after check = 1/3", profile["Jp"][1], 1 / 3),
        ("P1: call Q facing bet = 1/3", profile["Qb"][1], 1 / 3),
        ("game value = -1/18", ev, GAME_VALUE),
    ]
    print("\nChecks against the known equilibrium:")
    for name, got, want in checks:
        if isinstance(want, tuple):
            ok = want[0] - 1e-3 <= got <= want[1] + 1e-3
            want_s = f"[{want[0]:.3f}, {want[1]:.3f}]"
        else:
            ok = abs(got - want) < 1e-2
            want_s = f"{want:.4f}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<32} got {got:.4f}, want {want_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--csv", default="convergence.csv")
    args = ap.parse_args()

    solver = KuhnCFR()
    log_points = sorted({int(10 ** (e / 4)) for e in range(0, 40)
                         if int(10 ** (e / 4)) <= args.iters} | {args.iters})
    rows = []
    start = time.time()
    for t in range(1, args.iters + 1):
        solver.iterate()
        if t in log_points:
            prof = solver.average_profile()
            ex = exploitability(prof)
            rows.append((t, time.time() - start, expected_value(prof), ex))
            print(f"iter {t:>8}  exploitability {ex:.6f}")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "seconds", "ev_player0", "exploitability"])
        w.writerows(rows)

    prof = solver.average_profile()
    ev = expected_value(prof)
    print(f"\nEV for player 0: {ev:.5f} (theory {GAME_VALUE:.5f})")
    print(f"Exploitability:  {exploitability(prof):.6f}")
    print_strategy(prof)
    check_equilibrium(prof, ev)
    print(f"\nConvergence log written to {args.csv}")


if __name__ == "__main__":
    main()
