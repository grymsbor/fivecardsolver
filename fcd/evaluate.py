"""
Evaluation: opponents, exact-over-actions expected value, behaviour stats.

For each sampled deal the expected value is computed EXACTLY over both
players' action probabilities (a full tree walk), so the only sampling noise
comes from the cards. Matches are played in both seats on the same deals to
cancel most of the card luck.
"""

import math
import random
from collections import defaultdict

from .cfr import child, infoset_key, node_actions, terminal_utility
from .game import Deal, round_over as round_done

# ---------------- Agents ----------------
# An agent maps (deal, r1, d, r2, player, actions) to probabilities over actions.


class StrategyAgent:
    """Plays a CFR average strategy; uniform at information sets never visited."""

    def __init__(self, strategy, name="CFR"):
        self.s, self.name = strategy, name
        self.unseen = 0

    def __call__(self, deal, r1, d, r2, p, acts):
        probs = self.s.get(infoset_key(deal, r1, d, r2, p))
        if probs is None:
            self.unseen += 1
            return [1.0 / len(acts)] * len(acts)
        return probs


class RandomAgent:
    name = "Random"

    def __call__(self, deal, r1, d, r2, p, acts):
        return [1.0 / len(acts)] * len(acts)


def _pick(acts, wanted):
    """Probability 1 on the first wanted action that is legal."""
    for w in wanted:
        if w in acts:
            return [1.0 if a == w else 0.0 for a in acts]
    return [1.0 / len(acts)] * len(acts)


class GreedyAgent:
    """Fixed-threshold baseline from the proposal: draws to the most likely
    improvement and bets or calls by hand strength. It never bluffs."""

    name = "Greedy"
    DRAW = {"High card": 3, "One pair": 3, "Two pair": 1, "Three of a kind": 2}

    def __call__(self, deal, r1, d, r2, p, acts):
        deck = deal.deck
        # draw decision
        if acts[0].isdigit():
            cat = deck.category(deal.hands[p])
            k = self.DRAW.get(cat, 0)
            if cat == "High card" and deal.pre(p).endswith("+d"):
                k = 1
            return _pick(acts, [str(k)])
        # before the draw
        if len(d) < 2:
            cat = deck.category(deal.hands[p])
            strong = deck.strength[cat] >= deck.strength["Two pair"]
            ok = strong or cat == "One pair" or deal.pre(p).endswith("+d")
        # after the draw
        else:
            (hands, _, _) = deal.final((int(d[0]), int(d[1])))
            cat = deck.category(hands[p])
            strong = deck.strength[cat] >= deck.strength["Two pair"]
            ok = strong or cat == "One pair"
        if strong:
            return _pick(acts, ["r", "b", "c"])
        if ok:
            return _pick(acts, ["k", "c"])
        return _pick(acts, ["k", "f"])


# ---------------- Expected value ----------------

def deal_ev(cfg, deal, agent0, agent1, r1="", d="", r2=""):
    """Expected payoff to P0 on one deal, exact over both agents' mixing."""
    u = terminal_utility(cfg, deal, r1, d, r2)
    if u is not None:
        return u
    p, acts = node_actions(cfg, r1, d, r2)
    probs = (agent0 if p == 0 else agent1)(deal, r1, d, r2, p, acts)
    total = 0.0
    for a, q in zip(acts, probs):
        if q > 0:
            total += q * deal_ev(cfg, deal, agent0, agent1, *child(r1, d, r2, a))
    return total


def sample_deals(cfg, n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        order = cfg.deck.cards[:]
        rng.shuffle(order)
        out.append(Deal(cfg, order))
    return out


def summarise(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    se = sd / math.sqrt(n)
    z = m / se if se > 0 else 0.0
    # two-sided, normal approx.
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"mean": m, "sd": sd, "se": se, "z": z, "p": p, "n": n}


def self_play(cfg, agent, deals):
    """P0's expected payoff when both seats play `agent`."""
    return summarise([deal_ev(cfg, dl, agent, agent) for dl in deals])


def head_to_head(cfg, a, b, deals):
    """`a`'s payoff per hand against `b`, averaged over both seats."""
    xs = [(deal_ev(cfg, dl, a, b) - deal_ev(cfg, dl, b, a)) / 2 for dl in deals]
    return summarise(xs)


# ---------------- Behaviour of a strategy ----------------

def behaviour(cfg, agent, deals):
    """Reach-weighted action frequencies in self-play:
    - draw counts by pre-draw category,
    - first post-draw bet by final category (P0 first to act; P1 after a check),
    - P1's response to a post-draw bet, split by how many cards P0 drew."""
    draws = defaultdict(lambda: defaultdict(float))
    # [weight, bet weight]
    bets = defaultdict(lambda: [0.0, 0.0])
    resp = defaultdict(lambda: defaultdict(float))

    def walk(deal, r1, d, r2, reach):
        if reach < 1e-9 or terminal_utility(cfg, deal, r1, d, r2) is not None:
            return
        p, acts = node_actions(cfg, r1, d, r2)
        probs = agent(deal, r1, d, r2, p, acts)
        if round_done(r1) and len(d) < 2:
            cat = deal.deck.category(deal.hands[p])
            for a, q in zip(acts, probs):
                draws[cat][int(a)] += reach * q
        elif len(d) == 2 and r2 in ("", "k") and len(r2) == p:
            hands, _, _ = deal.final((int(d[0]), int(d[1])))
            cat = deal.deck.category(hands[p])
            bets[cat][0] += reach
            bets[cat][1] += reach * probs[acts.index("b")]
        elif len(d) == 2 and r2 == "b" and p == 1:
            for a, q in zip(acts, probs):
                resp[int(d[0])][a] += reach * q
        for a, q in zip(acts, probs):
            if q > 0:
                walk(deal, *child(r1, d, r2, a), reach * q)

    for dl in deals:
        walk(dl, "", "", "", 1.0)
    return draws, bets, resp
