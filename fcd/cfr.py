"""
Chance-sampled Counterfactual Regret Minimisation for five-card draw.

Each iteration samples one shuffle of the deck (both hands and the draw
cards), then walks every betting and drawing action for both players,
updating counterfactual regrets as in Zinkevich et al. (2007). Sampling the
chance outcomes is the "chance sampling" scheme of Lanctot et al. (2009).

Optional CFR+ (Tammelin, 2014): regrets floored at zero and the average
strategy weighted by iteration number.
"""

import random

from .game import (Deal, fold_utility, legal_actions, round_over,
                   showdown_utility)


def infoset_key(deal, r1, d, r2, player):
    """The information set a player is in, under the card abstraction."""
    if not round_over(r1):
        return f"1|{deal.pre(player)}|{r1}"
    if len(d) < 2:
        return f"D|{deal.pre(player)}|{r1}|{d}"
    _, _, buckets = deal.final((int(d[0]), int(d[1])))
    return f"2|{buckets[player]}|{r1}/{d}/{r2}"


def node_actions(cfg, r1, d, r2):
    """(player to act, legal actions) at a non-terminal node."""
    if not round_over(r1):
        return len(r1) % 2, legal_actions(r1, cfg.cap)
    if len(d) < 2:
        return len(d), [str(k) for k in cfg.draw_options]
    return len(r2) % 2, legal_actions(r2, cfg.cap)


def child(r1, d, r2, a):
    if not round_over(r1):
        return r1 + a, d, r2
    if len(d) < 2:
        return r1, d + a, r2
    return r1, d, r2 + a


def terminal_utility(cfg, deal, r1, d, r2):
    """Payoff to P0, or None if the node is not terminal."""
    if r1.endswith("f"):
        return fold_utility(cfg, r1, "", (len(r1) - 1) % 2)
    if r2.endswith("f"):
        return fold_utility(cfg, r1, r2, (len(r2) - 1) % 2)
    if len(d) == 2 and round_over(r2):
        _, scores, _ = deal.final((int(d[0]), int(d[1])))
        return showdown_utility(cfg, r1, r2, scores)
    return None


class Node:
    __slots__ = ("regret", "strat_sum")

    def __init__(self, n):
        self.regret = [0.0] * n
        self.strat_sum = [0.0] * n

    def strategy(self):
        pos = [r if r > 0 else 0.0 for r in self.regret]
        s = sum(pos)
        n = len(pos)
        return [x / s for x in pos] if s > 0 else [1.0 / n] * n

    def average(self):
        s = sum(self.strat_sum)
        n = len(self.strat_sum)
        return [x / s for x in self.strat_sum] if s > 0 else [1.0 / n] * n


class Trainer:
    def __init__(self, cfg, seed=0, plus=False):
        self.cfg, self.plus = cfg, plus
        self.rng = random.Random(seed)
        self.nodes = {}
        self.t = 0
        cfg.deck.table()                       # build the evaluator once

    def iterate(self):
        self.t += 1
        order = self.cfg.deck.cards[:]
        self.rng.shuffle(order)
        deal = Deal(self.cfg, order)
        return self._walk(deal, "", "", "", 1.0, 1.0)

    def train(self, n, callback=None, every=None):
        for _ in range(n):
            self.iterate()
            if callback and every and self.t % every == 0:
                callback(self)

    def _walk(self, deal, r1, d, r2, pi0, pi1):
        u = terminal_utility(self.cfg, deal, r1, d, r2)
        if u is not None:
            return u
        p, acts = node_actions(self.cfg, r1, d, r2)
        key = infoset_key(deal, r1, d, r2, p)
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = Node(len(acts))
        strat = node.strategy()

        vals = [0.0] * len(acts)
        for i, a in enumerate(acts):
            c1, cd, c2 = child(r1, d, r2, a)
            if p == 0:
                vals[i] = self._walk(deal, c1, cd, c2, pi0 * strat[i], pi1)
            else:
                vals[i] = self._walk(deal, c1, cd, c2, pi0, pi1 * strat[i])
        v = sum(s * x for s, x in zip(strat, vals))

        sign = 1.0 if p == 0 else -1.0
        opp, me = (pi1, pi0) if p == 0 else (pi0, pi1)
        w = self.t if self.plus else 1.0
        for i in range(len(acts)):
            r = node.regret[i] + opp * sign * (vals[i] - v)
            node.regret[i] = max(r, 0.0) if self.plus else r
            node.strat_sum[i] += w * me * strat[i]
        return v

    def average_strategy(self):
        return {k: n.average() for k, n in self.nodes.items()}
