"""
Chance-sampled Counterfactual Regret Minimisation for five-card draw.

Each iteration samples one shuffle of the deck (both hands and the draw
cards), then walks every betting and drawing action for both players,
updating counterfactual regrets as in Zinkevich et al. (2007). Sampling the
chance outcomes is the "chance sampling" scheme of Lanctot et al. (2009).

Optional CFR+ (Tammelin, 2014): regrets floored at zero ("regret matching+")
and the average strategy weighted by iteration number ("linear averaging").
The two ingredients can be switched on separately to test which one matters.

Optional external sampling (Lanctot et al., 2009): each iteration samples a
deal, then for each player in turn walks all of that player's actions while
sampling one action at the opponent's nodes from the current strategy.
"""

import random
import time

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
    """sampling: "chance" (default) or "external".
    plus=True switches on both CFR+ ingredients; rm_plus / linear set them
    individually."""

    def __init__(self, cfg, seed=0, plus=False, rm_plus=None, linear=None,
                 sampling="chance"):
        self.cfg = cfg
        self.rm_plus = plus if rm_plus is None else rm_plus
        self.linear = plus if linear is None else linear
        self.plus = self.rm_plus and self.linear
        self.sampling = sampling
        self.rng = random.Random(seed)
        self.nodes = {}
        self.t = 0
        # node visits, a machine-independent cost
        self.touched = 0
        # time spent in iterations only
        self.train_seconds = 0.0
        # build the evaluator once
        cfg.deck.table()

    def iterate(self):
        self.t += 1
        order = self.cfg.deck.cards[:]
        self.rng.shuffle(order)
        deal = Deal(self.cfg, order)
        if self.sampling == "external":
            for i in (0, 1):
                self._walk_es(deal, "", "", "", i)
            return None
        return self._walk(deal, "", "", "", 1.0, 1.0)

    def train(self, n, callback=None, every=None):
        for _ in range(n):
            t0 = time.perf_counter()
            self.iterate()
            self.train_seconds += time.perf_counter() - t0
            if callback and every and self.t % every == 0:
                callback(self)

    def _node(self, deal, r1, d, r2):
        p, acts = node_actions(self.cfg, r1, d, r2)
        key = infoset_key(deal, r1, d, r2, p)
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = Node(len(acts))
        return p, acts, node

    def _walk_es(self, deal, r1, d, r2, i):
        """External sampling for traverser i; returns i's sampled value."""
        self.touched += 1
        u = terminal_utility(self.cfg, deal, r1, d, r2)
        if u is not None:
            return u if i == 0 else -u
        p, acts, node = self._node(deal, r1, d, r2)
        strat = node.strategy()
        if p == i:
            vals = [self._walk_es(deal, *child(r1, d, r2, a), i) for a in acts]
            v = sum(s * x for s, x in zip(strat, vals))
            for k in range(len(acts)):
                r = node.regret[k] + vals[k] - v
                node.regret[k] = max(r, 0.0) if self.rm_plus else r
            return v
        # opponent node: average-strategy update
        w = self.t if self.linear else 1.0
        for k in range(len(acts)):
            node.strat_sum[k] += w * strat[k]
        x, k = self.rng.random(), 0
        while k < len(acts) - 1 and x >= strat[k]:
            x -= strat[k]
            k += 1
        return self._walk_es(deal, *child(r1, d, r2, acts[k]), i)

    def _walk(self, deal, r1, d, r2, pi0, pi1):
        self.touched += 1
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
        w = self.t if self.linear else 1.0
        for i in range(len(acts)):
            r = node.regret[i] + opp * sign * (vals[i] - v)
            node.regret[i] = max(r, 0.0) if self.rm_plus else r
            node.strat_sum[i] += w * me * strat[i]
        return v

    def average_strategy(self):
        return {k: n.average() for k, n in self.nodes.items()}
