"""
Five-card draw as a vectorised game for fcd.br (best response / exploitability).

A batch of N sampled deals is turned into integer feature arrays once:
for each player, the pre-draw bucket and, for every pair of draw counts
(k0, k1), the post-draw bucket and the showdown result. After that, every
operation on the public tree is a NumPy operation over all N deals.

The best-responding player observes its own cards through a perfect-recall
key ("bucket": the BASE card buckets, whatever abstraction the strategy uses,
with the post-draw key also remembering the pre-draw bucket; "fine": rank
multiset and largest suit count before the draw, plus final rank multiset and
flush flag after it). Using the base buckets for every strategy keeps the
best responder equally informed when abstractions are compared.
"""

import random
from collections import Counter
from itertools import combinations

import numpy as np

from .cfr import child as _child
from .cfr import node_actions
from .game import (Deal, contributions, fold_utility, keep_cards, post_bucket,
                   pre_bucket, round_over)


def _fine_pre(deck, hand):
    return (tuple(sorted(deck.rank(c) for c in hand)),
            max(Counter(deck.suit(c) for c in hand).values()))


def _fine_post(deck, hand):
    return (tuple(sorted(deck.rank(c) for c in hand)),
            len({deck.suit(c) for c in hand}) == 1)


class HandIndex:
    """Integer ids for every 5-card hand's buckets under one config."""

    def __init__(self, cfg):
        deck = cfg.deck
        self.pre_names, self.post_names = [], []
        pre_id, post_id, fpre_id, fpost_id, bpre_id, bpost_id = {}, {}, {}, {}, {}, {}
        self.hand = {}
        for h in combinations(deck.cards, 5):
            ids = []
            for name, table, names in ((pre_bucket(deck, h, cfg.fine), pre_id, self.pre_names),
                                       (post_bucket(deck, h, cfg.fine), post_id, self.post_names),
                                       (_fine_pre(deck, h), fpre_id, None),
                                       (_fine_post(deck, h), fpost_id, None),
                                       (pre_bucket(deck, h), bpre_id, None),
                                       (post_bucket(deck, h), bpost_id, None)):
                if name not in table:
                    table[name] = len(table)
                    if names is not None:
                        names.append(name)
                ids.append(table[name])
            # pre, post, fpre, fpost, bpre, bpost, score
            self.hand[h] = (*ids, deck.score(h))
        self.n_pre, self.n_post = len(pre_id), len(post_id)
        self.n_fpre, self.n_fpost = len(fpre_id), len(fpost_id)
        self.n_bpre, self.n_bpost = len(bpre_id), len(bpost_id)


def deal_features(cfg, n, seed, index=None):
    """Feature arrays for n deals shuffled with `seed`."""
    deck = cfg.deck
    index = index or HandIndex(cfg)
    D = len(cfg.draw_options)
    pre = np.zeros((2, n), np.int32)
    fpre = np.zeros((2, n), np.int32)
    bpre = np.zeros((2, n), np.int32)
    post = np.zeros((2, D, D, n), np.int32)
    fpost = np.zeros((2, D, D, n), np.int32)
    bpost = np.zeros((2, D, D, n), np.int32)
    res = np.zeros((D, D, n), np.int8)
    rng = random.Random(seed)
    look = index.hand
    for j in range(n):
        order = deck.cards[:]
        rng.shuffle(order)
        hands = (order[0:5], order[5:10])
        for p in (0, 1):
            e = look[tuple(sorted(hands[p]))]
            pre[p, j], fpre[p, j], bpre[p, j] = e[0], e[2], e[4]
        keep = [[keep_cards(deck, hands[p], k) for k in cfg.draw_options] for p in (0, 1)]
        for i0, k0 in enumerate(cfg.draw_options):
            e0 = look[tuple(sorted(keep[0][i0] + order[10:10 + k0]))]
            for i1, k1 in enumerate(cfg.draw_options):
                e1 = look[tuple(sorted(keep[1][i1] + order[10 + k0:10 + k0 + k1]))]
                post[0, i0, i1, j], fpost[0, i0, i1, j], bpost[0, i0, i1, j] = e0[1], e0[3], e0[5]
                post[1, i0, i1, j], fpost[1, i0, i1, j], bpost[1, i0, i1, j] = e1[1], e1[3], e1[5]
                res[i0, i1, j] = (e0[6] > e1[6]) - (e0[6] < e1[6])
    return {"pre": pre, "fpre": fpre, "bpre": bpre, "post": post, "fpost": fpost,
            "bpost": bpost, "res": res}


class DrawVGame:
    """Public tree of five-card draw over a batch of feature arrays.
    br_player decides which player's keys br_key returns."""

    def __init__(self, cfg, feats, index, br_player=0):
        self.cfg, self.f, self.ix, self.b = cfg, feats, index, br_player
        n = feats["pre"].shape[1]
        self.weights = np.full(n, 1.0 / n)
        self.didx = {str(k): i for i, k in enumerate(cfg.draw_options)}
        self._tables = {}

    # ----- tree -----
    def root(self):
        return ("", "", "")

    def player(self, node):
        return node_actions(self.cfg, *node)[0]

    def actions(self, node):
        return node_actions(self.cfg, *node)[1]

    def child(self, node, a):
        return _child(*node, a)

    def terminal_utility(self, node):
        r1, d, r2 = node
        if r1.endswith("f"):
            return fold_utility(self.cfg, r1, "", (len(r1) - 1) % 2)
        if r2.endswith("f"):
            return fold_utility(self.cfg, r1, r2, (len(r2) - 1) % 2)
        if len(d) == 2 and round_over(r2):
            stake = contributions(self.cfg, r1, r2)[0]
            return stake * self.f["res"][self.didx[d[0]], self.didx[d[1]]].astype(float)
        return None

    # ----- the strategy being evaluated -----
    def _ids(self, node, p):
        r1, d, r2 = node
        if not round_over(r1) or len(d) < 2:
            return self.f["pre"][p]
        return self.f["post"][p, self.didx[d[0]], self.didx[d[1]]]

    def strategy_probs(self, node, strategy):
        p, acts = node_actions(self.cfg, *node)
        r1, d, r2 = node
        key = (id(strategy), node)
        table = self._tables.get(key)
        if table is None:
            if not round_over(r1):
                names, fmt = self.ix.pre_names, lambda b: f"1|{b}|{r1}"
            elif len(d) < 2:
                names, fmt = self.ix.pre_names, lambda b: f"D|{b}|{r1}|{d}"
            else:
                names, fmt = self.ix.post_names, lambda b: f"2|{b}|{r1}/{d}/{r2}"
            uniform = [1.0 / len(acts)] * len(acts)
            table = np.array([strategy.get(fmt(b), uniform) for b in names])
            self._tables[key] = table
        return table[self._ids(node, p)]

    # ----- the best-responding player's information -----
    def br_key(self, node, mode):
        p = self.b
        r1, d, r2 = node
        before = not round_over(r1) or len(d) < 2
        if mode == "bucket":
            if before:
                return self.f["bpre"][p]
            post = self.f["bpost"][p, self.didx[d[0]], self.didx[d[1]]]
            return self.f["bpre"][p].astype(np.int64) * self.ix.n_bpost + post
        if before:
            return self.f["fpre"][p]
        post = self.f["fpost"][p, self.didx[d[0]], self.didx[d[1]]]
        return self.f["fpre"][p].astype(np.int64) * self.ix.n_fpost + post


def subset(feats, idx):
    """Feature arrays restricted to the deals in idx (deal axis is last)."""
    return {k: v[..., idx] for k, v in feats.items()}


def check_against_deal(cfg, feats, index, seed, j=0):
    """Rebuild deal j with the scalar game code and compare its features."""
    rng = random.Random(seed)
    for _ in range(j + 1):
        order = cfg.deck.cards[:]
        rng.shuffle(order)
    deal = Deal(cfg, order)
    ok = all(index.pre_names[feats["pre"][p, j]] == deal.pre(p) for p in (0, 1))
    for i0, k0 in enumerate(cfg.draw_options):
        for i1, k1 in enumerate(cfg.draw_options):
            _, scores, buckets = deal.final((k0, k1))
            r = (scores[0] > scores[1]) - (scores[0] < scores[1])
            ok &= feats["res"][i0, i1, j] == r
            ok &= all(index.post_names[feats["post"][p, i0, i1, j]] == buckets[p] for p in (0, 1))
    return bool(ok)
