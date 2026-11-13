"""
Best response and exploitability, vectorised over many deals at once.

Idea
----
Fix a strategy profile sigma. A best response (BR) for player b maximises b's
expected payoff against sigma_{1-b}. Exploitability of sigma is

    expl(sigma) = ( u_0(BR_0, sigma_1) + u_1(sigma_0, BR_1) ) / 2,

which is 0 exactly at a Nash equilibrium (the game value cancels).

The BR is computed by backward induction over the PUBLIC tree (betting
histories and draw counts), processing a whole batch of N deals as NumPy
vectors. At each node every deal carries the opponent's reach probability.
At a BR decision node the deals are grouped by the BR player's information
set; for each group we pick the action with the largest reach-weighted
total value, given the (already optimal) choices further down the tree.
Because the BR player's keys have PERFECT recall (the post-draw key includes
the pre-draw key), this backward induction is exact for the deals supplied.

Two uses:
  * Exact: with every chance outcome and its probability supplied (Kuhn
    poker; 6 deals), this is the exact best-response value.
  * Five-card draw: chance outcomes are sampled. We FIT the BR policy on one
    sample of deals and EVALUATE that fixed policy on an independent sample.
    A fixed policy can do no better than the true best response, so the
    out-of-sample value is an unbiased estimate of a LOWER BOUND on
    exploitability (in the spirit of Lisy and Bowling, 2017). The BR player
    may observe the cards more finely than the strategy's own buckets, which
    tightens the bound.
"""

import numpy as np


class VGame:
    """Interface a game must provide (all per-deal quantities are arrays of
    length N). Nodes are any hashable public-state objects."""

    # (N,) probability of each deal (sums to 1)
    weights = None

    def root(self): ...
    # returns (N,) or scalar payoff to P0, or None
    def terminal_utility(self, node): ...
    def player(self, node): ...
    def actions(self, node): ...
    def child(self, node, a): ...
    # returns (N, A) probabilities for player(node)
    def strategy_probs(self, node, strategy): ...
    # returns (N,) int ids of the BR player's information set
    def br_key(self, node, mode): ...


def _group_argmax(keys, vals, w):
    """keys (N,), vals (A, N), w (N,) -> best action per group, and per deal.
    Ties go to the lowest-index action."""
    uniq, inv = np.unique(keys, return_inverse=True)
    A = vals.shape[0]
    tot = np.zeros((len(uniq), A))
    for a in range(A):
        tot[:, a] = np.bincount(inv, weights=w * vals[a], minlength=len(uniq))
    best = tot.argmax(axis=1)
    return uniq, best, best[inv], tot


def best_response(game, strategy, b, mode="bucket"):
    """Fit a BR for player b. Returns (per-deal values to b, policy).
    policy: {node: (group ids, best action per group, fallback action)}."""
    sign = 1.0 if b == 0 else -1.0
    policy = {}
    N = len(game.weights)

    def walk(node, reach):
        u = game.terminal_utility(node)
        if u is not None:
            return sign * np.broadcast_to(np.asarray(u, dtype=float), (N,))
        p, acts = game.player(node), game.actions(node)
        if p != b:
            probs = game.strategy_probs(node, strategy)
            v = np.zeros(N)
            for i, a in enumerate(acts):
                q = probs[:, i]
                if not q.any():
                    continue
                v += q * walk(game.child(node, a), reach * q)
            return v
        vals = np.stack([walk(game.child(node, a), reach) for a in acts])
        keys = game.br_key(node, mode)
        uniq, best, per_deal, tot = _group_argmax(keys, vals, game.weights * reach)
        fallback = int(tot.sum(axis=0).argmax())
        policy[node] = (uniq, best, fallback)
        return vals[per_deal, np.arange(N)]

    v = walk(game.root(), np.ones(N))
    return v, policy


def evaluate_policy(game, strategy, b, policy, mode="bucket"):
    """Per-deal payoff to player b when b plays the fixed `policy` and the
    opponent plays `strategy`. Information sets the policy never saw use the
    node's fallback action. Returns (values, share of reach on unseen keys)."""
    sign = 1.0 if b == 0 else -1.0
    N = len(game.weights)
    # [unseen reach, total reach] at BR nodes
    unseen = [0.0, 0.0]

    def walk(node, reach):
        u = game.terminal_utility(node)
        if u is not None:
            return sign * np.broadcast_to(np.asarray(u, dtype=float), (N,))
        p, acts = game.player(node), game.actions(node)
        if p != b:
            probs = game.strategy_probs(node, strategy)
            v = np.zeros(N)
            for i, a in enumerate(acts):
                q = probs[:, i]
                if not q.any():
                    continue
                v += q * walk(game.child(node, a), reach * q)
            return v
        keys = game.br_key(node, mode)
        if node in policy:
            uniq, best, fallback = policy[node]
            pos = np.searchsorted(uniq, keys)
            pos_c = np.minimum(pos, len(uniq) - 1)
            known = uniq[pos_c] == keys
            act = np.where(known, best[pos_c], fallback)
        else:
            # node never reached in fitting
            known = np.zeros(N, bool)
            act = np.zeros(N, int)
        w = game.weights * reach
        unseen[0] += w[~known].sum()
        unseen[1] += w.sum()
        v = np.zeros(N)
        for i, a in enumerate(acts):
            m = act == i
            if m.any():
                v[m] = walk(game.child(node, a), reach)[m]
        return v

    v = walk(game.root(), np.ones(N))
    return v, (unseen[0] / unseen[1] if unseen[1] > 0 else 0.0)


def strategy_value(game, strategy):
    """Per-deal payoff to P0 when both players follow `strategy` (exact over actions)."""
    N = len(game.weights)

    def walk(node):
        u = game.terminal_utility(node)
        if u is not None:
            return np.broadcast_to(np.asarray(u, dtype=float), (N,))
        probs = game.strategy_probs(node, strategy)
        v = np.zeros(N)
        for i, a in enumerate(game.actions(node)):
            q = probs[:, i]
            if q.any():
                v += q * walk(game.child(node, a))
        return v

    return walk(game.root())


def weighted_mean_se(v, w):
    """Mean and standard error of per-deal values (w uniform for samples)."""
    m = float(np.sum(w * v))
    n = len(v)
    se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, se
