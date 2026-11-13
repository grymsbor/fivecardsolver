"""Best-response tests. Run with:  python tests/test_br.py"""

import itertools
import os
import pickle
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import kuhn_cfr as K
from fcd.br import best_response, evaluate_policy, strategy_value
from fcd.evaluate import StrategyAgent, deal_ev, sample_deals
from fcd.game import Config
from fcd.vgame import DrawVGame, HandIndex, check_against_deal, deal_features


# ---------------- Kuhn poker: exact check ----------------

class KuhnVGame:
    """Kuhn poker in the vectorised interface: all 6 deals, weight 1/6 each."""

    def __init__(self, br_player=0):
        self.deals = K.DEALS
        self.weights = np.full(len(self.deals), 1 / len(self.deals))
        self.b = br_player

    def root(self):
        return ""

    def terminal_utility(self, h):
        if not K.is_terminal(h):
            return None
        return np.array([K.utility(h, c) for c in self.deals], float)

    def player(self, h):
        return K.player(h)

    def actions(self, h):
        return K.ACTIONS

    def child(self, h, a):
        return h + a

    def strategy_probs(self, h, strategy):
        return np.array([strategy[K.infoset(h, c)] for c in self.deals])

    def br_key(self, h, mode):
        return np.array([c[self.b] for c in self.deals])


def kuhn_exploitability(strategy):
    vals = []
    for b in (0, 1):
        g = KuhnVGame(b)
        v, _ = best_response(g, strategy, b)
        vals.append(float(np.sum(g.weights * v)))
    return sum(vals) / 2


def random_kuhn_profile(rng):
    out = {}
    for k in K.INFOSETS:
        x = rng.random()
        out[k] = [x, 1 - x]
    return out


def test_kuhn_br_matches_brute_force():
    rng = random.Random(0)
    profiles = [random_kuhn_profile(rng) for _ in range(30)]
    solver = K.KuhnCFR()
    for t in range(1, 2001):
        solver.iterate()
        if t in (1, 10, 100, 2000):
            profiles.append(solver.average_profile())
    for prof in profiles:
        assert abs(kuhn_exploitability(prof) - K.exploitability(prof)) < 1e-12


def test_kuhn_equilibrium_has_zero_exploitability():
    # any alpha in [0, 1/3]
    a = 0.2
    eq = {"J": [1 - a, a], "Q": [1, 0], "K": [1 - 3 * a, 3 * a],
          "Jpb": [1, 0], "Qpb": [1 - (a + 1 / 3), a + 1 / 3], "Kpb": [0, 1],
          "Jp": [2 / 3, 1 / 3], "Qp": [1, 0], "Kp": [0, 1],
          "Jb": [1, 0], "Qb": [2 / 3, 1 / 3], "Kb": [0, 1]}
    assert abs(kuhn_exploitability(eq)) < 1e-12
    g = KuhnVGame()
    assert abs(np.sum(g.weights * strategy_value(g, eq)) - K.GAME_VALUE) < 1e-12


# ---------------- Five-card draw: consistency checks ----------------

def _setup(n=300, seed=7):
    with open(os.path.join(ROOT, "results", "cfr24.pkl"), "rb") as f:
        saved = pickle.load(f)
    cfg, strat = saved["config"], saved["strategy"]
    ix = HandIndex(cfg)
    feats = deal_features(cfg, n, seed, ix)
    return cfg, strat, ix, feats


def test_features_match_scalar_game():
    cfg, _, ix, feats = _setup(n=50)
    assert all(check_against_deal(cfg, feats, ix, 7, j) for j in range(50))


def test_vector_value_matches_scalar_evaluator():
    cfg, strat, ix, feats = _setup(n=200)
    g = DrawVGame(cfg, feats, ix)
    v = strategy_value(g, strat)
    agent = StrategyAgent(strat)
    ref = [deal_ev(cfg, d, agent, agent) for d in sample_deals(cfg, 200, seed=7)]
    assert np.allclose(v, ref, atol=1e-9)


def test_br_beats_the_strategy_itself_and_policy_replays():
    cfg, strat, ix, feats = _setup(n=2000)
    selfplay = strategy_value(DrawVGame(cfg, feats, ix), strat)
    for b in (0, 1):
        g = DrawVGame(cfg, feats, ix, br_player=b)
        for mode in ("bucket", "fine"):
            v, pol = best_response(g, strat, b, mode)
            # The strategy's own play is one policy the BR could choose.
            own = selfplay if b == 0 else -selfplay
            assert v.mean() >= own.mean() - 1e-12
            # Replaying the fitted policy on the same deals reproduces it.
            v2, unseen = evaluate_policy(g, strat, b, pol, mode)
            assert np.allclose(v, v2) and unseen == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
