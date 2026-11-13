"""Engine tests. Run with:  python -m pytest -q   (or  python tests/test_engine.py)"""

import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcd.cards import Deck
from fcd.game import (Config, keep_cards, legal_actions, round_contributions,
                      round_over)

APPENDIX_A_52 = {"Straight flush": 40, "Four of a kind": 624, "Full house": 3744,
                 "Flush": 5108, "Straight": 10200, "Three of a kind": 54912,
                 "Two pair": 123552, "One pair": 1098240, "High card": 1302540}


def test_52_card_counts_match_proposal_appendix():
    assert dict(Deck(13).category_counts) == APPENDIX_A_52


def test_reduced_deck_ranking_rule():
    assert Deck(13).order[-4:] == ["Flush", "Full house", "Four of a kind", "Straight flush"]
    # six-plus deck
    assert Deck(9).order[-4:] == ["Full house", "Flush", "Four of a kind", "Straight flush"]
    assert Deck(6).order[:4] == ["High card", "One pair", "Two pair", "Three of a kind"]


def test_scores_are_consistent_with_categories():
    d = Deck(6)
    for h in combinations(d.cards, 5):
        assert d.score(h) // 16 ** 5 == d.strength[d.category(h)]


def test_wheel_is_lowest_straight():
    # ranks 9 T J Q K A -> ids 0..5
    d = Deck(6)
    c = lambda r, s: r * 4 + s
    # A 9 T J Q
    wheel = [c(5, 0), c(0, 1), c(1, 2), c(2, 3), c(3, 0)]
    # T J Q K A
    top = [c(1, 0), c(2, 1), c(3, 2), c(4, 3), c(5, 0)]
    assert d.category(wheel) == d.category(top) == "Straight"
    assert d.score(top) > d.score(wheel)


def test_keep_cards():
    d = Deck(6)
    c = lambda r, s: r * 4 + s
    pair = [c(4, 0), c(4, 1), c(0, 2), c(1, 3), c(5, 2)]
    assert sorted(keep_cards(d, pair, 3)) == sorted([c(4, 0), c(4, 1)])
    flush_draw = [c(0, 0), c(2, 0), c(3, 0), c(5, 0), c(1, 2)]
    assert c(1, 2) not in keep_cards(d, flush_draw, 1)


def test_betting_rules():
    assert legal_actions("", 2) == ["k", "b"]
    assert legal_actions("b", 2) == ["f", "c", "r"]
    assert legal_actions("br", 2) == ["f", "c"]
    assert round_over("kk") and round_over("brc") and not round_over("kb")
    assert round_contributions("kbrc", 1) == [2, 2]
    assert round_contributions("bf", 2) == [2, 0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
