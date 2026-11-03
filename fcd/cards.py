"""
Cards, reduced decks and five-card hand evaluation.

A deck has `n_ranks` ranks (0 = lowest, n_ranks-1 = ace) and `n_suits` suits.
Card id = rank * n_suits + suit. A 24-card deck (6 ranks) is 9 T J Q K A.

Hand ranking rule (from our proposal): the standard order is kept for
three of a kind, two pair, one pair and high card, while the five "made"
categories (straight and above) are re-ordered by rarity in the chosen deck,
computed by exhaustive enumeration. Ties in rarity keep the standard order.
"""

from collections import Counter
from functools import lru_cache
from itertools import combinations

STANDARD_ORDER = ["High card", "One pair", "Two pair", "Three of a kind", "Straight",
                  "Flush", "Full house", "Four of a kind", "Straight flush"]   # weakest first
MADE = ["Straight", "Flush", "Full house", "Four of a kind", "Straight flush"]
RANK_CHARS = "23456789TJQKA"


class Deck:
    def __init__(self, n_ranks=6, n_suits=4):
        if not 5 <= n_ranks <= 13:
            raise ValueError("n_ranks must be between 5 and 13")
        self.n_ranks, self.n_suits = n_ranks, n_suits
        self.size = n_ranks * n_suits
        self.cards = list(range(self.size))
        self.straights = self._straight_sets()
        self.category_counts = self._count_categories()
        self.order = self._ranking()                     # weakest first
        self.strength = {c: i for i, c in enumerate(self.order)}
        self._table = None

    # --- basic helpers ---
    def rank(self, c):
        return c // self.n_suits

    def suit(self, c):
        return c % self.n_suits

    def name(self, c):
        return RANK_CHARS[13 - self.n_ranks + self.rank(c)] + "cdhs"[self.suit(c)]

    def hand_str(self, hand):
        return " ".join(self.name(c) for c in sorted(hand, reverse=True))

    def _straight_sets(self):
        n = self.n_ranks
        s = {frozenset(range(lo, lo + 5)): lo + 4 for lo in range(n - 4)}
        if n > 5:
            s[frozenset([n - 1, 0, 1, 2, 3])] = 3
        return s

    def category(self, hand):
        ranks = [self.rank(c) for c in hand]
        counts = sorted(Counter(ranks).values(), reverse=True)
        flush = len({self.suit(c) for c in hand}) == 1
        straight = len(counts) == 5 and frozenset(ranks) in self.straights
        if straight and flush:
            return "Straight flush"
        if counts[0] == 4:
            return "Four of a kind"
        if counts[:2] == [3, 2]:
            return "Full house"
        if flush:
            return "Flush"
        if straight:
            return "Straight"
        if counts[0] == 3:
            return "Three of a kind"
        if counts[:2] == [2, 2]:
            return "Two pair"
        if counts[0] == 2:
            return "One pair"
        return "High card"

    def _count_categories(self):
        return Counter(self.category(h) for h in combinations(self.cards, 5))

    def _ranking(self):
        made = sorted(MADE, key=lambda c: (-self.category_counts[c], STANDARD_ORDER.index(c)))
        return STANDARD_ORDER[:4] + made                 # rarest made hand last (strongest)

    # --- full evaluation with tie-breaks ---
    def evaluate(self, hand):
        """Integer score; higher wins. Ties are exact splits."""
        cat = self.category(hand)
        ranks = [self.rank(c) for c in hand]
        if cat in ("Straight", "Straight flush"):
            kickers = [self.straights[frozenset(ranks)]]
        else:
            cnt = Counter(ranks)
            kickers = sorted(cnt, key=lambda r: (cnt[r], r), reverse=True)
        score = self.strength[cat]
        for r in kickers + [0] * (5 - len(kickers)):
            score = score * 16 + r
        return score

    def table(self):
        """Lookup table: sorted hand tuple -> score, built once per deck."""
        if self._table is None:
            self._table = {h: self.evaluate(h) for h in combinations(self.cards, 5)}
        return self._table

    def score(self, hand):
        return self.table()[tuple(sorted(hand))]


@lru_cache(maxsize=None)
def make_deck(n_ranks=6, n_suits=4):
    return Deck(n_ranks, n_suits)
